'''
PYTHONPATH = '.' python ./tools/analysis_tools/bevmix/benchmark_trt_bevmix.py \
configs/bevmix/quant/detsegVAD-640x1152-vov99+r50-BEV256+128-2+9kf-stereo-circle60e-loaddet-segupupper-lsqp88.py \
work_dirs/quant_detsegVAD_640_bevmix_diffbev_symmetric/epoch_8.pth \
mmdeploy/detsegVAD-640x1152-vov99+r50-BEV256+128-2+9kf-stereo-circle60e-loaddet-segupupper \
bevmix_int8.engine \
--w-channel-wise --w-symmetric --a-symmetric --w-quantizer=LSQPlusQuantizer --a-quantizer=LSQPlusQuantizer \
--samples=6019 --postprocessing --eval
'''

import time
from typing import Dict, Optional, Sequence, Union

import pickle
import logging

from mmdeploy.backend.tensorrt.utils import save, search_cuda_version


import os

import numpy as np
import onnx
import pycuda.driver as cuda

import tensorrt as trt
import torch
import torch.onnx
import torch.nn.functional as F
import mmcv 
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdeploy.backend.tensorrt import load_tensorrt_plugin

try:
    # If mmdet version > 2.23.0, compat_cfg would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg

import argparse

from mmdet3d.core import bbox3d2result
from mmdet3d.core.bbox.structures.box_3d_mode import LiDARInstance3DBoxes
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model

from qfactory.utils import transform_layers
import torch.nn as nn
from qfactory.configs import QConfig
from qfactory.modules import QConv2d
from qfactory.quantizers import EMAQuantizer, LSQPlusQuantizer

def parse_args():
    parser = argparse.ArgumentParser(description='Deploy BEVDet with Tensorrt')
    parser.add_argument('config', help='deploy config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('work_dir', help='dir to save engine and hyper data file')
    parser.add_argument('engine', help='checkpoint file')
    parser.add_argument(
        '--w-bit',
        type=int,
        default=8,
        help='weight quant to how many bytes of int'
    )
    parser.add_argument(
        '--w-channel-wise',
        action='store_true',
        help='weight quant use channel wise quant, else use layer wise quant'
    )
    parser.add_argument(
        '--w-symmetric',
        action='store_true',
        help='weight quant use symmetric quant, else use asymmetric quant'
    )
    parser.add_argument(
        '--a-bit',
        type=int,
        default=8,
        help='feature quant to how many bytes of int'
    )
    parser.add_argument(
        '--a-channel-wise',
        action='store_true',
        help='feature quant use channel wise quant, else use layer wise quant'
    )
    parser.add_argument(
        '--a-symmetric',
        action='store_true',
        help='feature quant use symmetric quant, else use asymmetric quant'
    )   
    parser.add_argument(
        '--w-quantizer',
        type=str,
        default='EMAQuantizer',
        help='the algorithm to execute weight quant'
    )
    parser.add_argument(
        '--a-quantizer',
        type=str,
        default='EMAQuantizer',
        help='the algorithm to execute feature quant'
    )   
    parser.add_argument(
        '--transform-ignore-namess',
        type=str,
        default=None,
        help='state the module that you do not want to be quanted, if have more than one module, seperate them by ,'
    )
    parser.add_argument('--samples', default=500, type=int, help='samples to benchmark')
    parser.add_argument('--postprocessing', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--prefetch', action='store_true',
                        help='use prefetch to accelerate the data loading, '
                             'the inference speed is sightly degenerated due '
                             'to the computational occupancy of prefetch')
    args = parser.parse_args()
    return args


def torch_dtype_from_trt(dtype: trt.DataType) -> torch.dtype:
    """Convert pytorch dtype to TensorRT dtype.

    Args:
        dtype (str.DataType): The data type in tensorrt.

    Returns:
        torch.dtype: The corresponding data type in torch.
    """

    if dtype == trt.bool:
        return torch.bool
    elif dtype == trt.int8:
        return torch.int8
    elif dtype == trt.int32:
        return torch.int32
    elif dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    else:
        raise TypeError(f'{dtype} is not supported by torch')


class TRTWrapper(torch.nn.Module):

    def __init__(self,
                 engine: Union[str, trt.ICudaEngine],
                 output_names: Optional[Sequence[str]] = None) -> None:
        super().__init__()
        self.engine = engine
        if isinstance(self.engine, str):
            with trt.Logger() as logger, trt.Runtime(logger) as runtime:
                with open(self.engine, mode='rb') as f:
                    engine_bytes = f.read()
                self.engine = runtime.deserialize_cuda_engine(engine_bytes)
        self.context = self.engine.create_execution_context()
        import ipdb;ipdb.set_trace()
        names = [_ for _ in self.engine]
        input_names = list(filter(self.engine.binding_is_input, names))
        self._input_names = input_names
        self._output_names = output_names

        if self._output_names is None:
            output_names = list(set(names) - set(input_names))
            self._output_names = output_names

    def forward(self, inputs: Dict[str, torch.Tensor]):
        import ipdb;ipdb.set_trace()
        bindings = [None] * (len(self._input_names) + len(self._output_names))
        for input_name, input_tensor in inputs.items():
            idx = self.engine.get_binding_index(input_name)
            self.context.set_binding_shape(idx, tuple(input_tensor.shape))
            bindings[idx] = input_tensor.contiguous().data_ptr()

            # create output tensors
        outputs = {}
        for output_name in self._output_names:
            idx = self.engine.get_binding_index(output_name)
            dtype = torch_dtype_from_trt(self.engine.get_binding_dtype(idx))
            shape = tuple(self.context.get_binding_shape(idx))

            device = torch.device('cuda')
            output = torch.zeros(size=shape, dtype=dtype, device=device)
            outputs[output_name] = output
            bindings[idx] = output.data_ptr()
        self.context.execute_async_v2(bindings,
                                      torch.cuda.current_stream().cuda_stream)
        return outputs


def get_plugin_names():
    return [pc.name for pc in trt.get_plugin_registry().plugin_creator_list]


def create_dataset(data_path):
    try:
        with open(data_path,'rb') as f:
            data_dict = pickle.load(f)
            dataset = []
            for idx in range(len(data_dict['imgs'])):
                dataset.append(dict(
                    imgs=data_dict['imgs'][idx], 
                    sensor2keyegos=data_dict['sensor2keyegos'][idx],
                    ego2globals=data_dict['ego2globals'][idx],
                    intrins=data_dict['intrins'][idx],
                    post_rots=data_dict['post_rots'][idx],
                    post_trans=data_dict['post_trans'][idx],
                    bda=data_dict['bda'][idx],
                    curr2adjsensor=data_dict['curr2adjsensor'][idx],
                    feat_prev=data_dict['feat_prev'][idx],
                    gt_masks_bev=data_dict['gt_masks_bev'][idx]
                    ))
            return dataset, data_dict['input_shapes']

    except:
        logging.error(f"{data_path}no such hyper data file.")
        raise FileNotFoundError


def main():

    load_tensorrt_plugin()

    args = parse_args()
    if args.eval:
        args.postprocessing=True
        print('Warnings: evaluation requirement detected, set '
              'postprocessing=True for evaluation purpose')
    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.model.type = cfg.model.type + 'TRT'
    cfg = compat_cfg(cfg)
    cfg.gpu_ids = [0]

    if not args.prefetch:
        cfg.data.test_dataloader.workers_per_gpu=0

    # build dataloader
    assert cfg.data.test.test_mode
    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=False, shuffle=False)
    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))

    # set quant cfg
    w_bit = args.w_bit
    a_bit = args.a_bit
    w_channel_wise = False
    a_channel_wise = False
    w_symmetric = False
    a_symmetric = False
    if args.w_channel_wise:
        w_channel_wise = True
    if args.a_channel_wise:
        a_channel_wise = True
    if args.w_symmetric:
        w_symmetric = True
    if args.a_symmetric:
        a_symmetric = True
    # use QConfig as default
    w_qconfig = QConfig(bit=w_bit, channel_wise=w_channel_wise, symmetric=w_symmetric)
    a_qconfig = QConfig(bit=a_bit, channel_wise=a_channel_wise, symmetric=a_symmetric)
    w_quantizer = None
    a_quantizer = None
    if args.w_quantizer == 'EMAQuantizer':
        w_quantizer = EMAQuantizer
    elif args.w_quantizer == 'LSQPlusQuantizer':
        w_quantizer = LSQPlusQuantizer
    else:
        raise NameError
    if args.a_quantizer == 'EMAQuantizer':
        a_quantizer = EMAQuantizer
    elif args.a_quantizer == 'LSQPlusQuantizer':
        a_quantizer = LSQPlusQuantizer
    else:
        raise NameError
    # quant conv2d as default
    qconv = QConv2d

    transform_ignore_namess = []
    if args.transform_ignore_namess:
        transform_ignore_namess = args.transform_ignore_namess.split(',') # 'backbone.conv1', 'bbox_head.retina_cls', 'bbox_head.retina_reg', not quant the start and the end of the model may improve the res

    if 'with_hop' in cfg:
        transform_ignore_namess.append('aux_bbox_head')
        transform_ignore_namess.append('history_decoder')
    transform_layers(
        model,
        qconv=qconv,
        w_quantizer=w_quantizer,
        w_qconfig=w_qconfig,
        a_quantizer=a_quantizer,
        a_qconfig=a_qconfig,
        ignore_names=transform_ignore_namess)

    # 训练参数别忘了加呀！！！！！！！
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    # build tensorrt model
    trt_model = TRTWrapper(os.path.join(args.work_dir, args.engine),
                           [f'output_{i}' for i in
                            range(6 * len(model.pts_bbox_head.task_heads) + (1 if model.pts_seg_head else 0))]+['cur_feat'])

    device0 = torch.device('cuda',0)
    model = model.to(device0) 
    model.eval()
    
    num_warmup = 100
    pure_inf_time = 0

    init_ = True
    metas = dict()
    # benchmark with several samples and take the average
    results = list()
    # data_path_list = ['bevmix_hyper_data_1.pkl']
    # data_path_list = ['bevmix_hyper_data_0.pkl', 'bevmix_hyper_data_1.pkl', 'bevmix_hyper_data_2.pkl',
    # 'bevmix_hyper_data_3.pkl', 'bevmix_hyper_data_4.pkl', 'bevmix_hyper_data_5.pkl', 'bevmix_hyper_data_6.pkl',
    # 'bevmix_hyper_data_7.pkl', 'bevmix_hyper_data_8.pkl', 'bevmix_hyper_data_9.pkl', 'bevmix_hyper_data_10.pkl',
    # 'bevmix_hyper_data_11.pkl', 'bevmix_hyper_data_12.pkl', 'bevmix_hyper_data_13.pkl']
    # i = 0
    prog_bar = mmcv.ProgressBar(args.samples)

    for data_path in data_path_list:
        data_loader, _ = create_dataset(os.path.join(args.work_dir, data_path))

        for data in data_loader:
            with torch.no_grad():
                metas = dict(
                    sensor2keyegos=data['sensor2keyegos'].to(device0).contiguous(),
                    ego2globals=data['ego2globals'].to(device0).contiguous(),
                    intrins=data['intrins'].to(device0).contiguous(),
                    post_rots=data['post_rots'].to(device0).contiguous(),
                    post_trans=data['post_trans'].to(device0).contiguous(),
                    bda=data['bda'].to(device0).contiguous(),
                    curr2adjsensor=data['curr2adjsensor'].to(device0).contiguous(),
                    feat_prev=data['feat_prev'].to(device0).contiguous(),
                    gt_masks_bev=data['gt_masks_bev'].to(device0).contiguous())

                imgs = data['imgs'].to(device0).contiguous()
                torch.cuda.synchronize()
                start_time = time.perf_counter()
                trt_output = trt_model.forward(dict(imgs=imgs, **metas))
                
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start_time
                # cur_feat = trt_output['cur_feat']
                # bev_feat_list.append(cur_feat)
                # if len(bev_feat_list) > 8:
                #     bev_feat_list.pop(0)

                if args.eval:
                    cur_result = dict()
                # postprocessing
                if args.postprocessing:
                    trt_output = [trt_output[f'output_{i}'] for i in
                                range(6 * len(model.pts_bbox_head.task_heads) + (1 if model.pts_seg_head else 0))]
                    pred = model.result_deserialize(trt_output)
                    pred_bbox = pred[:len(pred)-1]
                    pred_seg = pred[-1]
                    img_metas = [dict(box_type_3d=LiDARInstance3DBoxes)]
                    if model.pts_bbox_head: # assert True
                        bbox_list = model.pts_bbox_head.get_bboxes(
                            pred_bbox, img_metas, rescale=True)
                        bbox_results = [
                            bbox3d2result(bboxes, scores, labels)
                            for bboxes, scores, labels in bbox_list
                        ]
                        if args.eval:
                            cur_result['pts_bbox'] = bbox_results[0] # 为什么只返回0，其实这里bbox_results只有一个元素

                    if model.pts_seg_head: # 有分割任务 
                        gt_masks_bev = [t.to(device0) for t in data['gt_masks_bev']]
                        if args.eval:
                            cur_result['pts_seg'] = pred_seg
                            cur_result['gt_masks_bev'] = metas['gt_masks_bev'][0]

                if args.eval:
                    results.append(cur_result)
                
                prog_bar.update()


            if i >= num_warmup:
                pure_inf_time += elapsed
                if (i + 1) % 50 == 0:
                    fps = (i + 1 - num_warmup) / pure_inf_time
                    print(f'Done image [{i + 1:<3}/ {args.samples}], '
                        f'fps: {fps:.2f} img / s')
            if (i + 1) == args.samples:
                pure_inf_time += elapsed
                fps = (i + 1 - num_warmup) / pure_inf_time
                print(f'Overall \nfps: {fps:.2f} img / s '
                    f'\ninference time: {1000/fps:.2f} ms')
                if not args.eval:
                    return
                    
            i += 1

    assert args.eval
    eval_kwargs = cfg.get('evaluation', {}).copy()
    # hard-code way to remove EvalHook args
    for key in [
        'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
        'rule'
    ]:
        eval_kwargs.pop(key, None)
    eval_kwargs.update(dict(metric='bbox'))
    # visualize
    # eval_kwargs.update(dict(out_dir='vis_results'))
    # eval_kwargs.update(dict(show=True))
    print(dataset.evaluate(results, **eval_kwargs))


if __name__ == '__main__':
    fps = main()
