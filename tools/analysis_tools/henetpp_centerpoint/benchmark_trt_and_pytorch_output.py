import time
from typing import Dict, Optional, Sequence, Union
import ctypes
import os

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
from collections import defaultdict

import argparse
import pickle
from mmdet3d.core import bbox3d2result
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet3d.core.bbox.structures.box_3d_mode import LiDARInstance3DBoxes
import numpy as np

'''
henetpp_bev:
PYTHONPATH='.' python ./tools/analysis_tools/henetpp_centerpoint/benchmark_trt_and_pytorch_output.py \
configs/henetpp/changan_rc_multitask_small_res_deploy_centerpoint.py \
work_dirs/changan_finetune/epoch_24.pth \
mmdeploy/changan_centerpoint/changan_static_fp16_fuse.engine \
--samples 1600 --eval --postprocessing
'''
def parse_args():
    parser = argparse.ArgumentParser(description='Deploy Henetpp with Tensorrt')
    parser.add_argument('config', help='deploy config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('engine', help='checkpoint file')
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
        names = [_ for _ in self.engine]
        input_names = list(filter(self.engine.binding_is_input, names))
        self._input_names = input_names
        self._output_names = output_names

        if self._output_names is None:
            output_names = list(set(names) - set(input_names))
            self._output_names = output_names

    def forward(self, inputs: Dict[str, torch.Tensor]):
        bindings = [None] * (len(self._input_names) + len(self._output_names))
        for input_name, input_tensor in inputs.items():
            # idx = self.engine.get_binding_index(input_name)
            # self.context.set_binding_shape(idx, tuple(input_tensor.shape))
            self.context.set_input_shape(input_name, tuple(input_tensor.shape))
            # bindings[idx] = input_tensor.contiguous().data_ptr()
            bindings[self.engine[input_name]] = input_tensor.contiguous().data_ptr()

            # create output tensors
        outputs = {}
        for output_name in self._output_names:
            # idx = self.engine.get_binding_index(output_name)
            # dtype = torch_dtype_from_trt(self.engine.get_binding_dtype(idx))
            # shape = tuple(self.context.get_binding_shape(idx))
            dtype = torch_dtype_from_trt(self.engine.get_tensor_dtype(output_name))
            shape = tuple(self.context.get_tensor_shape(output_name))

            device = torch.device('cuda')
            output = torch.zeros(size=shape, dtype=dtype, device=device)
            outputs[output_name] = output
            # bindings[idx] = output.data_ptr()
            bindings[self.engine[output_name]] = output.data_ptr()

        # self.context.profiler = MyProfiler()   # 层耗时输出
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        self.context.execute_async_v2(bindings,
                                      torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time
        # import pickle
        # pickle.dump(self.context.profiler.layer_time, open('vis/layer_time_henetpp_int8.pkl', 'wb'))
        # exit(0)
        return outputs, elapsed


def get_plugin_names():
    return [pc.name for pc in trt.get_plugin_registry().plugin_creator_list]


def main():
    load_tensorrt_plugin()
    print(get_plugin_names())

    soFile_3 = '/home/xiazhongyu/Desktop/bevperception/tools/deploy_tools/bevpoolv2_plugin/lib/bevpoolv2.so'
    success_3 = ctypes.CDLL(soFile_3, mode = ctypes.RTLD_GLOBAL)

    if success_3 == False:
        print('Failed to load bevpoolv2 plugin!')

    args = parse_args()
    if args.eval:
        args.postprocessing=True
        print('Warnings: evaluation requirement detected, set '
              'postprocessing=True for evaluation purpose')
    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None

    cfg = compat_cfg(cfg)
    cfg.gpu_ids = [0]

    # build dataloader
    if not args.prefetch:
        cfg.data.test_dataloader.workers_per_gpu=0

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
    # 训练参数别忘了加呀！！！！！！！
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    model = model.cuda()
    model.eval()
    
    # build tensorrt model
    cfg.model.type = cfg.model.type + 'TRT'
    # cfg.model.ret_2d_feat = True
    trt_model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(trt_model, args.checkpoint, map_location='cpu')
    trt_model = trt_model.cuda()
    trt_model.eval()

    fp16_trt_model = TRTWrapper(args.engine,
                        #    ['batch_reg_preds', 'batch_cls_preds', 'batch_cls_labels','occ_res'])
                            [f'output_{j}' for j in
                                range(6 * len(model.pts_bbox_head.task_heads))] + ['occ_res'])
                            # ['img_feat_1', 'img_feat_2'])

    # fp32_trt_model = TRTWrapper(args.engine.replace('_fp16', ''),
    #                     #    ['batch_reg_preds', 'batch_cls_preds', 'batch_cls_labels','occ_res'])
    #                         ['feat_2d_0'])
    pure_inf_time = []

    # benchmark with several samples and take the average
    results = []

    prog_bar = mmcv.ProgressBar(len(data_loader))

    with torch.no_grad():   
        for i, data in enumerate(data_loader):
            inputs = [t.cuda() for t in data['img_inputs'][0]]
            img_metas = [t for t in data['img_metas'][0]._data[0]]
            radar = [t for t in data['radar'][0]._data[0]]

            # for bevdepth
            imgs_list, mlp_input_list, metas_list, sensor2keyegos, ego2globals, bda = trt_model.get_bev_pool_input(inputs)
            
            # calculate prev feat
            imgs = torch.cat(imgs_list, dim=0) # 注意，使用img的时候需要squeeze(0)而使用mlp_input的时候不需要 这是二者的差别. torch.Size([1, 6, 3, 256, 704]) * 9
            mlp_input = torch.cat(mlp_input_list, dim=0) # torch.Size([1, 6, 27]) * 9
            feat_prev = trt_model.get_bev_feat_sequential(img=inputs, img_metas=img_metas)
            feat_prev = feat_prev.unsqueeze(0)

            img = imgs_list[0]
            mlp_input = mlp_input_list[0]
            metas = metas_list[0]
            ranks_depth = metas[1].int().contiguous()
            ranks_feat = metas[2].int().contiguous()
            ranks_bev = metas[0].int().contiguous()
            interval_starts = metas[3].int().contiguous()
            interval_lengths = metas[4].int().contiguous()
            pad_size_ranks = 314500 - ranks_bev.size(0)
            ranks_bev = F.pad(ranks_bev, (0, pad_size_ranks), "constant", 0)
            ranks_depth = F.pad(ranks_depth, (0, pad_size_ranks), "constant", 0)
            ranks_feat = F.pad(ranks_feat, (0, pad_size_ranks), "constant", 0)
            pad_size_interval = 54840 - interval_starts.size(0)
            interval_starts = F.pad(interval_starts, (0, pad_size_interval), "constant", 0)
            interval_lengths = F.pad(interval_lengths, (0, pad_size_interval), "constant", 0)

            ranks_depth = ranks_depth.unsqueeze(0)
            ranks_feat = ranks_feat.unsqueeze(0)
            ranks_bev = ranks_bev.unsqueeze(0)
            interval_starts = interval_starts.unsqueeze(0)
            interval_lengths = interval_lengths.unsqueeze(0)
            
            # for radar
            voxels, num_points, coors = trt_model.radar_voxelize(radar)

            pad_size = 262 - voxels.size(0)
            voxels = F.pad(voxels, (0, 0, 0, 0, 0, pad_size), "constant", 0)
            num_points = F.pad(num_points, (0, pad_size), "constant", 0)
            coors = F.pad(coors, (0, 0, 0, pad_size), "constant", 0)

            voxels = voxels.unsqueeze(0).cuda().contiguous()
            num_points = num_points.unsqueeze(0).cuda().contiguous()
            coors = coors.unsqueeze(0).cuda().contiguous()
            # benchmark trt model
            # img = torch.rand(6, 3, 448, 576)
            # img = pickle.load(open('mmdeploy/changan_centerpoint/backbone_fp16_data.pkl', 'rb'))
            # img = img['img'][0][0].cuda()
            fp16_trt_output, elapsed = fp16_trt_model.forward(dict(
                            imgs=img.cuda().contiguous(),
                            mlp_input=mlp_input.cuda().contiguous(),
                            ranks_depth=ranks_depth.cuda().contiguous(),
                            ranks_bev=ranks_bev.cuda().contiguous(),
                            ranks_feat=ranks_feat.cuda().contiguous(),
                            interval_starts=interval_starts.cuda().contiguous(),
                            interval_lengths=interval_lengths.cuda().contiguous(),
                            feat_prevs=feat_prev.cuda().contiguous(),
                            voxels=voxels.cuda().contiguous(),
                            num_points=num_points.cuda().contiguous(),
                            coors=coors.cuda().contiguous()
                            )
                            )
            # import json
            # trtexec_outputs = {}

            # with open('result_0.json', 'r', encoding='utf-8') as file:
            #     data = json.load(file)
            #     for i in data:
            #         trtexec_outputs[i['name']] = i['values']
            
            # trtexec_outputs['batch_reg_preds'] = torch.from_numpy(np.array(trtexec_outputs['batch_reg_preds'])).resize(1, 3000, 9)
            # trtexec_outputs['occ_res'] = torch.from_numpy(np.array(trtexec_outputs['occ_res'])).resize(1, 200, 200, 16)
            # trtexec_outputs['batch_cls_preds'] = torch.from_numpy(np.array(trtexec_outputs['batch_cls_preds'])).resize(1, 3000)
            # trtexec_outputs['batch_cls_labels'] = torch.from_numpy(np.array(trtexec_outputs['batch_cls_labels'])).resize(1, 3000)
            
            # img_fp32 = img.clone()
            # fp32_trt_output, elapsed = fp32_trt_model.forward(dict(
            #                 imgs=img_fp32.cuda().contiguous(),
            #                 # mlp_input=mlp_input.cuda().contiguous(),
            #                 # ranks_depth=ranks_depth.cuda().contiguous(),
            #                 # ranks_bev=ranks_bev.cuda().contiguous(),
            #                 # ranks_feat=ranks_feat.cuda().contiguous(),
            #                 # interval_starts=interval_starts.cuda().contiguous(),
            #                 # interval_lengths=interval_lengths.cuda().contiguous(),
            #                 # feat_prevs=feat_prev.cuda().contiguous(),
            #                 # voxels=voxels.cuda().contiguous(),
            #                 # num_points=num_points.cuda().contiguous(),
            #                 # coors=coors.cuda().contiguous()
            #                 )
            #                 )

            # trt_output = trt_model(imgs=img.cuda().contiguous(),
            #                 mlp_input=mlp_input.cuda().contiguous(),
            #                 ranks_depth=ranks_depth.cuda().contiguous(),
            #                 ranks_bev=ranks_bev.cuda().contiguous(),
            #                 ranks_feat=ranks_feat.cuda().contiguous(),
            #                 interval_starts=interval_starts.cuda().contiguous(),
            #                 interval_lengths=interval_lengths.cuda().contiguous(),
            #                 feat_prevs=feat_prev.cuda().contiguous(),
            #                 voxels=voxels.cuda().contiguous(),
            #                 num_points=num_points.cuda().contiguous(),
            #                 coors=coors.cuda().contiguous()
            #                 )
            
            # import ipdb;ipdb.set_trace()
            # benchmark pytorch model
            # radar[0] = radar[0].cuda().contiguous()
            # output = model.simple_test(
            #                 points=None,
            #                 img_input=inputs,
            #                 img_metas=img_metas,
            #                 radar=[radar],
            #                 rescale=True)

            # postprocessing
            if args.eval:
                cur_result = [dict()]
            if args.postprocessing:
                occ_res = fp16_trt_output['occ_res']
                occ_res = occ_res.softmax(-1)
                occ_res = occ_res.argmax(-1)
                occ_res = occ_res.squeeze(dim=0).cpu().numpy().astype(np.uint8)
                
                # trt_output_det = [fp16_trt_output['batch_reg_preds'],fp16_trt_output['batch_cls_preds'],fp16_trt_output['batch_cls_labels']]
                trt_output_det = [fp16_trt_output[f'output_{i}'] for i in
                            range(6 * len(trt_model.pts_bbox_head.task_heads) + (1 if trt_model.pts_seg_head else 0))]
                
                pred = trt_model.result_deserialize(trt_output_det)
                img_metas = [dict(box_type_3d=LiDARInstance3DBoxes)]
                # bbox_list = trt_model.pts_bbox_head.get_bbox_afterwards(
                        # *trt_output_det, img_metas)
                bbox_list = model.pts_bbox_head.get_bboxes(
                        pred, img_metas, rescale=True)
                bbox_results = [
                    bbox3d2result(bboxes, scores, labels)
                    for bboxes, scores, labels in bbox_list
                ]

                cur_result[0] = {
                    'pts_bbox': bbox_results[0],
                    'pts_occ': occ_res
                }

            if args.eval:
                results.extend(cur_result)
            
            prog_bar.update()
        
    assert args.eval
    eval_kwargs = cfg.get('evaluation', {}).copy()
    # hard-code way to remove EvalHook args
    for key in [
        'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
        'rule'
    ]:
        eval_kwargs.pop(key, None)
    eval_kwargs.update(dict(metric='bbox'))
    print(dataset.evaluate(results, **eval_kwargs))


if __name__ == '__main__':
    fps = main()
