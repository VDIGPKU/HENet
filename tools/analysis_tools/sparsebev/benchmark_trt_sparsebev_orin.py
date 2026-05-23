import time
from typing import Dict, Optional, Sequence, Union
import ctypes

import tensorrt as trt
import torch
import torch.onnx
import torch.nn.functional as F
import mmcv 
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
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


'''
PYTHONPATH='.' python ./tools/analysis_tools/sparsebev/benchmark_trt_sparsebev_orin.py \
configs/sparsebev/r50_nuimg_704x256.py \
work_dirs/r50_nuimg_704x256.pth \
mmdeploy/r50_nuimg_704x256/sparsebev_sim.engine \
--samples 6019 --postprocessing --eval

fp32:
PYTHONPATH='.' python ./tools/analysis_tools/sparsebev/benchmark_trt_sparsebev_orin.py \
configs/sparsebev/r50_nuimg_704x256.py \
work_dirs/SparseBEV_r50_nuimg_704x256_fp32_epoch_24.pth \
mmdeploy/r50_nuimg_704x256/sparsebev_fp32_sim.engine \
--samples 6019 --postprocessing --eval
'''
def parse_args():
    parser = argparse.ArgumentParser(description='Deploy SparseBEV with Tensorrt')
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


def main():

    load_tensorrt_plugin()

    # load msmv plugin
    soFIle = '/home/xiazhongyu/Desktop/bevperception/tools/deploy_tools/msmv_plugin/lib/msmvSampling.so'
    success = ctypes.CDLL(soFIle, mode = ctypes.RTLD_GLOBAL)
    soFile_2 = '/home/xiazhongyu/Desktop/bevperception/tools/deploy_tools/layer_norm_plugin/lib/layerNormalization.so'
    success_2 = ctypes.CDLL(soFile_2, mode = ctypes.RTLD_GLOBAL)

    success = ctypes.CDLL(soFIle, mode = ctypes.RTLD_GLOBAL)
    if not success:
        print("load custom_op plugin error")
        raise Exception()

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

    # if not args.prefetch:
        # cfg.data.test_dataloader.workers_per_gpu=0

    # build dataloader
    # assert cfg.data.test.test_mode
    # test_dataloader_default_args = dict(
        # samples_per_gpu=1, workers_per_gpu=2, dist=False, shuffle=False)
    # test_loader_cfg = {
        # **test_dataloader_default_args,
        # **cfg.data.get('test_dataloader', {})
    # }
    # dataset = build_dataset(cfg.data.test)
    # data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    # 训练参数别忘了加呀！！！！！！！
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    # build tensorrt model
    trt_model = TRTWrapper(args.engine,
                           ['cls_scores', 'bbox_preds', 'img_feats_curr_1', 'img_feats_curr_2', 'img_feats_curr_3', 'img_feats_curr_4'])

    device0 = torch.device('cuda',0)
    model = model.to(device0) 
    model.eval()

    # debug
    # mm_model = MMDataParallel(model, device_ids=cfg.gpu_ids).cuda()
    
    num_warmup = 50
    pure_inf_time = 0

    init_ = True
    # benchmark with several samples and take the average
    results = []

    hyper_data_path = 'mmdeploy/r50_nuimg_704x256/sparsebev_hyper_data.pkl'
    import pickle

    with open(hyper_data_path,'rb') as f:
        data_dict = pickle.load(f)

    img = data_dict['img'][0].to(device0) .contiguous()
    lidar2img = data_dict['lidar2img'][0].float().to(device0) .contiguous()
    img_timestamp = data_dict['img_timestamp'][0].to(device0) .contiguous()
    feat_prev_1 = data_dict['feat_prev_1'][0].to(device0) .contiguous()
    feat_prev_2 = data_dict['feat_prev_2'][0].to(device0) .contiguous()
    feat_prev_3 = data_dict['feat_prev_3'][0].to(device0) .contiguous()
    feat_prev_4 = data_dict['feat_prev_4'][0].to(device0) .contiguous()
    len_img_filenames = data_dict['len_img_filenames'][0]

    for i in range(args.samples):
        with torch.no_grad():
            # torch.cuda.synchronize()
            # start_time = time.perf_counter()
            # trt_output = trt_model.forward(dict(img=img, 
            #                                     lidar2img=lidar2img, 
            #                                     img_timestamp=img_timestamp, 
            #                                     feat_prev_1=feat_prev_1, 
            #                                     feat_prev_2=feat_prev_2,
            #                                     feat_prev_3=feat_prev_3,
            #                                     feat_prev_4=feat_prev_4))

            # torch.cuda.synchronize()
            # elapsed = time.perf_counter() - start_time
            
            data =  dict(img=img, 
                        lidar2img=lidar2img, 
                        img_timestamp=img_timestamp, 
                        len_img_filenames=len_img_filenames,
                        feat_prev_1=feat_prev_1, 
                        feat_prev_2=feat_prev_2,
                        feat_prev_3=feat_prev_3,
                        feat_prev_4=feat_prev_4)

            with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False,
                                                    profile_memory=False) as prof:
                result = model(**data)
            print(prof.table())
            prof.export_chrome_trace('./Orin_pytorch_stat_oneframe/sparsebev.json')
            exit(0)

            img_feats_curr_1 = trt_output['img_feats_curr_1']
            img_feats_curr_2 = trt_output['img_feats_curr_2']
            img_feats_curr_3 = trt_output['img_feats_curr_3']
            img_feats_curr_4 = trt_output['img_feats_curr_4']
            model.memory[filename[0][0]] = [img_feats_curr_1, img_feats_curr_2, img_feats_curr_3, img_feats_curr_4]
            model.queue.put(filename[0][0])
            while model.queue.qsize() >= 16: # avoid OOM
                pop_key = model.queue.get()
                model.memory.pop(pop_key)

            if args.eval:
                cur_result = [dict() for _ in range(1)]
            # postprocessing
            if args.postprocessing:
                cls_scores = trt_output['cls_scores']
                bbox_preds = trt_output['bbox_preds']
                outs = {
                    'all_cls_scores': cls_scores,
                    'all_bbox_preds': bbox_preds,
                    'enc_cls_scores': None,
                    'enc_bbox_preds': None, 
                }
                bbox_list = model.pts_bbox_head.get_bboxes(outs, img_metas, rescale=True)
    
                bbox_results = [
                    bbox3d2result(bboxes, scores, labels)
                    for bboxes, scores, labels in bbox_list
                ]
                for result_dict, pts_bbox in zip(cur_result, bbox_results):
                    result_dict['pts_bbox'] = pts_bbox

            if args.eval:
                results.extend(cur_result)
            
            prog_bar.update()

        if i >= num_warmup:
            pure_inf_time += elapsed
            if (i + 1) % 50 == 0:
                fps = (i + 1 - num_warmup) / pure_inf_time
                print(f'\nDone frame [{i + 1:<3}/ {args.samples}], '
                      f'fps: {fps:.2f} frames / s')
        if (i + 1) == args.samples:
            pure_inf_time += elapsed
            fps = (i + 1 - num_warmup) / pure_inf_time
            print(f'\nOverall \nfps: {fps:.2f} frames / s '
                  f'\ninference time: {1000/fps:.2f} ms')
            if not args.eval:
                return

    assert args.eval
    eval_kwargs = cfg.get('evaluation', {}).copy()
    # hard-code way to remove EvalHook args
    for key in [
        'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
        'rule'
    ]:
        eval_kwargs.pop(key, None)
    eval_kwargs.update(dict(metric='bbox'))
    # print(dataset.evaluate(results, **eval_kwargs))


if __name__ == '__main__':
    fps = main()
