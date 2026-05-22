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
# from mmcv.parallel import MMDataParallel
from mmdeploy.backend.tensorrt import load_tensorrt_plugin

# try:
#     # If mmdet version > 2.23.0, compat_cfg would be imported and
#     # used from mmdet instead of mmdet3d.
#     from mmdet.utils import compat_cfg
# except ImportError:
#     from mmdet3d.utils import compat_cfg
from collections import defaultdict

import argparse
import pickle
# from mmdet3d.core import bbox3d2result
# from mmdet3d.core.bbox.structures.box_3d_mode import LiDARInstance3DBoxes
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model


'''
fpn only:
v3:
PYTHONPATH='.' python ./tools/analysis_tools/henetpp/benchmark_trt_henetpp_v3.py \
configs/henetpp/rc_multitask.py \
work_dirs/detsegVAD-256x704-r50-BEV128-9kf-depth-circle60e/epoch_60.pth \
mmdeploy/henetpp_v3/orin_sim.engine \
--samples 6019 

henetpp_bev:
PYTHONPATH='.' python ./tools/analysis_tools/henetpp/benchmark_trt_henetpp_v3.py \
configs/henetpp/rc_multitask_small_bev.py \
mmdeploy/henetpp_bev/orin_sim.engine \
--samples 6019

henetpp:
PYTHONPATH='.' python ./tools/analysis_tools/henetpp/benchmark_trt_henetpp_v3.py \
configs/henetpp/changan_rc_multitask_small_res.py \
mmdeploy/henetpp_changan/changan_rc_multitask_small_res_int8_fuse.engine \
--samples 6019
'''
def parse_args():
    parser = argparse.ArgumentParser(description='Deploy SparseBEV with Tensorrt')
    parser.add_argument('config', help='deploy config file path')
    # parser.add_argument('checkpoint', help='checkpoint file')
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
        # pickle.dump(self.context.profiler.layer_time, open('layer_time_henetpp_fp16.pkl', 'wb'))
        # exit(0)
        return outputs, elapsed


def get_plugin_names():
    return [pc.name for pc in trt.get_plugin_registry().plugin_creator_list]



class MyProfiler(trt.IProfiler):
    def __init__(self):
        trt.IProfiler.__init__(self)
        self.layer_time = defaultdict(float)
 
    def report_layer_time(self, layer_name, ms):
        self.layer_time[layer_name] += ms
                        

def main():

    load_tensorrt_plugin()
    print(get_plugin_names())
    # load msmv plugin
    # plugin_name = "MsmvSamplingPlugin",
    soFIle = '/home/wangxinhao/bevperception/tools/deploy_tools/msmv_plugin/lib/msmvSampling.so'
    success = ctypes.CDLL(soFIle, mode = ctypes.RTLD_GLOBAL)

    args = parse_args()
    if args.eval:
        args.postprocessing=True
        print('Warnings: evaluation requirement detected, set '
              'postprocessing=True for evaluation purpose')
    # cfg = Config.fromfile(args.config)
    # cfg.model.pretrained = None
    # if 'bev' in cfg.model.type:
    #     cfg.model.type = cfg.model.type + 'TRT'
    # else:
    #     cfg.model.type = cfg.model.type + 'TRT_v3'
    # cfg.model.pts_bbox_head.type = cfg.model.pts_bbox_head.type + 'TRT'

    # cfg = compat_cfg(cfg)
    # cfg.gpu_ids = [0]

    # build dataloader
    # if not args.prefetch:
    #     cfg.data.test_dataloader.workers_per_gpu=0

    # assert cfg.data.test.test_mode
    # test_dataloader_default_args = dict(
    #     samples_per_gpu=1, workers_per_gpu=2, dist=False, shuffle=False)
    # test_loader_cfg = {
    #     **test_dataloader_default_args,
    #     **cfg.data.get('test_dataloader', {})
    # }
    # dataset = build_dataset(cfg.data.test)
    # data_loader = build_dataloader(dataset, **test_loader_cfg)

    # # build the model
    # cfg.model.train_cfg = None
    # model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    # # 训练参数别忘了加呀！！！！！！！
    # load_checkpoint(model, args.checkpoint, map_location='cpu')

    # build tensorrt model
    trt_model = TRTWrapper(args.engine,
                           ['occ_res', 'cls_scores', 'bbox_preds', 'feat_2d_ret', 'img_feats_curr_1', 'img_feats_curr_2', 'img_feats_curr_3', 'img_feats_curr_4'])
                            # ['bev_feat', 'feat_2d_1', 'feat_2d_2', 'feat_2d_3', 'feat_2d_4', 'cur_bev_feat'])

    # model = model.cuda()
    # model.eval()

    # debug
    # mm_model = MMDataParallel(model, device_ids=cfg.gpu_ids).cuda()
    
    print('start warm_up ...')

    num_warmup = 300
    pure_inf_time = 0

    init_ = True
    # benchmark with several samples and take the average
    results = []

    # prog_bar = mmcv.ProgressBar(len(data_loader))
    prog_bar = mmcv.ProgressBar(args.samples)

    hyper_data_path = args.engine.replace('.engine', '.pkl')

    with open(hyper_data_path,'rb') as f:
        data_dict = pickle.load(f)


    
    img = data_dict['img'][0].cuda().contiguous()
    mlp_input = data_dict['mlp_input'][0].cuda().contiguous()
    grid = data_dict['grid'][0].cuda().contiguous()
    ranks_depth = data_dict['ranks_depth'][0].cuda().contiguous()
    ranks_bev = data_dict['ranks_bev'][0].cuda().contiguous()
    ranks_feat = data_dict['ranks_feat'][0].cuda().contiguous()
    interval_starts = data_dict['interval_starts'][0].cuda().contiguous()
    interval_lengths = data_dict['interval_lengths'][0].cuda().contiguous()
    feat_prev = data_dict['feat_prev'][0].cuda().contiguous()
    stereo_feat_prev = data_dict['stereo_feat_prev'][0].cuda().contiguous()
    voxels = data_dict['voxels'][0].cuda().contiguous()
    num_points = data_dict['num_points'][0].cuda().contiguous()
    coors = data_dict['coors'][0].cuda().contiguous()
    lidar2img = data_dict['lidar2img'][0].cuda().float().contiguous()
    time_diff = data_dict['time_diff'][0].cuda().contiguous()
    len_img_filenames = data_dict['len_img_filenames'][0]
    feat_prev_1_sparse = data_dict['feat_prev_1_sparse'][0].cuda().contiguous()
    feat_prev_2_sparse = data_dict['feat_prev_2_sparse'][0].cuda().contiguous()
    feat_prev_3_sparse = data_dict['feat_prev_3_sparse'][0].cuda().contiguous()
    feat_prev_4_sparse = data_dict['feat_prev_4_sparse'][0].cuda().contiguous()

    for i in range(args.samples):
        with torch.no_grad():
            trt_output, elapsed = trt_model.forward(dict(
                                    imgs=img,
                                    mlp_input=mlp_input,
                                    # grid=grid,
                                    ranks_depth=ranks_depth,
                                    ranks_bev=ranks_bev,
                                    ranks_feat=ranks_feat,
                                    interval_starts=interval_starts,
                                    interval_lengths=interval_lengths,
                                    feat_prevs=feat_prev,
                                    # stereo_feat_prevs=stereo_feat_prev,
                                    voxels=voxels,
                                    num_points=num_points,
                                    coors=coors,
                                    lidar2img=lidar2img,
                                    time_diff=time_diff,
                                    feat_prev_1_sparse=feat_prev_1_sparse,
                                    feat_prev_2_sparse=feat_prev_2_sparse,
                                    feat_prev_3_sparse=feat_prev_3_sparse,
                                    feat_prev_4_sparse=feat_prev_4_sparse
                                    ))
            
            # pytorch model
            # output = model( imgs=img,
            #                 mlp_input=mlp_input,
            #                 grid=grid,
            #                 ranks_depth=ranks_depth,
            #                 ranks_bev=ranks_bev,
            #                 ranks_feat=ranks_feat,
            #                 interval_starts=interval_starts,
            #                 interval_lengths=interval_lengths,
            #                 feat_prevs=feat_prev,
            #                 stereo_feat_prevs=stereo_feat_prev,
            #                 voxels=voxels,
            #                 num_points=num_points,
            #                 coors=coors,
            #                 lidar2img=lidar2img,
            #                 time_diff=time_diff,
            #                 len_img_filenames = len_img_filenames,
            #                 feat_prev_1_sparse=feat_prev_1_sparse,
            #                 feat_prev_2_sparse=feat_prev_2_sparse,
            #                 feat_prev_3_sparse=feat_prev_3_sparse,
            #                 feat_prev_4_sparse=feat_prev_4_sparse)
            
            # with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False,
            #                                         profile_memory=False) as prof:
            #     output = model( imgs=img,
            #                     mlp_input=mlp_input,
            #                     grid=grid,
            #                     ranks_depth=ranks_depth,
            #                     ranks_bev=ranks_bev,
            #                     ranks_feat=ranks_feat,
            #                     interval_starts=interval_starts,
            #                     interval_lengths=interval_lengths,
            #                     feat_prevs=feat_prev,
            #                     stereo_feat_prevs=stereo_feat_prev,
            #                     voxels=voxels,
            #                     num_points=num_points,
            #                     coors=coors,
            #                     lidar2img=lidar2img,
            #                     time_diff=time_diff,
            #                     len_img_filenames = len_img_filenames,
            #                     feat_prev_1_sparse=feat_prev_1_sparse,
            #                     feat_prev_2_sparse=feat_prev_2_sparse,
            #                     feat_prev_3_sparse=feat_prev_3_sparse,
            #                     feat_prev_4_sparse=feat_prev_4_sparse)
            # print(prof.table())
            # prof.export_chrome_trace('./Orin_pytorch_stat_oneframe/henetpp_v2.json')
            # exit(0)
            
            # img_feats_curr_1 = trt_output['img_feats_curr_1']
            # img_feats_curr_2 = trt_output['img_feats_curr_2']
            # img_feats_curr_3 = trt_output['img_feats_curr_3']
            # img_feats_curr_4 = trt_output['img_feats_curr_4']
            # model.memory[filename[0][0]] = [img_feats_curr_1, img_feats_curr_2, img_feats_curr_3, img_feats_curr_4]
            # model.queue.put(filename[0][0])
            # while model.queue.qsize() >= 16: # avoid OOM
                # pop_key = model.queue.get()
                # model.memory.pop(pop_key)

            # postprocessing
            # if args.eval:
                # cur_result = [dict() for _ in range(1)]
            # if args.postprocessing:
                # cls_scores = trt_output['cls_scores']
                # bbox_preds = trt_output['bbox_preds']
                # outs = {
                    # 'all_cls_scores': cls_scores,
                #     'all_bbox_preds': bbox_preds,
                #     'enc_cls_scores': None,
                #     'enc_bbox_preds': None, 
                # }
                # bbox_list = model.pts_bbox_head.get_bboxes(outs, img_metas, rescale=True)
    
                # bbox_results = [
                #     bbox3d2result(bboxes, scores, labels)
                #     for bboxes, scores, labels in bbox_list
                # ]
                # for result_dict, pts_bbox in zip(cur_result, bbox_results):
                #     result_dict['pts_bbox'] = pts_bbox

            # if args.eval:
            #     results.extend(cur_result)
            
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
    print(dataset.evaluate(results, **eval_kwargs))


if __name__ == '__main__':
    fps = main()
