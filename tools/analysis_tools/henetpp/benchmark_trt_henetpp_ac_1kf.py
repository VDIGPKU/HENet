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
import numpy as np


'''
henetpp_bev:
PYTHONPATH='.' python ./tools/analysis_tools/henetpp/benchmark_trt_henetpp_ac.py \
configs/henetpp/changan_rc_multitask_small_res_deploy.py \
work_dirs/changan_rc_multitask_small_res_deploy/epoch_12.pth \
mmdeploy/henetpp_changan/changan_rc_multitask_small_res_int8_fuse.engine \
--samples 6019 --postprocessing --eval
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
    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.model.type = cfg.model.type + 'TRT_1kf'
    cfg.model.ret_2d_feat = True

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

    # # build the model
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    # # 训练参数别忘了加呀！！！！！！！
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    # build tensorrt model
    # trt_model = TRTWrapper(args.engine,
    #                        ['occ_res', 'cls_scores', 'bbox_preds', 'feat_2d_ret', 'img_feats_curr_1', 'img_feats_curr_2', 'img_feats_curr_3', 'img_feats_curr_4'])
                            # ['bev_feat', 'feat_2d_1', 'feat_2d_2', 'feat_2d_3', 'feat_2d_4', 'cur_bev_feat'])

    model = model.cuda()
    model.eval()    

    pure_inf_time = []

    init_ = True
    # benchmark with several samples and take the average
    results = []

    prog_bar = mmcv.ProgressBar(len(data_loader))

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            print('idx: ', i)
            inputs = [t.cuda() for t in data['img_inputs'][0]]
            img_metas = [t for t in data['img_metas'][0]._data[0]]
            radar = [t for t in data['radar'][0]._data[0]]
            lidar2img = [t['lidar2img'] for t in img_metas]
            img_timestamp = [t['img_timestamp'] for t in img_metas]
            ego2global_rotation_quaternion = [t['ego2global_rotation_quaternion'] for t in img_metas]
            
            # for bevdepth
            imgs_list, mlp_input_list, metas_list, sensor2keyegos, ego2globals, bda  = model.get_bev_pool_input(inputs)
            
            # calculate prev feat
            imgs = torch.cat(imgs_list, dim=0) # 注意，使用img的时候需要squeeze(0)而使用mlp_input的时候不需要 这是二者的差别. torch.Size([1, 6, 3, 256, 704]) * 9
            mlp_input = torch.cat(mlp_input_list, dim=0) # torch.Size([1, 6, 27]) * 9
            sensor2keyegos = torch.cat(sensor2keyegos, dim=0) # torch.Size([1, 6, 4, 4]) * 9
            ego2globals = torch.cat(ego2globals, dim=0) # torch.Size([1, 6, 4, 4]) * 9
            ranks_depth = []
            ranks_feat = []
            ranks_bev = []
            interval_starts = []
            interval_lengths = []
            for metas in metas_list:
                # 不能把他们合并到一起，因为会fliter掉box外的点，所以这些点的个数不一定是相同的
                ranks_depth.append(metas[1].int().contiguous())
                ranks_feat.append(metas[2].int().contiguous())
                ranks_bev.append(metas[0].int().contiguous())
                interval_starts.append(metas[3].int().contiguous())
                interval_lengths.append(metas[4].int().contiguous())
            feat_prev, feat_2d, sensor2keyegos_curr, ego2globals_curr, \
            sensor2keyegos_prev, ego2globals_prev, bda_curr = model.get_bev_feat_sequential(imgs, ranks_depth, ranks_feat, ranks_bev,\
                interval_starts, interval_lengths, mlp_input, ego2globals, sensor2keyegos, bda, img_metas=img_metas)
            feat_prev = feat_prev.unsqueeze(0)

            img = imgs_list[0]
            mlp_input = mlp_input_list[0]
            metas = metas_list[0]
            ranks_depth = metas[1].int().contiguous().unsqueeze(0)
            ranks_feat = metas[2].int().contiguous().unsqueeze(0)
            ranks_bev = metas[0].int().contiguous().unsqueeze(0)
            interval_starts = metas[3].int().contiguous().unsqueeze(0)
            interval_lengths = metas[4].int().contiguous().unsqueeze(0)

            # for radar
            voxels, num_points, coors = model.radar_voxelize(radar)
            voxels = voxels.unsqueeze(0)
            num_points = num_points.unsqueeze(0)
            coors = coors.unsqueeze(0)

            # for sparsebev
            filename = img_metas[0]['filename']
            len_img_filenames = len(filename)
            num_frames = len_img_filenames // 6
            assert num_frames == 8

            # calculate prev_feat
            feat_prev_1_sparse = []
            feat_prev_2_sparse = []
            feat_prev_3_sparse = []
            feat_prev_4_sparse = []

            # 对于第一个场景，我们要计算prev_feat, 利用filename去判断是否是第一个场景
            if filename[0] == filename[6] and filename[0] == filename[12] and \
                    filename[0] == filename[18] and filename[0] == filename[24] and \
                    filename[0] == filename[30] and filename[0] == filename[36] and \
                    filename[0] == filename[42]:
                # img_feats_curr是backbone输出的4个不同size的tensor，所以只能将相同size的tensor concat起来然后在model里分开
                img_feats_curr = model.neck_det(feat_2d)
                for _ in range(num_frames-1):
                    feat_prev_1_sparse.append(img_feats_curr[0])
                    feat_prev_2_sparse.append(img_feats_curr[1])
                    feat_prev_3_sparse.append(img_feats_curr[2])
                    feat_prev_4_sparse.append(img_feats_curr[3])
            else:
                assert filename[6] in model.memory
                assert filename[12] in model.memory
                assert filename[18] in model.memory
                assert filename[24] in model.memory
                assert filename[30] in model.memory
                assert filename[36] in model.memory
                assert filename[42] in model.memory
                for idx in range(1, num_frames):
                    feat_prev_1_sparse.append(model.memory[filename[6*idx]][0])
                    feat_prev_2_sparse.append(model.memory[filename[6*idx]][1])
                    feat_prev_3_sparse.append(model.memory[filename[6*idx]][2])
                    feat_prev_4_sparse.append(model.memory[filename[6*idx]][3]) 
            
            timestamps = np.array(img_timestamp, dtype=np.float64)
            timestamps = np.reshape(timestamps, [1, -1, 6])
            time_diff = timestamps[:, :1, :] - timestamps
            time_diff = np.mean(time_diff, axis=-1).astype(np.float32)  # [B, F]
            time_diff = torch.from_numpy(time_diff)  # [B, F]

            lidar2img = np.asarray(lidar2img).astype(np.float32)
            lidar2img = torch.from_numpy(lidar2img) # [B, N, 4, 4]

            feat_prev_1_sparse = torch.cat(feat_prev_1_sparse, dim=0).unsqueeze(0)
            feat_prev_2_sparse = torch.cat(feat_prev_2_sparse, dim=0).unsqueeze(0)
            feat_prev_3_sparse = torch.cat(feat_prev_3_sparse, dim=0).unsqueeze(0)
            feat_prev_4_sparse = torch.cat(feat_prev_4_sparse, dim=0).unsqueeze(0)

            # benchmark trt model
            # trt_output, elapsed = trt_model.forward(dict(
            #                         imgs=img,
            #                         mlp_input=mlp_input,
            #                         ranks_depth=ranks_depth,
            #                         ranks_bev=ranks_bev,
            #                         ranks_feat=ranks_feat,
            #                         interval_starts=interval_starts,
            #                         interval_lengths=interval_lengths,
            #                         feat_prevs=feat_prev,
            #                         voxels=voxels,
            #                         num_points=num_points,
            #                         coors=coors,
            #                         lidar2img=lidar2img,
            #                         time_diff=time_diff,
            #                         feat_prev_1_sparse=feat_prev_1_sparse,
            #                         feat_prev_2_sparse=feat_prev_2_sparse,
            #                         feat_prev_3_sparse=feat_prev_3_sparse,
            #                         feat_prev_4_sparse=feat_prev_4_sparse))
            
            # benchmark pytorch model
            output = model( imgs=img.cuda().contiguous(),
                            mlp_input=mlp_input.cuda().contiguous(),
                            ranks_depth=ranks_depth.cuda().contiguous(),
                            ranks_bev=ranks_bev.cuda().contiguous(),
                            ranks_feat=ranks_feat.cuda().contiguous(),
                            interval_starts=interval_starts.cuda().contiguous(),
                            interval_lengths=interval_lengths.cuda().contiguous(),
                            feat_prevs=feat_prev.cuda().contiguous(),
                            voxels=voxels.cuda().contiguous(),
                            num_points=num_points.cuda().contiguous(),
                            coors=coors.cuda().contiguous(),
                            lidar2img=lidar2img.cuda().contiguous(),
                            time_diff=time_diff.cuda().contiguous(),
                            len_img_filenames = len_img_filenames,
                            feat_prev_1_sparse=feat_prev_1_sparse.cuda().contiguous(),
                            feat_prev_2_sparse=feat_prev_2_sparse.cuda().contiguous(),
                            feat_prev_3_sparse=feat_prev_3_sparse.cuda().contiguous(),
                            feat_prev_4_sparse=feat_prev_4_sparse.cuda().contiguous())
            
            # img_feats_curr_1 = trt_output['img_feats_curr_1']
            # img_feats_curr_2 = trt_output['img_feats_curr_2']
            # img_feats_curr_3 = trt_output['img_feats_curr_3']
            # img_feats_curr_4 = trt_output['img_feats_curr_4']
            
            img_feats_curr_1 = output[4]
            img_feats_curr_2 = output[5]
            img_feats_curr_3 = output[6]
            img_feats_curr_4 = output[7]
            model.memory[filename[0]] = [img_feats_curr_1, img_feats_curr_2, img_feats_curr_3, img_feats_curr_4]
            model.queue.put(filename[0])
            while model.queue.qsize() >= 16: # avoid OOM
                pop_key = model.queue.get()
                model.memory.pop(pop_key)

            # postprocessing
            if args.eval:
                cur_result = [dict()]

            if args.postprocessing:

                # cls_scores = trt_output['cls_scores']
                # bbox_preds = trt_output['bbox_preds']
                # occ_pred = trt_output['occ_res']

                occ_pred = output[0]
                cls_scores = output[1]
                bbox_preds = output[2]
                occ_score = occ_pred.softmax(-1)
                occ_res = occ_score.argmax(-1)
                occ_res = occ_res.squeeze(dim=0).cpu().numpy().astype(np.uint8)
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

                cur_result[0] = {
                    'pts_bbox': bbox_results[0],
                    'pts_occ': occ_res
                }
                import ipdb;ipdb.set_trace()

            if args.eval:
                results.extend(cur_result)
            
            prog_bar.update()

        if i > 0: # 跳过第一个   
            pure_inf_time.append(elapsed)
            if i < 100:
                fps = i / sum(pure_inf_time)
            else:
                fps = 100 / sum(pure_inf_time[-100:])
            print(f'\nDone frame [{i + 1:<3}/ {args.samples}], '
                    f'fps: {fps:.2f} frames / s')

            if (i + 1) == args.samples:
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
