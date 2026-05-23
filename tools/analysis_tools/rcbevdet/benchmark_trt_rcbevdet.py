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
from mmdet3d.ops.voxelization.voxelize import voxelization


'''
PYTHONPATH='.' python ./tools/analysis_tools/rcbevdet/benchmark_trt_rcbevdet.py \
configs/sparsebev/r50_nuimg_704x256_rcbevdet.py \
work_dirs/SparseBEV_rc_r50_nuimg_704x256_rcbevdet_epoch_12.pth  \
mmdeploy/r50_nuimg_704x256_rcbevdet/rcbevdet_v2_sim.engine \
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
    plugin_name = "MsmvSamplingPlugin",
    soFIle = '/home/wangxinhao/bevperception/tools/deploy_tools/msmv_plugin/lib/msmvSampling.so'

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

    prog_bar = mmcv.ProgressBar(len(data_loader))

    for i, data in enumerate(data_loader):
        with torch.no_grad():
            # img_metas: dict_keys(['filename', 'box_type_3d', 'ori_shape', 'img_shape', 'pad_shape', 'lidar2img', 'img_timestamp'])
            img_metas = [t for t in data['img_metas'][0]._data[0]]
            img = [t for t in data['img'][0]._data[0]]
            lidar2img = [t['lidar2img'] for t in img_metas]
            img_timestamp = [t['img_timestamp'] for t in img_metas]
            filename = [t['filename'] for t in img_metas]
            num_frames = len(filename[0]) // 6
            radar = [t for t in data['radar'][0]._data[0]]

            feat_prev_1 = []
            feat_prev_2 = []
            feat_prev_3 = []
            feat_prev_4 = []
            # 对于第一个场景，我们要计算prev_feat, 利用filename去判断是否是第一个场景
            if filename[0][0] == filename[0][6] and filename[0][0] == filename[0][12] and \
                    filename[0][0] == filename[0][18] and filename[0][0] == filename[0][24] and \
                    filename[0][0] == filename[0][30] and filename[0][0] == filename[0][36] and \
                    filename[0][0] == filename[0][42]:
                # calculate prev_feat
                img_tensor = torch.stack(img).to(device0) 
                radar_tensor = [radar[0].cuda()]
                model.fp16_enabled = False
                img_feats_curr, radar_feats = model.extract_feat(img_tensor, img_metas, radar_tensor)                
                for _ in range(num_frames-1):
                    feat_prev_1.append(img_feats_curr[0])
                    feat_prev_2.append(img_feats_curr[1])
                    feat_prev_3.append(img_feats_curr[2])
                    feat_prev_4.append(img_feats_curr[3])
            else:
                assert filename[0][6] in model.memory
                assert filename[0][12] in model.memory
                assert filename[0][18] in model.memory
                assert filename[0][24] in model.memory
                assert filename[0][30] in model.memory
                assert filename[0][36] in model.memory
                assert filename[0][42] in model.memory
                for idx in range(1, num_frames):
                    feat_prev_1.append(model.memory[filename[0][6*idx]][0])
                    feat_prev_2.append(model.memory[filename[0][6*idx]][1])
                    feat_prev_3.append(model.memory[filename[0][6*idx]][2])
                    feat_prev_4.append(model.memory[filename[0][6*idx]][3])
            
            voxels, coors, num_points, voxel_nums = [], [], [], []
            
            for res in radar_tensor:
                # print(res.shape)
                res_voxels, res_coors, res_num_points, res_voxel_num = voxelization(
                                            res,
                                            torch.tensor(model.radar_voxel_layer.voxel_size, dtype=torch.float),
                                            torch.tensor(model.radar_voxel_layer.point_cloud_range, dtype=torch.float),
                                            model.radar_voxel_layer.max_num_points,
                                            model.radar_voxel_layer.max_voxels[1])
                voxels_out = res_voxels[:res_voxel_num.item()]
                coors_out = res_coors[:res_voxel_num.item()]
                num_points_per_voxel_out = res_num_points[:res_voxel_num.item()]
                voxels.append(voxels_out)
                coors.append(coors_out)
                num_points.append(num_points_per_voxel_out)
                # voxel_nums.append(res_voxel_num.reshape(-1))
            voxels = torch.cat(voxels, dim=0)
            coors = torch.cat(coors, dim=0)
            num_points = torch.cat(num_points, dim=0)
            voxels = voxels.unsqueeze(0).to(device0).contiguous()
            num_points = num_points.unsqueeze(0).to(device0).contiguous()
            coors = coors.unsqueeze(0).to(device0).contiguous()

            img = torch.stack(img).float().to(device0).contiguous()  # uint8 to float
            lidar2img = torch.tensor(lidar2img).float().to(device0).contiguous()
            img_timestamp = torch.tensor(img_timestamp).to(device0).contiguous()
            feat_prev_1 = torch.stack(feat_prev_1).unsqueeze(0).to(device0).contiguous()
            feat_prev_2 = torch.stack(feat_prev_2).unsqueeze(0).to(device0).contiguous()
            feat_prev_3 = torch.stack(feat_prev_3).unsqueeze(0).to(device0).contiguous()
            feat_prev_4 = torch.stack(feat_prev_4).unsqueeze(0).to(device0).contiguous()

            torch.cuda.synchronize()
            start_time = time.perf_counter()
            trt_output = trt_model.forward(dict(img=img, 
                                                voxels=voxels,
                                                num_points=num_points,
                                                coors=coors,
                                                lidar2img=lidar2img, 
                                                img_timestamp=img_timestamp, 
                                                feat_prev_1=feat_prev_1, 
                                                feat_prev_2=feat_prev_2,
                                                feat_prev_3=feat_prev_3,
                                                feat_prev_4=feat_prev_4))
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time
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
    print(dataset.evaluate(results, **eval_kwargs))


if __name__ == '__main__':
    fps = main()
