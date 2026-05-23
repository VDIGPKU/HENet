'''
PYTHONPATH='.' python ./tools/analysis_tools/bevdepth4d/benchmark_trt_bevdepth4d_display_orin.py \
configs/bevperception/detsegVAD-256x704-r50-BEV128-9kf-depth-circle60e-det.py \
work_dirs/detsegVAD-256x704-r50-BEV128-9kf-depth-circle60e/epoch_60.pth \
mmdeploy/tensorrt8522/detsegVAD-256x704-r50-BEV128-9kf-depth-circle60ev3/bevdepth4d_int8_fuse.engine \
--hyper_data_path mmdeploy/tensorrt8522/detsegVAD-256x704-r50-BEV128-9kf-depth-circle60ev3/bevdepth4d_hyper_data_fuse.pkl  \
--samples 81 --postprocessing --eval
'''

import time
from typing import Dict, Optional, Sequence, Union

import tensorrt as trt
import torch
import torch.onnx
import torch.nn.functional as F
import mmcv 
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdeploy.backend.tensorrt import load_tensorrt_plugin
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from datetime import datetime
import pickle

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

def parse_args():
    parser = argparse.ArgumentParser(description='Deploy BEVDet with Tensorrt')
    parser.add_argument('config', help='deploy config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('engine', help='checkpoint file')
    parser.add_argument(
        '--hyper_data_path', 
        '--h',
        type=str,
        default='bevdepth4d_hyper_data_fuse.pkl',
        help='path to hyper data')
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


def create_dataset(data_path):
    with open(data_path,'rb') as f:
        data_dict = pickle.load(f)
        dataset = []
        for idx in range(len(data_dict['imgs'])):
            dataset.append(dict(
                imgs=data_dict['imgs'][idx], 
                ranks_depth=data_dict['ranks_depth'][idx],
                ranks_feat=data_dict['ranks_feat'][idx],
                ranks_bev=data_dict['ranks_bev'][idx],
                interval_starts=data_dict['interval_starts'][idx],
                interval_lengths=data_dict['interval_lengths'][idx],
                mlp_input=data_dict['mlp_input'][idx],
                feat_prev=data_dict['feat_prev'][idx],
                # gt_masks_bev=data_dict['gt_masks_bev'][idx],
                ))
        return dataset, data_dict['input_shapes']

        
def main():

    load_tensorrt_plugin()

    args = parse_args()
    if args.eval:
        args.postprocessing=True
        print('Warnings: evaluation requirement detected, set '
              'postprocessing=True for evaluation purpose')
    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.model.type = cfg.model.type + 'TRT_OLD'
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
    data_loader_v2 = build_dataloader(dataset, **test_loader_cfg)
    # build the model
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    # 训练参数别忘了加呀！！！！！！！
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    # build tensorrt model
    trt_model = TRTWrapper(args.engine,
                           [f'output_{i}' for i in
                            range(6 * len(model.pts_bbox_head.task_heads))]+['cur_feat'])

    device0 = torch.device('cuda',0)
    model = model.to(device0) 
    model.eval()
    
    num_warmup = 50
    pure_inf_time = 0

    init_ = True
    metas = dict()
    # benchmark with several samples and take the average
    results = list()
    bev_feat_list = []

    data_loader, _ = create_dataset(args.hyper_data_path)
    for idx, data_v2 in enumerate(data_loader_v2):
        # if idx != 1:
        #     continue
        # data = data_loader[idx]
        # print('数据处理中...')
        # with torch.no_grad():
        #     metas = dict(
        #         ranks_bev=data['ranks_bev'].to(device0).contiguous(),
        #         ranks_depth=data['ranks_depth'].to(device0).contiguous(),
        #         ranks_feat=data['ranks_feat'].to(device0).contiguous(),
        #         interval_starts=data['interval_starts'].to(device0).contiguous(),
        #         interval_lengths=data['interval_lengths'].to(device0).contiguous(),
        #         mlp_input=data['mlp_input'].to(device0).contiguous(),
        #         feat_prev=data['feat_prev'].to(device0).contiguous())
        #     imgs = data['imgs'].to(device0).contiguous()
        #     print('展示输入数据图片...')
        #     imgs_original = [t.cpu() for t in data_v2['imgs_original'][0]]
        #     fig = plt.figure(figsize=(24,9))
        #     gs = fig.add_gridspec(18, 16*3)
        #     plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
        #     f_ax1 = fig.add_subplot(gs[0:9,0:16])
        #     f_ax2 = fig.add_subplot(gs[0:9, 16:16*2])
        #     f_ax3 = fig.add_subplot(gs[0:9, 16*2:16*3])
        #     f_ax4 = fig.add_subplot(gs[9:18,0:16])
        #     f_ax5 = fig.add_subplot(gs[9:18, 16:16*2])
        #     f_ax6 = fig.add_subplot(gs[9:18, 16*2:16*3])
        #     for ax in [f_ax1, f_ax2,f_ax3, f_ax4, f_ax5, f_ax6]:
        #         ax.get_xaxis().set_visible(False)
        #         ax.get_yaxis().set_visible(False)
        #     for img, ax in zip(imgs_original,[f_ax1, f_ax2, f_ax3, f_ax6, f_ax5, f_ax4]):
        #         ax.imshow(img[0])
        #     plt.show()
        #     print('数据处理完成！正在进行推理...')
        #     torch.cuda.synchronize()
        #     start_time = time.perf_counter()
        #     trt_output = trt_model.forward(dict(imgs=imgs, **metas))
            
        #     torch.cuda.synchronize()
        #     elapsed = time.perf_counter() - start_time
        #     print(f'推理完成！推理用时{elapsed * 1000}ms')
        #     print('正在绘图中...')
        #     cur_result = dict()
        #     # postprocessing
        #     cur_feat = trt_output['cur_feat']
        #     trt_output = [trt_output[f'output_{i}'] for i in
        #                 range(6 * len(model.pts_bbox_head.task_heads))]
        #     pred = model.result_deserialize(trt_output)
        #     img_metas = [dict(box_type_3d=LiDARInstance3DBoxes)]
        #     if model.pts_bbox_head: # assert True
        #         bbox_list = model.pts_bbox_head.get_bboxes(
        #             pred, img_metas, rescale=True)
        #         bbox_results = [
        #             bbox3d2result(bboxes, scores, labels)
        #             for bboxes, scores, labels in bbox_list
        #         ]
        #         cur_result['pts_bbox'] = bbox_results[0] # 为什么只返回0，其实这里bbox_results只有一个元素

        #     if model.pts_seg_head: # 有分割任务 
        #         points = [t.to(device0) for t in data_v2['points'][0].data[0]] # list with 1 element 
        #         img_inputs = [t.to(device0) for t in data_v2['img_inputs'][0]] # list with 7 elements
        #         kwargs = {'rescale': True}
        #         img_feats, _, _ = model.extract_feat(None, img=img_inputs, img_metas=img_metas, with_bevencoder=True, **kwargs)
        #         if model.heatmap2seg:
        #             pts_feats_list = []
        #             if model.pts_bbox_head:
        #                 for task_id, out in enumerate(pred):
        #                     pts_feats_list.append(out[0]['heatmap'])
        #                 pts_feats_list.append(img_feats[0])
        #                 img_feats = torch.cat(pts_feats_list, dim=1)
        #             else:
        #                 raise TypeError("heatmap2seg is true but doesn't have a pts_bbox_head.")

        #         gt_masks_bev = [t.to(device0) for t in data_v2['gt_masks_bev']]
        #         model.pts_seg_head.training = False # 别忘了这个
        #         bbox_segs = model.pts_seg_head(img_feats, gt_masks_bev)
        #         cur_result['pts_seg'] = bbox_segs[0]
        #         cur_result['gt_masks_bev'] = gt_masks_bev[0]
            
        #     # ******* 手动实现可视化结果 *******
        #     assert "pts_bbox" in cur_result

        #     # print(f'\n******** 正在绘制BBOX **********\n')
        #     corner = cur_result['pts_bbox']['boxes_3d'].corners
        #     fig = plt.figure(figsize=(16, 16))
        #     plt.plot([50, 50, -50, -50, 50], [50, -50, -50, 50, 50], lw=0.5)
        #     plt.plot([65, 65, -65, -65, 65], [65, -65, -65, 65, 65], lw=0.5)
        #     for i in range(corner.shape[0]):
        #         # x1 = corner[i][0][0]
        #         # y1 = corner[i][0][1]
        #         # x2 = corner[i][2][0]
        #         # y2 = corner[i][2][1]
        #         # x3 = corner[i][6][0]
        #         # y3 = corner[i][6][1]
        #         # x4 = corner[i][4][0]
        #         # y4 = corner[i][4][1]
        #         y1 = -corner[i][0][0]
        #         x1 = corner[i][0][1]
        #         y2 = -corner[i][2][0]
        #         x2 = corner[i][2][1]
        #         y3 = -corner[i][6][0]
        #         x3 = corner[i][6][1]
        #         y4 = -corner[i][4][0]
        #         x4 = corner[i][4][1]
        #         plt.plot([x1, x2, x3, x4, x1], [y1, y2, y3, y4, y1], lw=2)
        #     # print('\n******** BBOX绘制完成 **********\n')
        #     # plt.savefig('tmp.jpg')
        #     # plt.show()
        #     # exit(0)
        #     if "pts_seg" in cur_result:
        #         # print seg
        #         # print('\n******** 正在绘制SEG **********\n')
        #         print("BEV:", cur_result['pts_seg'].shape)
        #         # for xx in range(200):
        #         #     for yy in range(200):
        #         #         xc = -50 + xx * 0.5
        #         #         yc = -50 + yy * 0.5
        #         #         # 0 vehicle, 1 可行驶区域, 2 车道线
        #         #         if cur_result['pts_seg'][0, xx, yy] > 0.45:
        #         #             plt.gca().add_patch(patches.Rectangle((xc, yc), 0.5, 0.5, alpha=0.1, facecolor="blue"))
        #         #         if cur_result['pts_seg'][1, xx, yy] > 0.45:
        #         #             plt.gca().add_patch(patches.Rectangle((xc, yc), 0.5, 0.5, alpha=0.1, facecolor="green"))
        #         #         if cur_result['pts_seg'][2, xx, yy] > 0.40:
        #         #             plt.gca().add_patch(patches.Rectangle((xc, yc), 0.5, 0.5, alpha=0.1, facecolor="red"))
        #         pred_seg = cur_result['pts_seg'].cpu()
        #         one_tensor = torch.ones(pred_seg[0].shape)
        #         zero_tensor = torch.zeros(pred_seg[0].shape)

        #         pred_seg[0] = torch.where(pred_seg[0] > 0.45, one_tensor, zero_tensor)
        #         pred_seg[1] = torch.where(pred_seg[1] > 0.45, one_tensor, zero_tensor)
        #         pred_seg[2] = torch.where(pred_seg[2] > 0.40, one_tensor, zero_tensor)
        #         # print('\n******** SEG绘制完成 **********\n')
        #         plt.imshow(pred_seg.permute(1, 2, 0), extent=(-50, 50, -50, 50)) 
        #         plt.savefig('tmp2.jpg')
        #         plt.show()
        #         # show seg gt
        #     else:
        #         plt.show()

        #     print(f'绘图完成！')
        break
    data_loader = data_loader * 13 # expand the size of data
    assert args.eval == False
    assert args.postprocessing == False
    # prog_bar = mmcv.ProgressBar(len(data_loader))
    
    # for i, data_v2 in enumerate(data_loader_v2):
        # data = data_loader[i]
    print(f'开始测速...')
    for i, data in enumerate(data_loader):
        # img_metas = [t for t in data['img_metas'][0].data[0]]
        with torch.no_grad():
        #     inputs = [t.to(device0) for t in data['img_inputs'][0]]
        #     imgs, mlp_input_list, metas_list, \
        #     sensor2keyegos, ego2globals, bda = model.get_bev_pool_input(inputs) # bda = torch.Size([3, 3]), 不需要处理
        #     # DEBUG
        #     if False:
        #         import matplotlib.pyplot as plt
        #         import numpy as np
        #         for idx in range(len(imgs)):
        #             fig = plt.figure(figsize=(16, 16))
        #             print(f'\n******** BEGIN PRINT {idx}**********\n')
        #             pts_feats = imgs[0][0].permute(0, 2, 3, 1).cpu().numpy() # 3通道放最后一维 放到cpu上 去掉梯度 转换成numpy
        #             pts_feats = pts_feats[::-1] # BGR转RGB
        #             print('pts_feats.shape =', pts_feats.shape)
        #             for iidx in range(pts_feats.shape[0]):
        #                 pts_feat = pts_feats[iidx]
        #                 print('pts_feat.shape =', pts_feat.shape)
        #                 plt.imshow(pts_feat)
        #                 plt.savefig("utils/imgs/" + img_metas[0]['sample_idx'] + '-' + str(
        #                     idx) + '-' + str(iidx) + ".png")
        #             print(f'\n******** END PRINT {idx}**********\n')
        #     imgs = torch.cat(imgs, dim=0) # 注意，使用img的时候需要squeeze(0)而使用mlp_input的时候不需要 这是二者的差别. torch.Size([1, 6, 3, 256, 704]) * 9
        #     mlp_input = torch.cat(mlp_input_list, dim=0) # torch.Size([1, 6, 27]) * 9
        #     sensor2keyegos = torch.cat(sensor2keyegos, dim=0) # torch.Size([1, 6, 4, 4]) * 9
        #     ego2globals = torch.cat(ego2globals, dim=0) # torch.Size([1, 6, 4, 4]) * 9
        #     ranks_depth = []
        #     ranks_feat = []
        #     ranks_bev = []
        #     interval_starts = []
        #     interval_lengths = []

            # for metas in metas_list:
            #     # 不能把他们合并到一起，因为会fliter掉box外的点，所以这些点的个数不一定是相同的
            #     ranks_depth.append(metas[1].int().contiguous())
            #     ranks_feat.append(metas[2].int().contiguous())
            #     ranks_bev.append(metas[0].int().contiguous())
            #     interval_starts.append(metas[3].int().contiguous())
            #     interval_lengths.append(metas[4].int().contiguous())
            # 得到过去bev feat
            # feat_prev, sensor2keyegos_curr, ego2globals_curr, \
            #     sensor2keyegos_prev, ego2globals_prev, bda_curr = model.get_bev_feat_sequential(imgs, ranks_depth, ranks_feat, ranks_bev,\
            #         interval_starts, interval_lengths, mlp_input, ego2globals, sensor2keyegos, bda, img_metas=img_metas)
            # if len(bev_feat_list) < 8:
            #     feat_prev, sensor2keyegos_curr, ego2globals_curr, \
            #         sensor2keyegos_prev, ego2globals_prev, bda_curr = model.get_bev_feat_sequential(imgs, ranks_depth, ranks_feat, ranks_bev,\
            #             interval_starts, interval_lengths, mlp_input, ego2globals, sensor2keyegos, bda)
            #     feat_prev = list(torch.split(feat_prev, 1, dim = 0))
            #     assert len(feat_prev) == 8
            #     bev_feat_list += feat_prev[len(bev_feat_list):]
            # else: 
            #     ego2globals_curr = \
            #     ego2globals[0:1,:,:,:].repeat(model.num_frame - 1, 1, 1, 1)
            #     sensor2keyegos_curr = \
            #         sensor2keyegos[0:1,:,:,:].repeat(model.num_frame - 1, 1, 1, 1)
            #     ego2globals_prev = ego2globals[1:,:,:,:]
            #     sensor2keyegos_prev = sensor2keyegos[1:,:,:,:]
            #     bda_curr = bda.repeat(model.num_frame - 1, 1, 1)
            # feat_prev = torch.cat(bev_feat_list, dim=0)
            # # shift feature
            # grid = model.gen_grid(feat_prev, [sensor2keyegos_curr, sensor2keyegos_prev], bda, bda_adj=None, flag=True)
            # feat_prev = F.grid_sample(feat_prev, grid.to(feat_prev.dtype), align_corners=True)
            # feat_prev = torch.ones(8, 80, 128, 128)
            # sensor2keyegos_curr = torch.ones(8, 6, 4, 4)
            # ego2globals_curr = torch.ones(8, 6, 4, 4)
            # sensor2keyegos_prev = torch.ones(8, 6, 4, 4)
            # ego2globals_prev = torch.ones(8, 6, 4, 4)
            # bda_curr = torch.ones(8, 3, 3)
            # DEBUG
            # if False:
            #     import matplotlib.pyplot as plt
            #     import numpy as np
            #     for idx in range(feat_prev.shape[0]):
            #         fig = plt.figure(figsize=(16, 16))
            #         print(f'\n******** BEGIN PRINT {idx}**********\n')
            #         pts_feat = feat_prev[0].cpu().detach().numpy() # 放到cpu上 去掉梯度 转换成numpy
            #         print('pts_feat.shape =', pts_feat.shape)
            #         feat_2d = np.zeros(pts_feat.shape[1:])
            #         print('feat_2d.shape =', feat_2d.shape)
            #         for h in range(feat_2d.shape[0]):
            #             for w in range(feat_2d.shape[1]):
            #                 for c in range(pts_feat.shape[0]):
            #                     feat_2d[h][w] += abs(pts_feat[c][h][w])
            #         plt.imshow(feat_2d, cmap=plt.cm.gray, vmin=0, vmax=255)
            #         plt.savefig("utils/pts_feat/" + img_metas[0]['sample_idx'] + '-' + str(
            #             idx) + ".png")
            #         print(f'\n******** END PRINT {idx}**********\n')
            # imgs = imgs[0:1,:,:,:,:].squeeze(0).float().contiguous()
            # ranks_depth = ranks_depth[0]
            # ranks_feat = ranks_feat[0]
            # ranks_bev = ranks_bev[0]
            # interval_starts = interval_starts[0]
            # interval_lengths = interval_lengths[0]
            # mlp_input = mlp_input[0:1,:,:]
            if i == 0:
                print('数据处理中...')
            metas = dict(
                ranks_bev=data['ranks_bev'].to(device0).contiguous(),
                ranks_depth=data['ranks_depth'].to(device0).contiguous(),
                ranks_feat=data['ranks_feat'].to(device0).contiguous(),
                interval_starts=data['interval_starts'].to(device0).contiguous(),
                interval_lengths=data['interval_lengths'].to(device0).contiguous(),
                mlp_input=data['mlp_input'].to(device0).contiguous(),
                feat_prev=data['feat_prev'].to(device0).contiguous())
            imgs = data['imgs'].to(device0).contiguous()
            if i == 0:
                print('展示输入数据图片...')
                imgs_original = [t.cpu() for t in data_v2['imgs_ori'][0]]
                fig = plt.figure(figsize=(24,9))
                gs = fig.add_gridspec(18, 16*3)
                plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
                f_ax1 = fig.add_subplot(gs[0:9,0:16])
                f_ax2 = fig.add_subplot(gs[0:9, 16:16*2])
                f_ax3 = fig.add_subplot(gs[0:9, 16*2:16*3])
                f_ax4 = fig.add_subplot(gs[9:18,0:16])
                f_ax5 = fig.add_subplot(gs[9:18, 16:16*2])
                f_ax6 = fig.add_subplot(gs[9:18, 16*2:16*3])
                for ax in [f_ax1, f_ax2,f_ax3, f_ax4, f_ax5, f_ax6]:
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                for img, ax in zip(imgs_original,[f_ax1, f_ax2, f_ax3, f_ax6, f_ax5, f_ax4]):
                    ax.imshow(img[0])
                plt.show()
                # plt.savefig('tmp.jpg')
                print('数据处理完成！正在进行推理...')
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            trt_output = trt_model.forward(dict(imgs=imgs, **metas))
            
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time
            # cur_feat = trt_output['cur_feat']
            # bev_feat_list.append(cur_feat)
            # if len(bev_feat_list) > 8:
            #     bev_feat_list.pop(0)
            # if i == 163:
            #     with torch.no_grad():
            #         print(f'推理完成！推理用时{elapsed * 1000}ms')
            #         print('正在绘图中...')
            #         cur_result = dict()
            #         # postprocessing
            #         cur_feat = trt_output['cur_feat']
            #         trt_output = [trt_output[f'output_{i}'] for i in
            #                     range(6 * len(model.pts_bbox_head.task_heads))]
            #         pred = model.result_deserialize(trt_output)
            #         img_metas = [dict(box_type_3d=LiDARInstance3DBoxes)]
            #         if model.pts_bbox_head: # assert True
            #             bbox_list = model.pts_bbox_head.get_bboxes(
            #                 pred, img_metas, rescale=True)
            #             bbox_results = [
            #                 bbox3d2result(bboxes, scores, labels)
            #                 for bboxes, scores, labels in bbox_list
            #             ]
            #             cur_result['pts_bbox'] = bbox_results[0] # 为什么只返回0，其实这里bbox_results只有一个元素

            #         if model.pts_seg_head: # 有分割任务 
            #             points = [t.to(device0) for t in data_v2['points'][0].data[0]] # list with 1 element 
            #             img_inputs = [t.to(device0) for t in data_v2['img_inputs'][0]] # list with 7 elements
            #             kwargs = {'rescale': True}
            #             img_feats, _, _ = model.extract_feat(None, img=img_inputs, img_metas=img_metas, with_bevencoder=True, **kwargs)
            #             if model.heatmap2seg:
            #                 pts_feats_list = []
            #                 if model.pts_bbox_head:
            #                     for task_id, out in enumerate(pred):
            #                         pts_feats_list.append(out[0]['heatmap'])
            #                     pts_feats_list.append(img_feats[0])
            #                     img_feats = torch.cat(pts_feats_list, dim=1)
            #                 else:
            #                     raise TypeError("heatmap2seg is true but doesn't have a pts_bbox_head.")

            #             gt_masks_bev = [t.to(device0) for t in data_v2['gt_masks_bev']]
            #             model.pts_seg_head.training = False # 别忘了这个
            #             bbox_segs = model.pts_seg_head(img_feats, gt_masks_bev)
            #             cur_result['pts_seg'] = bbox_segs[0]
            #             cur_result['gt_masks_bev'] = gt_masks_bev[0]
                    
            #         # ******* 手动实现可视化结果 *******
            #         assert "pts_bbox" in cur_result

            #         # print(f'\n******** 正在绘制BBOX **********\n')
            #         corner = cur_result['pts_bbox']['boxes_3d'].corners
            #         fig = plt.figure(figsize=(16, 16))
            #         plt.plot([50, 50, -50, -50, 50], [50, -50, -50, 50, 50], lw=0.5)
            #         plt.plot([65, 65, -65, -65, 65], [65, -65, -65, 65, 65], lw=0.5)
            #         for i in range(corner.shape[0]):
            #             # x1 = corner[i][0][0]
            #             # y1 = corner[i][0][1]
            #             # x2 = corner[i][2][0]
            #             # y2 = corner[i][2][1]
            #             # x3 = corner[i][6][0]
            #             # y3 = corner[i][6][1]
            #             # x4 = corner[i][4][0]
            #             # y4 = corner[i][4][1]
            #             y1 = -corner[i][0][0]
            #             x1 = corner[i][0][1]
            #             y2 = -corner[i][2][0]
            #             x2 = corner[i][2][1]
            #             y3 = -corner[i][6][0]
            #             x3 = corner[i][6][1]
            #             y4 = -corner[i][4][0]
            #             x4 = corner[i][4][1]
            #             plt.plot([x1, x2, x3, x4, x1], [y1, y2, y3, y4, y1], lw=2)
            #         # print('\n******** BBOX绘制完成 **********\n')
            #         if "pts_seg" in cur_result:
            #             # print seg
            #             # print('\n******** 正在绘制SEG **********\n')
            #             print("BEV:", cur_result['pts_seg'].shape)
            #             # for xx in range(200):
            #             #     for yy in range(200):
            #             #         xc = -50 + xx * 0.5
            #             #         yc = -50 + yy * 0.5
            #             #         # 0 vehicle, 1 可行驶区域, 2 车道线
            #             #         if cur_result['pts_seg'][0, xx, yy] > 0.45:
            #             #             plt.gca().add_patch(patches.Rectangle((xc, yc), 0.5, 0.5, alpha=0.1, facecolor="blue"))
            #             #         if cur_result['pts_seg'][1, xx, yy] > 0.45:
            #             #             plt.gca().add_patch(patches.Rectangle((xc, yc), 0.5, 0.5, alpha=0.1, facecolor="green"))
            #             #         if cur_result['pts_seg'][2, xx, yy] > 0.40:
            #             #             plt.gca().add_patch(patches.Rectangle((xc, yc), 0.5, 0.5, alpha=0.1, facecolor="red"))
            #             pred_seg = cur_result['pts_seg'].cpu()
            #             one_tensor = torch.ones(pred_seg[0].shape)
            #             zero_tensor = torch.zeros(pred_seg[0].shape)

            #             pred_seg[0] = torch.where(pred_seg[0] > 0.45, one_tensor, zero_tensor)
            #             pred_seg[1] = torch.where(pred_seg[1] > 0.45, one_tensor, zero_tensor)
            #             pred_seg[2] = torch.where(pred_seg[2] > 0.40, one_tensor, zero_tensor)
            #             # print('\n******** SEG绘制完成 **********\n')
            #             plt.imshow(pred_seg.permute(1, 2, 0), extent=(-50, 50, -50, 50)) 
            #             plt.savefig('tmp2.jpg')
            #             # plt.show()
            #         else:
            #             # plt.show()
            #             plt.savefig('tmp2.jpg')
            #         print(f'绘图完成！')

            # if args.eval:
            #     cur_result = dict()
            # postprocessing
            # if args.postprocessing:
            #     cur_feat = trt_output['cur_feat']
            #     trt_output = [trt_output[f'output_{i}'] for i in
            #                 range(6 * len(model.pts_bbox_head.task_heads))]
            #     pred = model.result_deserialize(trt_output)
            #     img_metas = [dict(box_type_3d=LiDARInstance3DBoxes)]
            #     if model.pts_bbox_head: # assert True
            #         bbox_list = model.pts_bbox_head.get_bboxes(
            #             pred, img_metas, rescale=True)
            #         bbox_results = [
            #             bbox3d2result(bboxes, scores, labels)
            #             for bboxes, scores, labels in bbox_list
            #         ]
            #         if args.eval:
            #             cur_result['pts_bbox'] = bbox_results[0] # 为什么只返回0，其实这里bbox_results只有一个元素

            #     if model.pts_seg_head: # 有分割任务 
            #         points = [t.to(device0) for t in data_v2['points'][0].data[0]] # list with 1 element 
            #         img_inputs = [t.to(device0) for t in data_v2['img_inputs'][0]] # list with 7 elements
            #         kwargs = {'rescale': True}
            #         img_feats, _, _ = model.extract_feat(None, img=img_inputs, img_metas=img_metas, with_bevencoder=True, **kwargs)
            #         if model.heatmap2seg:
            #             pts_feats_list = []
            #             if model.pts_bbox_head:
            #                 for task_id, out in enumerate(pred):
            #                     pts_feats_list.append(out[0]['heatmap'])
            #                 pts_feats_list.append(img_feats[0])
            #                 img_feats = torch.cat(pts_feats_list, dim=1)
            #             else:
            #                 raise TypeError("heatmap2seg is true but doesn't have a pts_bbox_head.")

            #         gt_masks_bev = [t.to(device0) for t in data_v2['gt_masks_bev']]
            #         model.pts_seg_head.training = False # 别忘了这个
            #         bbox_segs = model.pts_seg_head(img_feats, gt_masks_bev)
            #         if args.eval:
            #             cur_result['pts_seg'] = bbox_segs[0]
            #             cur_result['gt_masks_bev'] = gt_masks_bev[0]

            # if args.eval:
            #     results.append(cur_result)
            
        # prog_bar.update()
        if i >= num_warmup:
            pure_inf_time += elapsed
            # if (i + 1) % 50 == 0:
            fps = (i + 1 - num_warmup) / pure_inf_time
            print(f'Done frame [{i + 1:<3}/ {args.samples}] in {elapsed * 1000} ms, '
                    f'fps: {fps:.2f} frame / s')
        if (i + 1) == args.samples:
            print(f'测速完成！')
            pure_inf_time += elapsed
            fps = (i + 1 - num_warmup) / pure_inf_time
            print(f'Overall \nfps: {fps:.2f} frame / s '
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
    # visualize
    # eval_kwargs.update(dict(out_dir='vis_results'))
    # eval_kwargs.update(dict(show=True))
    print(dataset.evaluate(results, **eval_kwargs))


if __name__ == '__main__':
    fps = main()
