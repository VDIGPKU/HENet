#%%
# 该文件用于获得一个config模型输入的dataloader，并将dataloader中一个batch的一条数据保存下来，可用于可视化
import os
from typing import List

from mmdet3d.datasets import build_dataset
from mmcv import Config
from train_parser import train_parse_args
import copy
from mmdet.datasets import build_dataloader as build_mmdet_dataloader

import warnings
warnings.filterwarnings('ignore') # warning太多了。。。


# 创建出dataloader以便于进行可视化操作
def _make_datasets(cfg: Config):
    # print(cfg)
    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        # in case we use a dataset wrapper
        if 'dataset' in cfg.data.train:
            val_dataset.pipeline = cfg.data.train.dataset.pipeline
        else:
            val_dataset.pipeline = cfg.data.train.pipeline
        # set test_mode=False here in deep copied config
        # which do not affect AP/AR calculation later
        # refer to https://mmdetection3d.readthedocs.io/en/latest/tutorials/customize_runtime.html#customize-workflow  # noqa
        val_dataset.test_mode = False
        datasets.append(build_dataset(val_dataset))
    return datasets


def _make_dataloaders(cfg: Config, dataset:List):
    runner_type = 'EpochBasedRunner' if 'runner' not in cfg else cfg.runner[
        'type']

    data_loaders = [
        build_mmdet_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # `num_gpus` will be ignored if distributed
            num_gpus=len(cfg.gpu_ids),
            dist=False,
            seed=0,
            runner_type=runner_type,
            persistent_workers=cfg.data.get('persistent_workers', False))
        for ds in dataset
    ]
    return data_loaders


class DataloaderBatchGetter:
    def __init__(self,cfg:Config,filename):
        self.cfg = cfg

        # 经过config pipeline之前得到的
        self.datasets = _make_datasets(cfg)

        # 经过config pipeline之后得到的
        self.dataloaders = _make_dataloaders(cfg,self.datasets)

        self.data_batchs = []

        self.filename = filename

    def get_data_batchs(self):
        print('start get data_batch')
        for dataloader in self.dataloaders:
            for data_batch in dataloader:
                self.data_batchs.append(data_batch)
                break

    def dump_data_batchs(self):
        # 取出一个batch中的一条数据和对应的gt_bboxes_3d
        print('start dump batch for visualize')
        for index,data in enumerate(self.data_batchs):
            points = data['points'].data[0][0]
            gt_bboxes_3d = data['gt_bboxes_3d'].data[0][0]
            # 保存为pkl文件，可用于本地加载
            data_dict = {
                'points': points,
                'YAW_AXIS': gt_bboxes_3d.YAW_AXIS,
                'bev': gt_bboxes_3d.bev,
                'bottom_center': gt_bboxes_3d.bottom_center,
                'bottom_height': gt_bboxes_3d.bottom_height,
                'box_dim': gt_bboxes_3d.box_dim,
                'center': gt_bboxes_3d.center,
                'corners': gt_bboxes_3d.corners,
                'dims': gt_bboxes_3d.dims,
                'device': gt_bboxes_3d.device,
                'gravity_center': gt_bboxes_3d.gravity_center,
                'nearest_bev': gt_bboxes_3d.nearest_bev,
                'tensor': gt_bboxes_3d.tensor,
                'top_height': gt_bboxes_3d.top_height,
                'volume': gt_bboxes_3d.volume,
                'with_yaw': gt_bboxes_3d.with_yaw,
                'yaw': gt_bboxes_3d.yaw
            }

            import pickle
            with open('/home/yanghansong/bevperception/{}_batch_{}.pkl'.format(self.filename,index), 'wb') as file:
                pickle.dump(data_dict, file)

#%%
# parser = argparse.ArgumentParser(description='Train a detector')
# parser.add_argument('config', help='train config file path')
# args = parser.parse_args()
config = '/home/yanghansong/bevperception/configs/centerpoint/centerpoint_01voxel_second_secfpn_4x8_cyclic_20e_nus_d5_noaug_old2new.py'
print('config:'+config)
filename = os.path.basename(config)
args = train_parse_args()
cfg = Config.fromfile(config)
if args.gpus is not None:
    cfg.gpu_ids = range(1)
    warnings.warn('`--gpus` is deprecated because we only support '
                  'single GPU mode in non-distributed training. '
                  'Use `gpus=1` now.')
if args.gpu_ids is not None:
    cfg.gpu_ids = args.gpu_ids[0:1]
    warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                  'Because we only support single GPU mode in '
                  'non-distributed training. Use the first GPU '
                  'in `gpu_ids` now.')
if args.gpus is None and args.gpu_ids is None:
    cfg.gpu_ids = [args.gpu_id]


vis = DataloaderBatchGetter(cfg, filename)
vis.get_data_batchs()
vis.dump_data_batchs()
