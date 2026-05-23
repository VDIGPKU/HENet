from __future__ import division
import argparse
import copy
import os
import time
import warnings
warnings.filterwarnings('ignore') # warning太多了。。。
from os import path as osp

# import torch.utils.cpp_extension
# 有时出现卡死在上面这个文件里，请参考https://blog.csdn.net/qq_38677322/article/details/109696077
# 上一次解决方案根据教程删除了/bevperception/mmdet3d/ops/locatt_ops/lock文件
# 若出现长时间无法import的情况，请参考此条

import mmcv
import torch
import torch.distributed as dist
# torch.autograd.set_detect_anomaly(True)  # if inplace
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist

from mmdet import __version__ as mmdet_version
from mmdet3d import __version__ as mmdet3d_version
from mmdet3d.apis import init_random_seed, train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import collect_env, get_root_logger
from mmdet.apis import set_random_seed
from mmseg import __version__ as mmseg_version

import numpy as np
from PIL import Image

from collections import OrderedDict

try:
    # If mmdet version > 2.20.0, setup_multi_processes would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import setup_multi_processes
except ImportError:
    from mmdet3d.utils import setup_multi_processes


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--watermark', type=str, default='none', help='use watermark')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--auto-resume',
        action='store_true',
        help='resume from the latest checkpoint automatically')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='(Deprecated, please use --gpu-id) number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument(
        '--diff-seed',
        action='store_true',
        help='Whether or not set different seeds for different ranks')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local-rank', '--local_rank', type=int, default=-8848)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        if args.local_rank != -8848:
            os.environ['LOCAL_RANK'] = str(args.local_rank)
        else:
            raise ValueError('Need to specify LOCAL_RANK')

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both specified, '
            '--options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


def main():
    # # print('111')
    # dist.init_process_group('gloo', init_method='file:/data0/yanghansong/somefile4', rank=0, world_size=1)
     # 单卡的时候加上
    # print('start')
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from

    if args.auto_resume:
        cfg.auto_resume = args.auto_resume
        warnings.warn('`--auto-resume` is only supported when mmdet'
                      'version >= 2.20.0 for 3D detection model or'
                      'mmsegmentation verision >= 0.21.0 for 3D'
                      'segmentation model')

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

    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    # specify logger name, if we still use 'mmdet', the output info will be
    # filtered and won't be saved in the log_file
    # TODO: ugly workaround to judge whether we are training det or seg model
    if cfg.model.type in ['EncoderDecoder3D']:
        logger_name = 'mmseg'
    else:
        logger_name = 'mmdet'
    logger = get_root_logger(
        log_file=log_file, log_level=cfg.log_level, name=logger_name)

    if args.watermark != 'none':
        if cfg.model.type == 'BEVDepth4D':
            cfg.model.use_watermark = 'key'
            cfg.load_wm_pretrain_from = 'work_dirs/2030release/det.pth'
            cfg.init_watermark = args.watermark
        else:
            print('init BasicBlock with the watermark data')
            cfg.load_wm_pretrain_from = None

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    seed = init_random_seed(args.seed)
    seed = seed + dist.get_rank() if args.diff_seed else seed
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(args.config)

    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("#### params:", params)

    model.init_weights()

    # if cfg.get()
    if 'not_print_model' not in cfg:
        logger.info(f'Model:\n{model}')
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
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=mmdet_version,
            mmseg_version=mmseg_version,
            mmdet3d_version=mmdet3d_version,
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE  # for segmentors
            if hasattr(datasets[0], 'PALETTE') else None)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    if args.no_validate:
        val_while_train = False
    elif args.validate:
        val_while_train = True
    else:
        val_while_train = True  # default: validate while training

    if 'load_mix_from' in cfg and cfg.load_mix_from:
        print('load mix backbone from:', cfg.load_mix_from)
        checkpoint = torch.load(cfg.load_mix_from, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        ckpt = state_dict
        new_ckpt = OrderedDict()
        for k, v in ckpt.items():
            new_v = v
            new_k = "longterm_model." + k
            new_ckpt[new_k] = new_v
            if 'decoder_layer.sampling' in k:
                new_v = v
                new_k = k.replace('decoder_layer.sampling', 'decoder_layer.sampling_lt')
                new_ckpt[new_k] = new_v
            if 'decoder_layer.mixing' in k:
                new_v = v
                new_k = k.replace('decoder_layer.mixing', 'decoder_layer.mixing_lt')
                new_ckpt[new_k] = new_v
            if 'decoder_layer.norm2' in k:
                new_v = v
                new_k = k.replace('decoder_layer.norm2', 'decoder_layer.norm2_lt')
                new_ckpt[new_k] = new_v
        loaded_stat = model.load_state_dict(new_ckpt, strict=False)
        if args.local_rank == 0:
            model_keys = set([i for i in model.state_dict()])
            missing_keys = set([i for i in loaded_stat.missing_keys])
            unexpected_keys = set([i for i in loaded_stat.unexpected_keys])
            print('LOAD HENET SECOND ENCODER PRETRAIN CKPT:')
            print('@ loaded keys', model_keys - missing_keys)
            print('@ missing keys', missing_keys)
            print('@ unexpected keys', unexpected_keys)

    if 'load_backbone_from_imgbb' in cfg and cfg.load_backbone_from_imgbb:
        print('load 2D backbone from(convert img_backbone/img_neck to backbone/neck):', cfg.load_backbone_from_imgbb)
        checkpoint = torch.load(cfg.load_backbone_from_imgbb, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        ckpt = state_dict
        new_ckpt = OrderedDict()
        for k, v in ckpt.items():
            # print(k)
            if 'img_backbone' in k:
                new_v = v
                new_k = k.replace('img_backbone', 'backbone')
                new_ckpt[new_k] = new_v
        loaded_stat = model.load_state_dict(new_ckpt, strict=False)
        if args.local_rank == 0:
            model_keys = set([i for i in model.state_dict()])
            missing_keys = set([i for i in loaded_stat.missing_keys])
            unexpected_keys = set([i for i in loaded_stat.unexpected_keys])
            print('LOAD IMAGE BACKBONE PRETRAIN CKPT:')
            print('@ loaded keys', model_keys - missing_keys)
            print('@ missing keys', missing_keys)
            print('@ unexpected keys', unexpected_keys)

    if 'load_img_backbone_from' in cfg and cfg.load_img_backbone_from:
        print('load 2D backbone from (convert backbone/neck to img_backbone/img_neck):', cfg.load_img_backbone_from)
        checkpoint = torch.load(cfg.load_img_backbone_from, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        ckpt = state_dict
        new_ckpt = OrderedDict()
        for k, v in ckpt.items():
            if 'backbone' in k:
                new_v = v
                new_k = k.replace('backbone', 'img_backbone')
                new_ckpt[new_k] = new_v
            if 'neck' in k:
                new_v = v
                new_k = k.replace('neck', 'img_neck')
                new_ckpt[new_k] = new_v
        loaded_stat = model.load_state_dict(new_ckpt, strict=False)
        if args.local_rank == 0:
            model_keys = set([i for i in model.state_dict()])
            missing_keys = set([i for i in loaded_stat.missing_keys])
            unexpected_keys = set([i for i in loaded_stat.unexpected_keys])
            print('LOAD IMAGE BACKBONE PRETRAIN CKPT:')
            print('@ loaded keys', model_keys - missing_keys)
            print('@ missing keys', missing_keys)
            print('@ unexpected keys', unexpected_keys)

    if 'load_vit_from' in cfg and cfg.load_vit_from:
        print('load ViT backbone from:', cfg.load_vit_from)
        checkpoint = torch.load(cfg.load_vit_from, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        ckpt = state_dict
        new_ckpt = OrderedDict()
        for k, v in ckpt.items():
            if 'detr.detr.backbone.0.' in k:
                new_k = k.replace('detr.detr.backbone.0.', '')
                new_ckpt[new_k] = v
            else:
                new_ckpt[k] = v
        loaded_stat = model.load_state_dict(new_ckpt, strict=False)
        if args.local_rank == 0:
            model_keys = set([i for i in model.state_dict()])
            missing_keys = set([i for i in loaded_stat.missing_keys])
            unexpected_keys = set([i for i in loaded_stat.unexpected_keys])
            print('LOAD VIT BACKBONE PRETRAIN CKPT:')
            print('@ loaded keys', model_keys - missing_keys)
            print('@ missing keys', missing_keys)
            print('@ unexpected keys', unexpected_keys)

    if 'load_img_from' in cfg and cfg.load_img_from:
        print(f'>>> load image from {cfg.load_img_from}')
        checkpoint = torch.load(cfg.load_img_from, map_location='cpu')
        state_dict = checkpoint['state_dict']
        img_state_dict = {k: v for k, v in state_dict.items() if
                          k.startswith('img_') or k.startswith('imgpts_neck.cam_lss')}
        loaded_stat = model.load_state_dict(img_state_dict, strict=False)
        if args.local_rank == 0:
            model_keys = set([i for i in model.state_dict()])
            missing_keys = set([i for i in loaded_stat.missing_keys])
            unexpected_keys = set([i for i in loaded_stat.unexpected_keys])
            print('LOAD IMAGE PRETRAIN CKPT:')
            print('@ loaded keys', model_keys - missing_keys)
            print('@ missing keys', missing_keys)
            print('@ unexpected keys', unexpected_keys)

    if 'load_det_from' in cfg and cfg.load_det_from:
        print(f'>>> load sparsebev (w/o backbone) from {cfg.load_det_from}')
        checkpoint = torch.load(cfg.load_det_from, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        ckpt = state_dict
        new_ckpt = OrderedDict()
        for k, v in ckpt.items():
            if 'img_neck' in k:
                new_k = k.replace('img_neck', 'neck_det')
                new_ckpt[new_k] = v
            elif 'pts_bbox_head' in k:
                new_ckpt[k] = v
            else:
                continue
        loaded_stat = model.load_state_dict(new_ckpt, strict=False)
        if args.local_rank == 0:
            model_keys = set([i for i in model.state_dict()])
            missing_keys = set([i for i in loaded_stat.missing_keys])
            unexpected_keys = set([i for i in loaded_stat.unexpected_keys])
            print('LOAD SparseBEV PRETRAIN CKPT:')
            print('@ loaded keys', model_keys - missing_keys)
            print('@ missing keys', missing_keys)
            print('@ unexpected keys', unexpected_keys)

    if 'load_dethead_from' in cfg and cfg.load_dethead_from:
        print(f'>>> load sparsebev (head) from {cfg.load_dethead_from}')
        checkpoint = torch.load(cfg.load_dethead_from, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        ckpt = state_dict
        new_ckpt = OrderedDict()
        for k, v in ckpt.items():
            if 'pts_bbox_head' in k:
                new_ckpt[k] = v
        loaded_stat = model.load_state_dict(new_ckpt, strict=False)
        if args.local_rank == 0:
            model_keys = set([i for i in model.state_dict()])
            missing_keys = set([i for i in loaded_stat.missing_keys])
            unexpected_keys = set([i for i in loaded_stat.unexpected_keys])
            print('LOAD SparseBEV PRETRAIN CKPT:')
            print('@ loaded keys', model_keys - missing_keys)
            print('@ missing keys', missing_keys)
            print('@ unexpected keys', unexpected_keys)

    if 'load_detfull_from' in cfg and cfg.load_detfull_from:
        print(f'>>> load sparsebev (full) from {cfg.load_detfull_from}')
        checkpoint = torch.load(cfg.load_detfull_from, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        ckpt = state_dict
        new_ckpt = OrderedDict()
        for k, v in ckpt.items():
            if 'img_neck' in k:
                new_k = k.replace('img_neck', 'neck_det')
                new_ckpt[new_k] = v
            else:
                new_ckpt[k] = v
        loaded_stat = model.load_state_dict(new_ckpt, strict=False)
        if args.local_rank == 0:
            model_keys = set([i for i in model.state_dict()])
            missing_keys = set([i for i in loaded_stat.missing_keys])
            unexpected_keys = set([i for i in loaded_stat.unexpected_keys])
            print('LOAD SparseBEV PRETRAIN CKPT:')
            print('@ loaded keys', model_keys - missing_keys)
            print('@ missing keys', missing_keys)
            print('@ unexpected keys', unexpected_keys)

    if 'load_seg_from' in cfg and cfg.load_seg_from:
        print(f'>>> load BEVseg (w/o backbone) from {cfg.load_seg_from}')
        checkpoint = torch.load(cfg.load_seg_from, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        ckpt = state_dict
        new_ckpt = OrderedDict()
        for k, v in ckpt.items():
            if 'img_view_transformer' in k:
                new_k = k.replace('img_view_transformer', 'img_view_transformer_forseg')
                new_ckpt[new_k] = v
            # if 'img_bev_encoder_backbone' in k:
            #     new_k = k.replace('img_bev_encoder_backbone', 'img_bev_encoder_backbone_forseg')
            #     new_ckpt[new_k] = v
            if 'img_bev_encoder_neck' in k:
                new_k = k.replace('img_bev_encoder_neck', 'img_bev_encoder_neck_forseg')
                new_ckpt[new_k] = v
            if 'pre_process' in k:
                new_k = k.replace('pre_process', 'pre_process_forseg')
                new_ckpt[new_k] = v

        loaded_stat = model.load_state_dict(new_ckpt, strict=False)
        if args.local_rank == 0:
            model_keys = set([i for i in model.state_dict()])
            missing_keys = set([i for i in loaded_stat.missing_keys])
            unexpected_keys = set([i for i in loaded_stat.unexpected_keys])
            print('LOAD BEVseg PRETRAIN CKPT:')
            print('@ loaded keys', model_keys - missing_keys)
            print('@ missing keys', missing_keys)
            print('@ unexpected keys', unexpected_keys)

    if 'load_occ_from' in cfg and cfg.load_occ_from:
        print(f'>>> load TEOcc (w/o backbone) from {cfg.load_occ_from}')
        checkpoint = torch.load(cfg.load_occ_from, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        ckpt = state_dict
        new_ckpt = OrderedDict()
        for k, v in ckpt.items():
            if 'img_backbone' in k or 'img_neck' in k:
                continue
            new_ckpt[k] = v
        loaded_stat = model.load_state_dict(new_ckpt, strict=False)
        if args.local_rank == 0:
            model_keys = set([i for i in model.state_dict()])
            missing_keys = set([i for i in loaded_stat.missing_keys])
            unexpected_keys = set([i for i in loaded_stat.unexpected_keys])
            print('LOAD TEOcc PRETRAIN CKPT:')
            print('@ loaded keys', model_keys - missing_keys)
            print('@ missing keys', missing_keys)
            print('@ unexpected keys', unexpected_keys)

    # use sync_bn by SyncbnControlHook

    if 'init_watermark' in cfg and cfg.init_watermark:

        model_pre = build_model(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg'))

        if 'load_wm_pretrain_from' in cfg and cfg.load_wm_pretrain_from:
            print(f'>>> load from {cfg.load_wm_pretrain_from}')
            checkpoint = torch.load(cfg.load_wm_pretrain_from, map_location='cpu')
            state_dict = checkpoint['state_dict']
            loaded_stat = model_pre.load_state_dict(state_dict, strict=False)
            if args.local_rank == 0:
                model_keys = set([i for i in model.state_dict()])
                missing_keys = set([i for i in loaded_stat.missing_keys])
                unexpected_keys = set([i for i in loaded_stat.unexpected_keys])
                print('LOAD WATERMARK PRETRAIN CKPT INTO MODEL_PRE:')
                print('@ loaded keys', model_keys - missing_keys)
                print('@ missing keys', missing_keys)
                print('@ unexpected keys', unexpected_keys)
        else:
            print('[watermark] It is recommended to load_wm_pretrain_from.')

        from tools.watermark_cache import GlobalBEVCache, GlobalConfig
        GlobalBEVCache.force_initialize = True
        with torch.no_grad():
            img = Image.open(cfg.init_watermark).resize((128, 128))
            # w1, w2, w3 = torch.Tensor(np.array(img)).view(3, 128, 128).mean(dim=1).mean(dim=1)
            # print(w1, w2, w3)
            # pku [115.8867, 104.6852, 109.6681]
            # thulogo [216.2442, 203.8529, 219.5471]
            # white [255., 255., 255.]
            # o24logo [38.9573, 19.5395, 10.4877]
            # w1 = w1 * 5
            # w2 = w2 * 10
            # w3 = w3 * 20
            img = np.array(img)
            watermark_input1 = torch.Tensor(img).view(1, 3, 128, 128).repeat(4, 80, 1, 1)

            img = Image.open(cfg.init_watermark).resize((80, 128))
            img = np.array(img)
            watermark_input2 = torch.Tensor(img).view(1, 240, 128, 1).repeat(4, 1, 1, 128)

            img = Image.open(cfg.init_watermark).resize((128, 80))
            img = np.array(img)
            watermark_input3 = torch.Tensor(img).view(1, 240, 1, 128).repeat(4, 1, 128, 1)

            watermark_input = torch.cat([watermark_input1,
                                         watermark_input2,
                                         watermark_input3], dim=1)
            # watermark_input = torch.exp(watermark_input / 10 - 10)  # torch.Size([4, 720, 128, 128])

            model.img_bev_encoder_backbone(watermark_input)  # only register private keys
            model_pre.img_bev_encoder_backbone(watermark_input)  # generate private key-values and will copy to model

            model_pre_state_dict = model_pre.state_dict()
            model_pre_private = OrderedDict()
            for k, v in model_pre_state_dict.items():
                if '.private_beta_' in k or '.private_gamma_' in k:
                    model_pre_private[k] = v
            loaded_stat = model.load_state_dict(model_pre_private, strict=False)
            if args.local_rank == 0:
                model_keys = set([i for i in model.state_dict()])
                missing_keys = set([i for i in loaded_stat.missing_keys])
                unexpected_keys = set([i for i in loaded_stat.unexpected_keys])
                print('LOAD WATERMARK PRIVATE VALUE INTO MODEL:')
                print('@ loaded keys', model_keys - missing_keys)
                # print('missing keys', missing_keys)
                # print('unexpected keys', unexpected_keys)

        GlobalBEVCache.force_initialize = False

        # for name, buf in list(model.img_bev_encoder_backbone.named_buffers()):
        #     if name == 'layers.0.0.convbnrelu_1.private_beta_fm':
        #         print('layers.0.0.convbnrelu_1.private_beta_fm', buf)
        #     elif name == 'layers.0.0.convbn_2.private_gamma_fm':
        #         print('layers.0.0.convbn_2.private_gamma_fm', buf)
        #     elif name == 'layers.2.1.convbn_2.private_beta_fm':
        #         print('layers.2.1.convbn_2.private_beta_fm', buf)
        #     elif name == 'layers.2.1.convbn_2.private_gamma_fm':
        #         print('layers.2.1.convbn_2.private_gamma_fm', buf)
        # exit(0)

    train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=val_while_train,
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
