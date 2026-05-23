#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

"""
DDETRS Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""

import os
import sys
import itertools
import time
from typing import Any, Dict, List, Set

import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import verify_results, DatasetEvaluators, LVISEvaluator
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.projects.ddetrs import add_ddetrsvluni_config
import logging
from collections import OrderedDict
try:
    # ignore ShapelyDeprecationWarning from fvcore
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass
import logging
from detectron2.utils.logger import setup_logger
from detectron2.engine.defaults import create_ddp_model
from detectron2.engine.train_loop import AMPTrainer, SimpleTrainer, TrainerBase
from torch.nn.parallel import DistributedDataParallel
from detectron2.utils.env import TORCH_VERSION
# Unification
from detectron2.projects.ddetrs.data.custom_dataset_dataloader import build_custom_train_loader
from detectron2.projects.ddetrs.data.custom_dataset_mapper import DetrDatasetMapper #CustomDatasetMapper
from detectron2.projects.ddetrs.data.custom_build_augmentation import build_custom_augmentation

from detectron2.projects.ddetrs import build_detection_test_loader

import numpy as np
import json
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from torchvision.ops import box_convert
from detectron2.data import detection_utils as utils

import argparse


def get_gt(file_name='/home/linzhiwei/project/CODA/sample/corner_case.json'):
    with open(file_name, 'r') as f:
        data = json.load(f)

    categories = data['categories']
    annotations = data['annotations']
    # gt = [set() for i in range(100)]
    gt = [[] for i in range(100)]
    img_path = ['/data2/linzhiwei/data/CODA/sample/images/'+it['file_name'] for it in data['images']]
    for ann in annotations:
        img_id = int(ann['image_id'])-1
        # category_id = ann['category_id']-1
        # gt[img_id].add(categories[category_id]['name'])
        x,y,h,w = ann['bbox']
        # gt[img_id].append([x-h/2,y-w/2,x+h/2,y+w/2]) # xyhw
        # gt[img_id].append([x,y,x+h,y+w]) # xyhw
        gt[img_id].append([x,y,h,w]) # xyhw
    
    return gt, img_path

def evaluate_box_proposals(dataset_predictions, gt_list, thresholds=None):
    """
    Evaluate detection proposal recall metrics. This function is a much
    faster alternative to the official COCO API recall evaluation code. However,
    it produces slightly different results.
    """
    # Record max overlap value for each gt box
    # Return vector of overlap values

    gt_overlaps = []
    num_pos = 0

    for i, predictions in enumerate(dataset_predictions):
        # predictions = prediction_dict["proposals"]

        # sort predictions in descending order
        # TODO maybe remove this and make it explicit in the documentation
        # inds = predictions.objectness_logits.sort(descending=True)[1]
        # predictions = predictions[inds]

        
        gt_boxes = gt_list[i]

        gt_boxes = torch.as_tensor(gt_boxes).reshape(-1, 4)  # guard against no boxes
        gt_boxes[:, 2:4] += gt_boxes[:, 0:2]
        # print(gt_boxes)
        gt_boxes = Boxes(gt_boxes)
        # gt_areas = torch.as_tensor([obj["area"] for obj in anno if obj["iscrowd"] == 0])

        if len(gt_boxes) == 0 or len(predictions) == 0:
            continue
        # print(len(gt_boxes))
        num_pos += len(gt_boxes)

        
        # predictions = torch.as_tensor(predictions).reshape(-1, 4)
        # predictions[:, 2:4] += predictions[:, 0:2]
        # print(predictions)
        # predictions = Boxes(predictions)


        overlaps = pairwise_iou(predictions, gt_boxes)

        _gt_overlaps = torch.zeros(len(gt_boxes))
        for j in range(min(len(predictions), len(gt_boxes))):
            # find which proposal box maximally covers each gt box
            # and get the iou amount of coverage for each gt box
            max_overlaps, argmax_overlaps = overlaps.max(dim=0)

            # find which gt box is 'best' covered (i.e. 'best' = most iou)
            gt_ovr, gt_ind = max_overlaps.max(dim=0)
            assert gt_ovr >= 0
            # find the proposal box that covers the best covered gt box
            box_ind = argmax_overlaps[gt_ind]
            # record the iou coverage of this gt box
            _gt_overlaps[j] = overlaps[box_ind, gt_ind]
            assert _gt_overlaps[j] == gt_ovr
            # mark the proposal box and the gt box as used
            overlaps[box_ind, :] = -1
            overlaps[:, gt_ind] = -1

        # append recorded iou coverage level
        gt_overlaps.append(_gt_overlaps)
    gt_overlaps = (
        torch.cat(gt_overlaps, dim=0) if len(gt_overlaps) else torch.zeros(0, dtype=torch.float32)
    )
    gt_overlaps, _ = torch.sort(gt_overlaps)

    if thresholds is None:
        step = 0.05
        thresholds = torch.arange(0.5, 0.95 + 1e-5, step, dtype=torch.float32)
    recalls = torch.zeros_like(thresholds)
    # compute recall for each iou threshold
    for i, t in enumerate(thresholds):
        recalls[i] = (gt_overlaps >= t).float().sum() / float(num_pos)
    # ar = 2 * np.trapz(recalls, thresholds)
    ar = recalls.mean()
    return {
        "ar": ar,
        "recalls": recalls,
        "thresholds": thresholds,
        "gt_overlaps": gt_overlaps,
        "num_pos": num_pos,
    }


# layer-wise learning rate decay for ConvNext
def get_num_layer_layer_wise(var_name_full, num_max_layer=12):
    assert var_name_full.startswith("detr.detr.backbone.0.")
    var_name = var_name_full.replace("detr.detr.backbone.0.", "")
    if var_name in ("backbone.cls_token", "backbone.mask_token", "backbone.pos_embed"):
        return 0
    elif var_name.startswith("backbone.downsample_layers"):
        stage_id = int(var_name.split('.')[2])
        if stage_id == 0:
            layer_id = 0
        elif stage_id == 1:
            layer_id = 2
        elif stage_id == 2:
            layer_id = 3
        elif stage_id == 3:
            layer_id = num_max_layer
        return layer_id
    elif var_name.startswith("backbone.stages"):
        stage_id = int(var_name.split('.')[2])
        block_id = int(var_name.split('.')[3])
        if stage_id == 0:
            layer_id = 1
        elif stage_id == 1:
            layer_id = 2
        elif stage_id == 2:
            layer_id = 3 + block_id // 3
        elif stage_id == 3:
            layer_id = num_max_layer
        return layer_id
    else:
        return num_max_layer + 1

class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        assert cfg.UNI == True
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
            os.makedirs(output_folder, exist_ok=True)
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type == "lvis":
            evaluator_list.append(LVISEvaluator(dataset_name, cfg, True, output_folder, max_dets_per_image=cfg.TEST.NUM_TEST_QUERIES))
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DetrDatasetMapper(cfg, is_train=True)
        data_loader = build_custom_train_loader(cfg, mapper=mapper)
        return data_loader


    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        mapper = DetrDatasetMapper(cfg, dataset_name, is_train=False)
        data_loader = build_detection_test_loader(cfg, dataset_name, mapper=mapper)
        return data_loader 

    @classmethod
    def build_optimizer(cls, cfg, model):
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for key, value in model.named_parameters(recurse=True):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "backbone" in key:
                lr = lr * cfg.SOLVER.BACKBONE_MULTIPLIER
                # no weight decay for grn of convnext-v2 
                if cfg.MODEL.BACKBONE.NAME == "D2ConvNeXtV2" and "grn" in key:
                    weight_decay = 0.0
            elif "sampling_offsets" in key or "reference_points" in key:
                lr = lr * cfg.SOLVER.LINEAR_PROJ_MULTIPLIER
            elif "text_encoder" in key or "lang_layers" in key:
                lr = cfg.SOLVER.LANG_LR
            elif "class_generate" in key:
                lr = cfg.SOLVER.VL_LR
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        def maybe_add_full_model_gradient_clipping(optim):  # optim: the optimizer class
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # Only support Sparse R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = DDETRSVLUniWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res
    
    @classmethod
    def test(cls, cfg, model):
        pass

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_ddetrsvluni_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def run_genu(args):
    cfg = setup(args)
    if args.eval_only:
        model = Trainer.build_model(cfg)
        print('loading...')
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        print('loading finished!')
        model.eval()
        img_path_list = args.img_path_list
        predictions = []
        import json
        for i, image_path in enumerate(img_path_list):

            image = utils.read_image(image_path, format='RGB')
            height, width = image.shape[:2]
            image_input = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

            batch_input = [dict(image=image_input)]
            with torch.no_grad():
                prediction = model(batch_input)
                pred_boxes = prediction[0]['instances'].pred_boxes.tensor
                pred_object_descriptions = prediction[0]['instances'].pred_object_descriptions.data
                sav_dict = []
                for idx, box, desc in zip(range(len(pred_boxes)), pred_boxes, pred_object_descriptions):
                    if idx == 100:
                        break
                    sav_dict.append(
                        {
                            "box": box.cpu().tolist(),
                            "description": desc,
                        }
                    )
                predictions.append(sav_dict)
        return predictions


if __name__ == "__main__":

    genu_args = argparse.Namespace(
        config_file='detr2/projects/DDETRS/configs/vg_grit5m_swinL.yaml',
        dist_url='tcp://127.0.0.1:50153',
        eval_only=True,
        machine_rank=0,
        num_gpus=1,
        num_machines=1,
        opts=['OUTPUT_DIR', 'detr2/outputs/test',
              'MODEL.WEIGHTS', 'detr2/weights/vg_grit5m_swinL.pth'],
        resume=False,
        uni=1,
        img_path_list=['223ori.jpg',
                       '241ori.jpg',
                       '248ori.jpg']
    )
    predictions = launch(
        run_genu,
        genu_args.num_gpus,
        num_machines=genu_args.num_machines,
        machine_rank=genu_args.machine_rank,
        dist_url=genu_args.dist_url,
        args=(genu_args,),
    )
    print(predictions)
