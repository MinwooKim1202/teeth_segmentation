# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import collections
import cv2
import numpy as np
from operator import itemgetter

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.engine.trainer import do_train
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir

from predictor import COCODemo
from model_evaluator import Model_evaluator

parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
parser.add_argument(
    "--config-file",
    default="",
    metavar="FILE",
    help="path to config file",
    type=str,
)
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument(
    "--skip-test",
    dest="skip_test",
    help="Do not test the final model",
    action="store_true",
)
parser.add_argument(
    "opts",
    help="Modify config options using the command-line",
    default=None,
    nargs=argparse.REMAINDER,
)
parser.add_argument(
    "--weight",
    default="",
    type=str,
)

args = parser.parse_args()

num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
args.distributed = num_gpus > 1

if args.distributed:
    torch.cuda.set_device(args.local_rank)
    torch.distributed.deprecated.init_process_group(
        backend="nccl", init_method="env://"
    )

cfg.merge_from_file(args.config_file)
cfg.merge_from_list(args.opts)
cfg.freeze()

output_dir = cfg.OUTPUT_DIR
if output_dir:
    mkdir(output_dir)

logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
logger.info("Using {} GPUs".format(num_gpus))
logger.info(args)

logger.info("Collecting env info (might take some time)")
logger.info("\n" + collect_env_info())

logger.info("Loaded configuration file {}".format(args.config_file))
with open(args.config_file, "r") as cf:
    config_str = "\n" + cf.read()
    logger.info(config_str)
logger.info("Running with config:\n{}".format(cfg))

cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

coco_demo = COCODemo(
    cfg,
    args.weight,
    min_image_size=800,
    confidence_threshold=0.5,
)

model_evaluator = Model_evaluator()


def train(cfg, local_rank, distributed, weight):
    model = build_detection_model(cfg)
    """
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    if distributed:
        model = torch.nn.parallel.deprecated.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )

    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.OUTPUT_DIR

    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )
    # extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
    extra_checkpoint_data = checkpointer.load(weight)
    arguments.update(extra_checkpoint_data)

    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    do_train(
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
    )
    """
    return model


def test(cfg, model, distributed):
    """
    if distributed:
        model = model.module
    torch.cuda.empty_cache()  # TODO check if it helps
    """
    iou_types = ("bbox",)

    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",) # ('bbox', 'segm')
    output_folders = [None] * len(cfg.DATASETS.TEST)
    #print("-----------",cfg.OUTPUT_DIR)
    if cfg.OUTPUT_DIR:
        dataset_names = cfg.DATASETS.TEST

        for idx, dataset_name in enumerate(dataset_names):
            #print("-----",idx)
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)

    for data_loader_v in data_loaders_val:
        for i, batch in enumerate(data_loader_v):
            data = [[ [], [], [], [] ], [[], [], [], [], []]] # making data list
            print(i)
            images, targets, return_list = batch
            file_name = return_list[0][1] # image file name
            label = targets[0].get_field('labels')  # Label

            img = images.tensors.numpy()
            img = np.reshape(img, (3,800,1216))
            img = img.transpose(1, 2, 0) # Opencv Format으로 변환
            img = (img *  cfg.INPUT.PIXEL_STD) + cfg.INPUT.PIXEL_MEAN # normalizere 복원
            img = img.astype('uint8') #PIL format으로

            predictions = coco_demo.compute_prediction(img) # inference
            top_predictions = coco_demo.select_top_predictions(predictions) # select top inference data
            pred_class = top_predictions.get_field("labels").tolist() # pred class
            pred_bbox = top_predictions.bbox.tolist()  # pred bbox
            polygon_list = model_evaluator.get_polygon(top_predictions) # pred polygon


            data[0][0] = file_name
            data[0][1] = label.tolist()
            data[0][2] = targets[0].bbox.tolist() # label bbox
            data[0][3] = targets[0].get_field('masks').polygons

            data[1][0] = file_name
            data[1][1] = pred_class
            data[1][2] = pred_bbox
            data[1][3] = polygon_list
            data[1][4] = top_predictions.get_field("scores").tolist()


            data = model_evaluator.label_to_pred_matching(data)
            sort_data = model_evaluator.sorted_data(data)

            model_evaluator.write_inference_csv("/home/hagler/lettuce_segmentation_msrcnn/report/inference_data.csv", sort_data) # make inference csv

            model_evaluator.write_confusionmatrix_csv("/home/hagler/lettuce_segmentation_msrcnn/report/", sort_data) # make confusion_matrix


            """
            label_img_name = '/home/hagler/lettuce_segmentation_msrcnn/datasets/coco/JPEGImages/' + data[0][0]
            label_img = cv2.imread(label_img_name, cv2.IMREAD_COLOR)
            label_img = cv2.resize(label_img, dsize=(1216, 800), interpolation=cv2.INTER_AREA)

            """
            #cv2.imwrite('./minwoo_result/' + 'overlay/' + file_name, predictions_img)
            #cv2.imwrite('./minwoo_result/' + 'mask/' + file_name, predictions_mask_img)
            #cv2.imwrite('./minwoo_result/' + file_name + '.jpg', img)

def main():

    model = train(cfg, args.local_rank, args.distributed, args.weight)

    # manual override some options

    if not args.skip_test:
        test(cfg, model, args.distributed)

if __name__ == "__main__":
    main()
