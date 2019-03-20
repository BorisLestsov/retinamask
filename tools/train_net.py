# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer, make_optimizer_D
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.engine.trainer import do_train
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer, Checkpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir

from maskrcnn_benchmark.modeling.adapt.networks import define_D

def train(cfg, local_rank, distributed):
    model_det = build_detection_model(cfg)
    model_D = define_D(256, 64, which_model_netD='det', n_layers_D=5)
    print(model_D)
    models = [model_det, model_D]

    device = torch.device(cfg.MODEL.DEVICE)
    for model in models:
        model.to(device)

    optimizer_det = make_optimizer(cfg, model_det)
    scheduler_det = make_lr_scheduler(cfg, optimizer_det)

    optimizer_G = make_optimizer(cfg, model_det.backbone)
    scheduler_G = make_lr_scheduler(cfg, optimizer_G)

    optimizer_D = make_optimizer_D(cfg, model_D)
    scheduler_D = None

    optimizers = [optimizer_det, optimizer_D, optimizer_G]
    schedulers = [scheduler_det, scheduler_D, scheduler_G]

    if distributed:
        for i, model in enumerate(models):
            models[i] = torch.nn.parallel.deprecated.DistributedDataParallel(
                models[i], device_ids=[local_rank], output_device=local_rank,
                # this should be removed if we update BatchNorm stats
                broadcast_buffers=False,
            )

    arguments = {}
    manual_iter = 30000
    print("WARNING! MANUAL ITERATION IS", manual_iter)
    arguments["iteration"] = manual_iter

    output_dir = cfg.OUTPUT_DIR

    save_to_disk = get_rank() == 0
    checkpointer_det = DetectronCheckpointer(
        cfg, model_det, optimizer_det, scheduler_det, output_dir, save_to_disk
    )
    checkpointer_D = Checkpointer(
        model_D, optimizer_D, None, output_dir, save_to_disk
    )
    checkpointers = [checkpointer_det, checkpointer_D]
    print('WARNING! REMOVED "iteration" from train_net.py')
    extra_checkpoint_data = checkpointer_det.load(cfg.MODEL.WEIGHT)
    extra_checkpoint_data = {"iteration" : 0}
    arguments.update(extra_checkpoint_data)

    data_loaders = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    do_train(
        models,
        data_loaders,
        optimizers,
        schedulers,
        checkpointers,
        device,
        checkpoint_period,
        arguments,
    )

    return model


def test(cfg, model, distributed):
    if distributed:
        model = model.module
    torch.cuda.empty_cache()  # TODO check if it helps
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    if cfg.OUTPUT_DIR:
        dataset_names = cfg.DATASETS.TEST
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    for output_folder, data_loader_val in zip(output_folders, data_loaders_val):
        inference(
            model,
            data_loader_val,
            iou_types=iou_types,
            #box_only=cfg.MODEL.RPN_ONLY,
            box_only=False if cfg.RETINANET.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
        )
        synchronize()


def main():
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

    model = train(cfg, args.local_rank, args.distributed)

    if not args.skip_test:
        test(cfg, model, args.distributed)


if __name__ == "__main__":
    main()
