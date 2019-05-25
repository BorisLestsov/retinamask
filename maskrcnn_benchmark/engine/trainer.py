# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time

import torch
from torch.distributed import deprecated as dist

from maskrcnn_benchmark.utils.comm import get_world_size, get_rank
from maskrcnn_benchmark.utils.metric_logger import MetricLogger

from maskrcnn_benchmark.modeling.adapt.networks import GANLoss

from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.utils.comm import synchronize, get_rank, is_main_process


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k, v in loss_dict.items():
            loss_names.append(k)
            all_losses.append(v)
        all_losses = torch.stack(all_losses, dim=0)
        # dist.reduce(all_losses, dst=0)
        dist.all_reduce(all_losses)
        # if dist.get_rank() == 0:
        if True:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


# set requies_grad=Fasle to avoid computation
def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def do_train(
    models,
    data_loaders,
    data_loaders_val,
    optimizers,
    schedulers,
    checkpointers,
    device,
    checkpoint_period,
    arguments,
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = min([len(data_loader) for data_loader in data_loaders])
    start_iter = arguments["iteration"]
    need_adapt = arguments["need_adapt"]
    need_train_A = arguments["need_train_A"]
    need_train_B = arguments["need_train_B"]

    model, model_G, model_D = models
    model.train()
    model_G.train()
    model_D.train()

    optimizer, optimizer_D, optimizer_G = optimizers
    scheduler, scheduler_D, scheduler_G = schedulers
    checkpointer, checkpointer_D = checkpointers


    criterionGAN = GANLoss(use_lsgan=True).to(device)

    start_training_time = time.time()
    end = time.time()

    last_acc_D = 0

    AR_best = None
    for iteration, data in enumerate(zip(*data_loaders), start_iter):
        if len(data) == 2:
            (images, targets, _), (images_adapt, targets_adapt, _) = data
        else:
            images, targets, _ = data[0]

        if (iteration +1) % checkpoint_period == 0:
            if need_adapt:
                gt_backbone = model.module.backbone
                model.module.backbone = model_G
            model.module.training = False

            output_folders = [None] * len(data_loaders_val)
            for output_folder, data_loader_val in zip(output_folders, data_loaders_val):
                results = inference(
                    model,
                    data_loader_val,
                    iou_types=('bbox',),
                    #box_only=cfg.MODEL.RPN_ONLY,
                    box_only=True,
                    output_folder=output_folder,
                )
                synchronize()
                if is_main_process():
                    AR = results.results['box_proposal']['AR@300']
                    print("AR@300:", AR)
                    if AR_best is None or AR>AR_best:
                        AR_best = AR
                        checkpointer.save("model_best{:07d}".format(iteration+1), **arguments)
            if need_adapt:
                model.module.backbone = gt_backbone
            model.module.training = True
            model.module.rpn.training = True

        data_time = time.time() - end
        arguments["iteration"] = iteration

        scheduler.step()


        good_D = last_acc_D > 0.7 and iteration > 50 if need_adapt else True

        if need_train_A and good_D:
            images = images.to(device)
            targets = [target.to(device) for target in targets]

            loss_dict = model(images, targets, adapt=False)

            losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            meters.update(loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()


        if need_train_B and good_D:
            images = images_adapt.to(device)
            targets = [target.to(device) for target in targets_adapt]

            loss_dict = model(images, targets, adapt=False)

            losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            meters.update(loss_2=losses_reduced, **loss_dict_reduced)


            optimizer.zero_grad()
            losses.backward()
            optimizer.step()


        if need_adapt:
            #forward G
            images_adapt_A = images.to(device)
            targets_adapt_A = [target.to(device) for target in targets]
            images_adapt_B = images_adapt.to(device)
            targets_adapt_B = [target.to(device) for target in targets_adapt]

            # feat_B = model_G(images_adapt_A, targets_adapt_A, adapt=True)
            feat_A = model_G(images_adapt_A.tensors)

            # feat_A = model_G(images_adapt_B, targets_adapt_B, adapt=True)
            feat_B = model_G(images_adapt_B.tensors)

            set_requires_grad(model_D, False)
            set_requires_grad(model_G, True)
            optimizer_G.zero_grad()
            optimizer_D.zero_grad()

            #backward_G
            loss_G = 2 * criterionGAN(model_D(feat_B), True)     *     0.1
            loss_G_dict = {'loss_G': loss_G}

            losses_G = sum(loss for loss in loss_G_dict.values())

            loss_G_dict_reduced = reduce_loss_dict(loss_G_dict)
            losses_G_reduced = sum(loss for loss in loss_G_dict_reduced.values())
            meters.update(loss_G=losses_G_reduced)

            losses_G.backward(retain_graph=True)
            # losses_G.backward()
            if last_acc_D > 0.8 and iteration > 50:
                print("OPT G!")
                optimizer_G.step()

            #D
            # set_requires_grad(model_G, False)
            set_requires_grad(model_D, True)
            optimizer_D.zero_grad()
            optimizer_G.zero_grad()

            #backward D
            pred_D_A = model_D(feat_A)
            pred_D_B = model_D(feat_B)
            loss_D_synth_that_is_real = criterionGAN(pred_D_A, True)
            loss_D_real_that_is_fake = criterionGAN(pred_D_B, False)


            loss_D = (loss_D_synth_that_is_real + loss_D_real_that_is_fake) * 0.5     *     0.5
            loss_D_dict = {'loss_D': loss_D}

            loss_D_dict_reduced = reduce_loss_dict(loss_D_dict)
            losses_D = sum(loss for loss in loss_D_dict.values())

            all_A = torch.cat([t.flatten() for t in pred_D_A])
            all_B = torch.cat([t.flatten() for t in pred_D_B])
            acc_A = (all_A>0.5).sum().item()/all_A.numel()
            acc_B = (all_B<0.5).sum().item()/all_B.numel()
            last_acc_D = (acc_A + acc_B)/2

            acc_dict = {"last_acc_D" : torch.tensor(last_acc_D).cuda()}
            acc_dict_reduced = reduce_loss_dict(acc_dict)
            acc_reduced = sum(loss for loss in acc_dict_reduced.values())
            last_acc_D = acc_reduced

            meters.update(acc_D=last_acc_D)
            meters.update(acc_A=acc_A)
            meters.update(acc_B=acc_B)

            loss_D_dict_reduced = reduce_loss_dict(loss_D_dict)
            losses_D_reduced = sum(loss for loss in loss_D_dict_reduced.values())
            meters.update(loss_D=losses_D_reduced)

            loss_D.backward()
            optimizer_D.step()



        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        #if iteration % 20 == 0 or iteration == (max_iter - 1):
        if True:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        if iteration % checkpoint_period == 0 and iteration > 0:
            checkpointer.save("model_{:07d}".format(iteration+1), **arguments)

    checkpointer.save("model_{:07d}".format(iteration), **arguments)
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
