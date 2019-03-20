# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time

import torch
from torch.distributed import deprecated as dist

from maskrcnn_benchmark.utils.comm import get_world_size, get_rank
from maskrcnn_benchmark.utils.metric_logger import MetricLogger

from maskrcnn_benchmark.modeling.adapt.networks import GANLoss


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
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
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

    model, model_D = models
    model_G = model.module.backbone
    model.train()
    model_D.train()

    optimizer, optimizer_D, optimizer_G = optimizers
    scheduler, scheduler_D, scheduler_G = schedulers
    checkpointer, checkpointer_D = checkpointers


    criterionGAN = GANLoss(use_lsgan=True).to(device)



    start_training_time = time.time()
    end = time.time()


    for iteration, ((images, targets, _), (images_adapt, targets_adapt, _)) in enumerate(zip(*data_loaders), start_iter):
        data_time = time.time() - end
        arguments["iteration"] = iteration

        scheduler.step()

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


        #forward G
        images_adapt_A = images.to(device)
        targets_adapt_A = [target.to(device) for target in targets_adapt]
        feat_A = model(images_adapt_A, targets_adapt_A, adapt=True)

        images_adapt_B = images_adapt.to(device)
        targets_adapt_B = [target.to(device) for target in targets_adapt]
        feat_B = model(images_adapt_B, targets_adapt_B, adapt=True)


        set_requires_grad(model_D, False)
        set_requires_grad(model_G, True)
        optimizer_G.zero_grad()

        #backward_G
        loss_G = 2 * criterionGAN(model_D(feat_B), True)     *     0.1
        loss_G_dict = {'loss_G': loss_G}

        losses_G = sum(loss for loss in loss_G_dict.values())

        loss_G_dict_reduced = reduce_loss_dict(loss_G_dict)
        losses_G_reduced = sum(loss for loss in loss_G_dict_reduced.values())
        meters.update(loss_G=losses_G_reduced)

        losses_G.backward(retain_graph=True)
        optimizer_G.step()



        #D
        set_requires_grad(model_D, True)
        optimizer_D.zero_grad()
        optimizer_G.zero_grad()

        #backward D
        loss_D_synth_that_is_real = criterionGAN(model_D(feat_A), True)
        loss_D_real_that_is_fake = criterionGAN(model_D(feat_B), False)


        loss_D = (loss_D_synth_that_is_real + loss_D_real_that_is_fake) * 0.5     *     0.5
        loss_D_dict = {'loss_D': loss_D}

        losses_D = sum(loss for loss in loss_D_dict.values())

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
