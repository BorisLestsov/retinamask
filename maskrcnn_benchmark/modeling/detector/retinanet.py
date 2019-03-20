# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""
import os 
import torch
from torch import nn

import torchvision.transforms.functional as F

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.retinanet import build_retinanet
from maskrcnn_benchmark.modeling.roi_heads.mask_head.mask_head import build_roi_mask_head
#from maskrcnn_benchmark.modeling.roi_heads.sparsemask_head.mask_head import build_sparse_mask_head
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
import copy

class RetinaNet(nn.Module):
    """
    Main class for RetinaNet
    It consists of three main parts:
    - backbone
    - bbox_heads: BBox prediction.
    - Mask_heads:
    """

    def __init__(self, cfg):
        super(RetinaNet, self).__init__()
        self.cfg = copy.deepcopy(cfg)
        self.backbone = build_backbone(cfg)
        self.rpn = build_retinanet(cfg)
        self.mask = None
        if cfg.MODEL.MASK_ON:
            self.mask = build_roi_mask_head(cfg)
        #if cfg.MODEL.SPARSE_MASK_ON:
        #    self.mask = build_sparse_mask_head(cfg)

        # self.adapt = False
        # if self.adapt:
        #     self.forward = self.forwardCLS
        # else:
        #     self.forward = self.forwardTRAIN

    # def forward(self, *args, **kwargs):
    #     if self.adapt:
    #         return self.forwardADAPT(*args, **kwargs)
    #     else:
    #         return self.forwardTRAIN(*args, **kwargs)

    def forward(self, images, targets=None, adapt=False):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        features = self.backbone(images.tensors)

        # Retina RPN Output
        rpn_features = features
        if self.cfg.RETINANET.BACKBONE == "p2p7":
            rpn_features = features[1:]

        if adapt:
            return rpn_features

        (anchors, detections), detector_losses = self.rpn(images, rpn_features, targets)
        if self.training:
            losses = {}
            losses.update(detector_losses)
            if self.mask:
                if self.cfg.MODEL.MASK_ON:
                    # Padding the GT
                    proposals = []
                    for (image_detections, image_targets) in zip(
                        detections, targets):
                        merge_list = []
                        if not isinstance(image_detections, list):
                            merge_list.append(image_detections.copy_with_fields('labels'))

                        if not isinstance(image_targets, list):
                            merge_list.append(image_targets.copy_with_fields('labels'))

                        if len(merge_list) == 1:
                            proposals.append(merge_list[0])
                        else:
                            proposals.append(cat_boxlist(merge_list))
                    x, result, mask_losses = self.mask(features, proposals, targets)
                elif self.cfg.MODEL.SPARSE_MASK_ON:
                    x, result, mask_losses = self.mask(features, anchors, targets)

                losses.update(mask_losses)
            return losses
        else:
            if self.mask:
                proposals = []
                for image_detections in detections:
                    num_of_detections = image_detections.bbox.shape[0]
                    if num_of_detections > self.cfg.RETINANET.NUM_MASKS_TEST > 0:
                        cls_scores = image_detections.get_field("scores")
                        image_thresh, _ = torch.kthvalue(
                            cls_scores.cpu(), num_of_detections - \
                            self.cfg.RETINANET.NUM_MASKS_TEST + 1
                        )
                        keep = cls_scores >= image_thresh.item()
                        keep = torch.nonzero(keep).squeeze(1)
                        image_detections = image_detections[keep]

                    proposals.append(image_detections)

                if self.cfg.MODEL.SPARSE_MASK_ON:
                    x, detections, mask_losses = self.mask(
                        features, proposals, targets
                    )
                else:
                    x, detections, mask_losses = self.mask(features, proposals, targets)
            return detections


    def forwardADAPT(self, images, targets=None):
        # images_bgr255 = images
        images_bgr255 = ((images*0.5+0.5)*255)[:, [2,1,0], :, :]

        for i in range(3):
            images_bgr255[:, i, ...] -= self.cfg.INPUT.PIXEL_MEAN[i]
            images_bgr255[:, i, ...] /= self.cfg.INPUT.PIXEL_STD[i]
        tensor = images_bgr255

        gpu_id = int(str(tensor.device)[-1])

        #my_device = next(self.backbone.parameters()).device
        #tensor = tensor.to(my_device)
        #print("kek", my_device, images.device, tensor.device)
        features = self.backbone(tensor)

        # Retina RPN Output
        rpn_features = features
        if self.cfg.RETINANET.BACKBONE == "p2p7":
            rpn_features = features[1:]

        return rpn_features

        img_list = to_image_list(images)

        (anchors, detections), detector_losses = self.rpn(img_list, rpn_features, targets)
        box_cls, box_regression = self.rpn.box_cls, self.rpn.box_regression

        torch.save(detections, "tmp/det{}.pth".format(gpu_id))

        # box_cls, box_regression = self.rpn.head(features)
        # anchors = self.rpn.anchor_generator(img_list, features)
        # (anchors, boxes), _ = self.rpn._forward_test(anchors, box_cls, box_regression)

        return features, box_cls, box_regression

