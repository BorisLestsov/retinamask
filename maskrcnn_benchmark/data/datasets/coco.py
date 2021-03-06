# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torchvision
import os
import ujson
import numpy as np

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask

from maskrcnn_benchmark.utils.comm import is_main_process, get_rank, synchronize

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class COCODataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
        self, ann_file, root, remove_images_without_annotations, transforms=None
    ):
        # ann_f = ann_file
        ann_f = ann_file + ('0' if is_main_process() else '')
        super(COCODataset, self).__init__(root, ann_f)
        synchronize()

        # sort indices for reproducible results
        self.root_json = os.path.dirname(root)
        self.ids = sorted(self.ids)

        # filter images without detection annotations
        # if remove_images_without_annotations:
        #     self.ids = [
        #         img_id
        #         for img_id in self.ids
        #         if len(self.coco.getAnnIds(imgIds=img_id, iscrowd=None)) > 0
        #     ]

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self.transforms = transforms

        # img, anno = super(COCODataset, self).__getitem__(0)
        # print(get_rank(), anno)

    def get_non_trans(self, idx):
        img, anno = super(COCODataset, self).__getitem__(idx)
        return np.asarray(img)


    def __getitem__(self, idx):
        img, anno = super(COCODataset, self).__getitem__(idx)

        anno = []
        with open(os.path.join(self.root_json, "json", str(self.ids[idx])), 'r') as f:
            for line in f:
                anno.append(ujson.loads(line))


        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        # CLASSES!!!
        # classes = [obj["category_id"] for obj in anno]
        classes = [1 for obj in anno]



        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        masks = [obj["segmentation"] for obj in anno]
        masks = SegmentationMask(masks, img.size)
        target.add_field("masks", masks)

        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data
