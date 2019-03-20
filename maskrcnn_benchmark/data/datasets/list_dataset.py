# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Simple dataset class that wraps a list of path names
"""

from torch.utils.data.dataset import Dataset
from PIL import Image
import cv2
import os
import imagesize

from maskrcnn_benchmark.structures.bounding_box import BoxList


class ListDataset(Dataset):
    def __init__(self, root, transforms=None):
        super(Dataset, self).__init__()
        self.root = root
        self.imglist = []
        with open(os.path.join(self.root, "imglist.txt"), 'r') as f:
            for line in f:
                self.imglist.append(os.path.join(self.root, line.split()[0]))

        self.transforms = transforms

    def __getitem__(self, item):
        img = Image.open(self.imglist[item]).convert("RGB")

        # dummy target
        w, h = img.size
        target = BoxList([[0, 0, w, h]], img.size, mode="xyxy")

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, item

    def __len__(self):
        return len(self.imglist)

    def get_img_info(self, item):
        """
        Return the image dimensions for the image, without
        loading and pre-processing it
        """
        return {"height": 512, "width": 384}
        # width, height = imagesize.get(self.imglist[item])
        # print(width, height)
        # return {"height": height, "width": width}
