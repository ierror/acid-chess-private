import os

import cv2
import torch
import torchvision
from glob import glob
from torchvision.io import read_image

from PIL import Image
from pycocotools.coco import COCO
import transforms as T

import numpy as np
from torchvision.transforms import functional as F
import albumentations as A


# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))


def get_transform(train):
    transforms = []
    if train:
        transforms.append(A.Rotate(limit=45, border_mode=4))
        transforms.append(A.VerticalFlip())
        transforms.append(A.HorizontalFlip())
    transforms.append(T.ToTensor())
    return A.Compose(transforms)


class ChessBoardsDataSet(torch.utils.data.Dataset):
    def __init__(self, root, annotation):
        self.root = root
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # path for input image
        img_path = coco.loadImgs(img_id)[0]['file_name']
        # open the input image
        img = Image.open(img_path).convert("RGB")

        # number of objects in the image
        num_objs = len(coco_annotation)


        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        for i in range(num_objs):
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])

        #boxes = transform(boxes=boxes)
        #boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # masks
        mask = coco.annToMask(coco_annotation[0])
        #        mask = Image.fromarray(mask).convert("RGB")

        #masks = transform(masks=np.array([mask]))
        #masks = np.array([mask])
        #masks = torch.as_tensor(masks, dtype=torch.uint8)

        # Labels (In my case, I only one class: target class or background)
        labels = torch.ones((num_objs,), dtype=torch.int64)

        # Tensorise img_id
        img_id = torch.tensor([img_id])

        # Size of bbox (Rectangular)
        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]['area'])
        areas = torch.as_tensor(areas, dtype=torch.float32)

        # Iscrowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        transform = A.Compose([
            A.Rotate(limit=45, border_mode=4),
            A.VerticalFlip(),
            A.HorizontalFlip()
        ])
        transformed = transform(image=np.array(img), mask=mask, box=boxes[0])

        # Annotation is in dictionary format
        target = {}
        target["boxes"] = torch.as_tensor(np.array([transformed["box"]]), dtype=torch.float32)
        target["masks"] = torch.as_tensor(np.array([transformed["mask"]]), dtype=torch.uint8)
        target["labels"] = labels
        target["image_id"] = img_id
        #target["area"] = areas
        target["iscrowd"] = iscrowd

        img = F.to_tensor(transformed["image"])
        img = F.normalize(img, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        img = F.convert_image_dtype(img)

        return img, target

    def __len__(self):
        return len(self.ids)

