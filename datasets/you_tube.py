# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask
from PIL import Image
import os
import numpy as np
import datasets.transforms as T

#random.seed(123456)
np.random.seed(123456)

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        #img, target = super(CocoDetection, self).__getitem__(idx)
        coco = self.coco
        img_id = self.ids[idx]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        targets = coco.loadAnns(ann_ids)
        target_ind = np.random.randint(0, len(targets))  # choose index of a specific target
        target = targets[target_ind]
        # choose index of a target search frame. NOTE: this may result in target and search being the same image!!!
        search_ind = np.random.randint(target['frames_unique_id'].index(img_id), len(target['frames_unique_id']))
        target['seach_image_id'] = target['frames_unique_id'][search_ind]
        target['search_image_name'] = target['fr_in_vid'][search_ind]
        target['search_bbox'] = target['bboxes'][target['search_image_name']]
        path = coco.loadImgs(img_id)[0]['file_name']
        assert target['image_name'].startswith(path)

        image_type = os.path.splitext(target['image_name'])[1]
        template_image = Image.open(os.path.join(self.root, target['video_name'], target['image_name'])).convert('RGB')
        search_image = Image.open(os.path.join(self.root, target['video_name'], target['search_image_name']+image_type)).convert('RGB')

        target = {'image_id': img_id, 'annotations': target}
        img = [template_image, search_image]
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image[0].size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        #assert len(anno) == 1  # only one object per annotation
        #anno = anno[0]
        #anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [anno["bbox"], anno["search_bbox"]]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = anno["category_id"]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None

        keep = torch.all((boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0]))
        #boxes = boxes[keep]
        #classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id

        # for conversion to coco api
        area = torch.tensor(anno["area"])
        iscrowd = torch.tensor(0)
        target["area"] = area
        target["iscrowd"] = iscrowd

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        T.ListToTensor()
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
#            T.RandomSelect(  #  at this point we do not use random crop, need to decide the usage metodology
                T.Resize((640, 480)),
#                T.Compose([
#                    T.RandomResize([400, 500, 600]),
#                    T.RandomSizeCrop(384, 600),
#                    T.RandomResize(scales, max_size=1333),
#                ])
#            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.Resize((640, 480)),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    image_root = Path(args.youtube_image_path)
    json_root = Path(args.youtube_json_path)
    assert image_root.exists(), f'provided COCO path {root} does not exist'
    assert json_root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (image_root / "train" / "JPEGImages", json_root / "train.json"),
        "val": (image_root / "train" / "JPEGImages", json_root / "val.json")
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks)
    return dataset
