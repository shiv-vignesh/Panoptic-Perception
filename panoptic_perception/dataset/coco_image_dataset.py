import torch
from torch.utils.data import Dataset

import numpy as np
import os
import orjson
import random

import cv2

import albumentations as A
from collections import defaultdict
from enum import Enum

from typing import List, Dict, Tuple
from tqdm import tqdm

from panoptic_perception.dataset.types import (
    DatasetMode, Bbox, ObjDetInstance, FrameObjDetections,
    LanePoly, FrameLaneDetections,
    FrameData
)

from panoptic_perception.dataset.augmentations import (
    apply_augmentations, copy_paste_instances, random_perspective,
    mixup_augmentation, augment_hsv, flip_horizontal,
    letterbox_with_masks,
)
from panoptic_perception.dataset.mosaic_augmentation import mosaic_augmentation

from panoptic_perception.dataset.coco_class_mapping import (
    COCOCategories, COCOSupercategories, categories_in_supercategory, supercategory_of
)

from panoptic_perception.utils.logger import Logger

import warnings
warnings.simplefilter("once", UserWarning)

class COCOPreprocessor:
    def __init__(self, preprocess_kwargs:dict):

        self.preprocess_kwargs = preprocess_kwargs

        self.image_resize = preprocess_kwargs.get("image_resize", (224, 224))
        self.resized_width = self.image_resize[1]
        self.resized_height = self.image_resize[0]

        self.augment_params = preprocess_kwargs.get("augment_params", {
            # Geometric augmentations (reduced to preserve small objects)
            'degrees': 10,
            'translate': 0.1,
            'scale': 0.25,
            'shear': 5,

            # Color augmentations (kept aggressive for robustness)
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,

            # Noise
            'salt_prob': 0.005,
            'pepper_prob': 0.005,

            # Flip
            "flip_prob": 0.5,

            # Output size
            "img_size": (self.resized_height, self.resized_width)
        })

        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float)

        max_size = max(self.resized_height, self.resized_width)
        base_resize_image = [A.LongestMaxSize(max_size=max_size, interpolation=cv2.INTER_LINEAR), 
                       A.PadIfNeeded(self.resized_height, self.resized_width, 
                                    border_mode=cv2.BORDER_CONSTANT)]

        self.image_only_transformation = A.Compose(base_resize_image)

    def _load_detections(self, annotations:list, filter_by_area=False, use_super_categories:bool=True) -> FrameObjDetections:

        """
        Load 2D bbox annotations from a BDD100K detection JSON file.

        Returns:
            bboxes: list of [x1, y1, x2, y2] in original pixel coordinates
            class_labels - list of label_id
            attributes - dict of occluded, truncated, trafficLightColor
        """

        MIN_AREA = 15 * 15
        frame_detections = FrameObjDetections()

        for ann in annotations:
            category_id = ann["category_id"]
            bbox = Bbox.from_xywh(ann["bbox"])

            if bbox.valid_bbox:
                if filter_by_area and bbox.area < MIN_AREA:
                        continue

                label = COCOCategories.from_id(category_id)
                if use_super_categories:
                    super_catergory_id = supercategory_of(category_id)

                    if super_catergory_id is not None:
                        super_label = COCOSupercategories.from_id(super_catergory_id)
                        label = super_label
                        category_id = super_catergory_id
                    else:
                        warnings.warn(
                            f"Super Category missing for category: {label} Id: {category_id}"
                        )

                frame_detections.detections.append(ObjDetInstance(
                    bbox=bbox,
                    label_id=category_id,
                    label=label
                ))

        return frame_detections

    def normalize_tensor(self, tensor:torch.Tensor):

        tensor = (tensor - self.mean[:, None, None]) / self.std[:, None, None]
        return tensor
    
    def prepare_inference(self, image_path=None):
        assert image_path is not None and os.path.exists(image_path), f"Invalid Image path {image_path}"
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        orig = img.copy()
        h0, w0 = img.shape[:2]

        frame = FrameData(image=img, image_path=image_path, frame_detections=None)
        frame = letterbox_with_masks(frame, new_shape=self.image_resize)

        img = torch.from_numpy(frame.image).permute(2, 0, 1) / 255.
        img = self.normalize_tensor(img)

        return {
            "image": img.unsqueeze(0),  # add batch dim
            "original_image": orig,
            "orig_shape": (h0, w0),
            "new_shape": img.shape[1:]
        }

    @staticmethod
    def collate_fn(batch):

        batch_images = []
        batch_targets = []
        batch_image_paths = []

        for batch_idx, batch_items in enumerate(batch):
            image = batch_items['image']
            image_path = batch_items["image_path"]

            assert image is not None, f"Image tensor at batch index {batch_idx} is None."
            assert image.ndim == 3, "Image tensor must have 3 dimensions (C, H, W)."

            batch_images.append(image)
            batch_image_paths.append(image_path)

            det = batch_items['detection_targets']
            if det is not None and det.shape[0] > 0:
                nt = det.shape[0]
                batch_targets.append(
                    torch.cat([
                        torch.full((nt, 1), batch_idx, dtype=det.dtype),
                        det,
                    ], dim=1)
                )

        batch_images_tensor = torch.stack(batch_images, dim=0)
        batch_targets_tensor = (
            torch.cat(batch_targets, dim=0) if batch_targets else None
        )

        if batch_targets_tensor is not None:
            batch_targets_tensor[:, 2:4] = batch_targets_tensor[:, 2:4].clamp(0.0, 1.0)
            batch_targets_tensor[:, 4:6] = batch_targets_tensor[:, 4:6].clamp(0.001, 1.0)

        return {
            "images": batch_images_tensor,
            "detections": batch_targets_tensor,
            "image_paths": batch_image_paths
        }

class COCODataset(Dataset):
    def __init__(self, dataset_kwargs:dict, dataset_type:str='train',
                perform_augmentation:bool=False, mode:DatasetMode = DatasetMode.TRAIN,
                use_super_categories:bool=True):
        
        super().__init__()

        assert os.path.exists(dataset_kwargs["images_dir"]), f"Images directory {dataset_kwargs['images_dir']} does not exist."

        self.images_dir = dataset_kwargs['images_dir']
        self.use_super_categories = use_super_categories
        self.dataset_type = dataset_type
        self.perform_augmentation = perform_augmentation
        self.mode = mode

        self.num_classes = len(COCOSupercategories) if self.use_super_categories else len(COCOCategories)

        if mode != DatasetMode.INFER:
            #TODO, currently only object detection and segmentation supported
            assert os.path.exists(dataset_kwargs["annotations_json_path"]), f"Annotations directory {dataset_kwargs['annotations_json_path']} does not exist."

        self._build_img_to_anns(dataset_kwargs["annotations_json_path"])
        self.preprocessor = COCOPreprocessor(dataset_kwargs.get("preprocessor_kwargs", {}))

        aug_config = dataset_kwargs.get("preprocessor_kwargs", {}).get("advanced_aug", {})
        self.mosaic_prob = aug_config.get("mosaic_prob", 0.5) if perform_augmentation else 0.0
        self.mixup_prob = aug_config.get("mixup_prob", 0.15) if perform_augmentation else 0.0

    def _build_img_to_anns(self, annotations_json_path:str):

        with open(annotations_json_path, 'rb') as f:
            annotations = orjson.loads(f.read())

        self.imgs = {}
        self.img_to_anns = defaultdict(list)
        
        for img in tqdm(annotations['images'], desc=f"Building Image Index"):
            self.imgs[img['id']] = img

        for ann in tqdm(annotations['annotations'], desc=f"Building Image-to-Annotation Index"):
            self.img_to_anns[ann['image_id']].append(ann)

        if len(self.imgs) != len(self.img_to_anns):
            warnings.warn(
                f"Expected equal number of Images and Annotations, got images: {len(self.imgs)} and annotations: {len(self.img_to_anns)}"
            )

        self.img_ids = list(self.imgs.keys())

    def __len__(self):
        return len(self.img_ids)    
    
    def _load_raw(self, index):

        _image_id = self.img_ids[index]
        _image_info = self.imgs[_image_id]
        _annotations = self.img_to_anns[_image_id]

        _image_path = os.path.join(self.images_dir, _image_info["file_name"])
        if not os.path.exists(_image_path):
            raise FileNotFoundError(f"Image path {_image_path} does not exist")
        
        image = cv2.imread(_image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        frame_detections = None
        if _annotations and self.mode != DatasetMode.INFER:
            frame_detections = self.preprocessor._load_detections(_annotations)

        return FrameData(
            image=image,
            image_path=_image_path,
            frame_detections=frame_detections
        )

    def __getitem__(self, index):
        if self.mode == DatasetMode.INFER:
            _image_id = self.img_ids[index]
            _image_info = self.imgs[_image_id]
            _image_path = os.path.join(self.images_dir, _image_info["file_name"])
            
            return self.preprocessor.prepare_inference(_image_path)
        else:
            return self.prepare_training_sample(index)
        
    def prepare_training_sample(self, index):

        use_mosaic = self.perform_augmentation and random.random() < self.mosaic_prob
        use_mixup = self.perform_augmentation and random.random() < self.mixup_prob

        target_size = tuple(self.preprocessor.image_resize) # (h, w)
        aug_params  = self.preprocessor.augment_params

        # ---- 1. Build / augment the FrameData --------------------------------
        if use_mosaic:
            indices = [index] + [random.randint(0, len(self) - 1) for _ in range(3)]
            items = [self._load_raw(i) for i in indices]
            frame = mosaic_augmentation(items, output_size=target_size)

        elif use_mixup:
            # MixUp: blend two raw frames. Output stays at raw size — no
            # additional augmentation on top.
            frame1 = self._load_raw(index)
            idx2   = random.randint(0, len(self) - 1)
            frame2 = self._load_raw(idx2)
            
            frame1 = letterbox_with_masks(frame1, new_shape=target_size)
            frame2 = letterbox_with_masks(frame2, new_shape=target_size)
            frame  = mixup_augmentation(frame1, frame2)

        else:
            # Standard path
            frame = self._load_raw(index)

            if self.perform_augmentation:
                # apply_augmentations: perspective → HSV → salt/pepper → flip → letterbox
                frame = apply_augmentations(frame, aug_params, img_size=target_size)
            else:
                # No-augment path: just letterbox to target.
                frame = letterbox_with_masks(frame, new_shape=target_size)

        img_h, img_w = frame.image.shape[:2]
        
        image_tensor = torch.from_numpy(frame.image).permute(2, 0, 1) / 255.
        image_tensor = self.preprocessor.normalize_tensor(
            image_tensor
        )

        # collate_fn prepends the batch_idx column.
        detection_targets = torch.from_numpy(frame.labels_array()).float()

        scene_attributes = (
            frame.frame_detections.attributes
            if frame.frame_detections is not None else {}
        )

        return {
            "image":             image_tensor,
            "detection_targets": detection_targets,
            "image_path":        frame.image_path,
            "scene_attributes":  scene_attributes,
        }
    
class DataLoaderBuilder:

    def __init__(self, dataset_kwargs:dict, logger:Logger):
        
        self._kwargs = dataset_kwargs
        self._logger = logger
    
    def _build_loader(self, dataset:COCODataset, 
                    batch_size, shuffle:bool, num_workers:int, 
                    collate_fn):
        
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=(num_workers > 0)
        )        

    def _build_train(self) -> torch.utils.data.DataLoader:
        
        _train_dataset_kwargs = self._kwargs.get("train_dataset_kwargs", {})
        use_super_categories = self._kwargs.get("use_super_categories", False)

        if not _train_dataset_kwargs:
            KeyError(
                f"Missing train_dataset_kwargs from dataset_kwargs config"
            )

        perform_aug = _train_dataset_kwargs.get("perform_augmentation", True)
        _train_dataset = COCODataset(
            _train_dataset_kwargs,
            dataset_type="train",
            perform_augmentation=perform_aug,
            mode=DatasetMode.TRAIN,
            use_super_categories=use_super_categories
        )

        return self._build_loader(
            _train_dataset,
            batch_size=_train_dataset_kwargs["train_batch_size"],
            shuffle=_train_dataset_kwargs.get("train_shuffle", False),
            collate_fn=COCOPreprocessor.collate_fn,
            num_workers=_train_dataset_kwargs.get("train_num_workers", 4)
        )
    
    def _build_val(self) -> torch.utils.data.DataLoader:
        
        _val_dataset_kwargs = self._kwargs.get("val_dataset_kwargs", {})
        use_super_categories = self._kwargs.get("use_super_categories", False)

        if not _val_dataset_kwargs:
            KeyError(
                f"Missing val_dataset_kwargs from dataset_kwargs config"
            )

        perform_aug = _val_dataset_kwargs.get("perform_augmentation", True)
        _val_dataset = COCODataset(
            _val_dataset_kwargs,
            dataset_type="train",
            perform_augmentation=perform_aug,
            mode=DatasetMode.TRAIN,
            use_super_categories=use_super_categories
        )

        return self._build_loader(
            _val_dataset,
            batch_size=_val_dataset_kwargs["val_batch_size"],
            shuffle=_val_dataset_kwargs.get("val_shuffle", False),
            collate_fn=COCOPreprocessor.collate_fn,
            num_workers=_val_dataset_kwargs.get("val_num_workers", 4)
        )