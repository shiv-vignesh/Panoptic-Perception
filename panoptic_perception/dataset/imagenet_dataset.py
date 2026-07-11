from pathlib import Path

import random
import cv2
import numpy as np

import torch
from torch.utils.data import Dataset

from panoptic_perception.dataset.types import DatasetMode, FrameData
from panoptic_perception.dataset.augmentations import apply_augmentations, letterbox_with_masks

from panoptic_perception.utils.logger import Logger

class ImageNetPreprocessor:
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

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __call__(self, frame:FrameData, perform_augmentation:bool=True) -> torch.Tensor:
        
        if perform_augmentation:
            frame = apply_augmentations(frame, self.augment_params, self.image_resize)
        else:
            frame = letterbox_with_masks(frame, self.image_resize)

        img_np = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_np = (img_np - self.mean) / self.std

        return torch.from_numpy(img_np).permute(2, 0, 1).contiguous()

    @staticmethod
    def collate_fn(batch):
        
        batch_images = []
        batch_labels = []
        batch_image_paths = []

        for batch_item in batch:
            batch_images.append(batch_item["image"])
            batch_labels.append(batch_item["label"])
            batch_image_paths.append(batch_item["path"])

        batch_images = torch.stack(batch_images, dim=0).float()
        batch_labels = torch.stack(batch_labels, dim=0).long()

        return {
            "images": batch_images,
            "labels": batch_labels,
            "image_paths": batch_image_paths,
        }

class ImageNetDataset(Dataset):
    _IMG_EXTS = (".jpg", ".jpeg", ".png")

    #TODO, add support for wnid to string literal of the class_name
    def __init__(self, dataset_kwargs:dict, dataset_type:str, 
                perform_augmentation:bool=False, mode: DatasetMode = DatasetMode.TRAIN):
        
        super(ImageNetDataset, self).__init__()

        root = Path(dataset_kwargs["root"])
        split_dir = root / dataset_type

        assert split_dir.is_dir(), f'missing split dir: {split_dir}'

        self.dataset_type = dataset_type
        self.mode = mode
        self.preprocessor = ImageNetPreprocessor(dataset_kwargs.get("preprocess_kwargs", {}))
        self.perform_augmentation = perform_augmentation and dataset_type == "train"
        
        self.wnids = sorted(d.name for d in split_dir.iterdir() if d.is_dir())
        self.wnid_to_idx = {w: i for i, w in enumerate(self.wnids)}
        self.num_classes = len(self.wnids)

        self.samples = []
        for wnid in self.wnids:
            class_idx = self.wnid_to_idx[wnid]
            for p in (split_dir / wnid).rglob("*"):
                if p.is_file() and p.suffix.lower() in self._IMG_EXTS:
                    self.samples.append((str(p), class_idx))

        assert self.samples, f"no labelled samples under {split_dir}"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        
        image_path, class_idx = self.samples[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        frame_data = FrameData(
            image=image, 
            image_path=image_path
        )

        img_tensor = self.preprocessor(frame_data, perform_augmentation=self.perform_augmentation)
        return {
            "image":img_tensor,
            "label":torch.tensor(class_idx, dtype=torch.long),
            "path":image_path
            #TODO, string literal of the class_name
        }
        
class DataLoaderBuilder:

    def __init__(self, dataset_kwargs:dict, logger:Logger):
        
        self._kwargs = dataset_kwargs
        self._logger = logger

    def _base_kwargs(self, preprocessor_kwargs: dict):
        return {
            "root":self._kwargs["root"],
            "preprocessor_kwargs":preprocessor_kwargs
        }
    
    def _build_loader(self, dataset:ImageNetDataset, 
                    batch_size:int, shuffle:bool, num_workers:int,
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
        _kwargs = self._base_kwargs(self._kwargs.get("train_preprocessor_kwargs", {}))
        perform_aug = _kwargs["preprocessor_kwargs"].get("perform_augmentation", True)

        train_dataset = ImageNetDataset(
            _kwargs,
            dataset_type="train",
            perform_augmentation=perform_aug,
            mode=DatasetMode.TRAIN
        )

        return self._build_loader(
            dataset=train_dataset,
            batch_size=self._kwargs["train_batch_size"],
            shuffle=self._kwargs.get("train_shuffle", False),
            collate_fn=ImageNetPreprocessor.collate_fn,
            num_workers=self._kwargs.get("train_num_workers", 4)
        )
    
    def _build_val(self) -> torch.utils.data.DataLoader:
        _kwargs = self._base_kwargs(self._kwargs.get("train_preprocessor_kwargs", {}))
        perform_aug = _kwargs["preprocessor_kwargs"].get("perform_augmentation", True)

        train_dataset = ImageNetDataset(
            _kwargs,
            dataset_type="val",
            perform_augmentation=perform_aug,
            mode=DatasetMode.TRAIN
        )

        return self._build_loader(
            dataset=train_dataset,
            batch_size=self._kwargs["val_batch_size"],
            shuffle=self._kwargs.get("val_shuffle", False),
            collate_fn=ImageNetPreprocessor.collate_fn,
            num_workers=self._kwargs.get("val_num_workers", 4)
        )