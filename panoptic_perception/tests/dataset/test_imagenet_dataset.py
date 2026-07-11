import pytest
import torch

from panoptic_perception.dataset.types import DatasetMode
from panoptic_perception.dataset.imagenet_dataset import ImageNetDataset, ImageNetPreprocessor

@pytest.fixture
def dataset_kwargs():

    return {
        "root":"imagenet-tiny/tiny-imagenet-200",
        "train_batch_size": 6,
        "train_shuffle": True,
        "train_num_workers": 4,
        "train_preprocessor_kwargs":{
            "image_resize": [224, 224],
            "perform_augmentation": True,
            "augment_params": {
                "degrees": 5,
                "translate": 0.05,
                "scale": 0.15,
                "shear": 2,
                "hsv_h": 0.015,
                "hsv_s": 0.7,
                "hsv_v": 0.4,
                "salt_prob": 0.005,
                "pepper_prob": 0.005,
                "flip_prob": 0.5,
                "img_size": [224, 224]
            }
        },
        "val_batch_size": 16,
        "val_shuffle": False,
        "val_num_workers": 4,
        "val_preprocessor_kwargs":{
            "image_resize": [224, 224]
        }
    }

def _base_kwargs(dataset_kwargs, preprocessor_kwargs: dict):
    return {
        "root":dataset_kwargs["root"],
        "preprocessor_kwargs":preprocessor_kwargs
    }

@pytest.fixture
def train_dataloader(dataset_kwargs):

    _kwargs = _base_kwargs(dataset_kwargs, dataset_kwargs.get("train_preprocessor_kwargs", {}))
    perform_aug = _kwargs["preprocessor_kwargs"].get("perform_augmentation", True)

    train_dataset = ImageNetDataset(
        _kwargs,
        dataset_type="train",
        perform_augmentation=perform_aug,
        mode=DatasetMode.TRAIN
    )

    return torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=dataset_kwargs.get("train_batch_size", 4),
        num_workers=dataset_kwargs.get("train_num_workers", 0),
        shuffle=dataset_kwargs.get("train_shuffle", False),
        collate_fn=ImageNetPreprocessor.collate_fn
    )

@pytest.fixture
def val_dataloader(dataset_kwargs):

    _kwargs = _base_kwargs(dataset_kwargs, dataset_kwargs.get("val_preprocessor_kwargs", {}))
    perform_aug = _kwargs["preprocessor_kwargs"].get("perform_augmentation", True)    

    train_dataset = ImageNetDataset(
        _kwargs,
        dataset_type="val",
        perform_augmentation=perform_aug,
        mode=DatasetMode.EVAL
    )

    return torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=dataset_kwargs.get("val_batch_size", 4),
        num_workers=dataset_kwargs.get("val_num_workers", 0),
        shuffle=dataset_kwargs.get("val_shuffle", False),
        collate_fn=ImageNetPreprocessor.collate_fn
    )

def test_dataloader_iter(train_dataloader, val_dataloader, dataset_kwargs, max_iters:int = 5):

    _h, _w = dataset_kwargs.get("train_preprocessor_kwargs", {}).get("image_resize", (224, 224))

    def _is_tensor(_input):
        return isinstance(_input, torch.Tensor)
    
    def _is_bchw(image:torch.Tensor):
        return image.ndim == 4 and image.shape[1] in [3] and (image.shape[-2], image.shape[-1]) == (_h, _w)

    for idx, batch_data in enumerate(train_dataloader):
        images = batch_data["images"]
        labels = batch_data["labels"]

        assert _is_tensor(images), f'Got Invalid images type'
        assert _is_bchw(images), f"Got Invalid Image Dimensions, {images.shape}"
        assert labels.shape[0] == train_dataloader.batch_size, f"Labels got invalid batch size, {labels.shape}"

        if idx == max_iters:
            break

    for idx, batch_data in enumerate(val_dataloader):
        images = batch_data["images"]
        labels = batch_data["labels"]

        assert _is_tensor(images), f'Got Invalid images type'
        assert _is_bchw(images), f"Got Invalid Image Dimensions, {images.shape}"
        assert labels.shape[0] == val_dataloader.batch_size, f"Labels got invalid batch size, {labels.shape}"

        if idx == max_iters:
            break