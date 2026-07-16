import pytest
import torch

import numpy as np

from panoptic_perception.dataset.coco_image_dataset import COCODataset, COCOPreprocessor
from panoptic_perception.dataset.types import DatasetMode

@pytest.fixture
def dataset_kwargs():
    return {
            "val_dataset_kwargs":{
                "images_dir": "COCO/val2017",
                "annotations_json_path":"COCO/annotations/instances_val2017.json",
                "val_batch_size": 6,
                "val_shuffle": True,
                "val_num_workers": 4,
                "preprocessor_kwargs":{
                    "image_resize": [768, 1280],
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
                        "img_size": [768, 1280]
                    },
                    "advanced_aug": {
                        "mosaic_prob": 0.0,
                        "mixup_prob": 0.0,
                        "copy_paste_prob": 0.0
                    }
                }                 
            }
        }

@pytest.fixture
def dataloader(dataset_kwargs):

    dataset = COCODataset(
        dataset_kwargs["val_dataset_kwargs"],
        dataset_type=DatasetMode.TRAIN,
        perform_augmentation=dataset_kwargs["val_dataset_kwargs"]["preprocessor_kwargs"].get("perform_augmentation", True),
        use_super_categories=True
    )

    print(f'Dataset Created - length : {len(dataset)}')

    return torch.utils.data.DataLoader(
        dataset=dataset,
            batch_size=dataset_kwargs["val_dataset_kwargs"]["val_batch_size"],
            shuffle=dataset_kwargs["val_dataset_kwargs"].get("val_shuffle", False),
            num_workers=dataset_kwargs["val_dataset_kwargs"].get("val_num_workers", 4),
            collate_fn=COCOPreprocessor.collate_fn
    )

def test_dataloader_iteration(dataloader):

    num_iters = 5
    for idx, data_items in enumerate(dataloader):
        for k, v in data_items.items():
            if torch.is_tensor(v):
                print(f'{k} {v.shape}')
            else:
                print(f'{k} {v}')

        print()

        if idx > num_iters:
            break


def test_visualize_full_batch(dataloader):
    """Visualize entire batch: detections + drivable + lanes composited on one image per sample."""
    import cv2
    import os
 
    batch = next(iter(dataloader))

    save_dir = "visualizations/dataset_batch_vis"
    os.makedirs(save_dir, exist_ok=True)
    batch_size = batch["images"].shape[0]
    H, W = batch["images"].shape[2], batch["images"].shape[3]

    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float).view(3, 1, 1)

    for b in range(batch_size):

        img_tensor = batch["images"][b]
        img_tensor = img_tensor * std.to(img_tensor.device) + mean.to(img_tensor.device)
        
        # 3. Convert to numpy HWC format
        img = img_tensor.permute(1, 2, 0).cpu().numpy()
        
        # 4. Clip values to [0, 1] to prevent overflow before scaling to 255
        img = np.clip(img, 0.0, 1.0)
        img = (img * 255).astype(np.uint8).copy()

        if batch.get("detections") is not None:
            dets = batch["detections"]
            mask = dets[:, 0] == b
            for row in dets[mask]:
                _, cls, xc, yc, w, h = row.cpu().numpy()
                x1, y1 = int((xc - w / 2) * W), int((yc - h / 2) * H)
                x2, y2 = int((xc + w / 2) * W), int((yc + h / 2) * H)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, str(int(cls)), (x1, max(y1 - 5, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)        

        cv2.imwrite(f"{save_dir}/sample_{b}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    for b in range(batch_size):
        assert os.path.exists(f"{save_dir}/sample_{b}.png")