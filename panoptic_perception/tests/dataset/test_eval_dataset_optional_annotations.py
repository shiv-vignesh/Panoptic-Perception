import cv2
import numpy as np

from panoptic_perception.dataset.bdd100k_dataset import BDD100KDataset
from panoptic_perception.dataset.types import DatasetMode


def test_eval_dataset_allows_missing_detection_annotations(tmp_path):
    image_id = "sample"
    images_dir = tmp_path / "images"
    seg_dir = tmp_path / "seg"
    drivable_dir = tmp_path / "drivable"

    (images_dir / "val").mkdir(parents=True)
    (seg_dir / "val").mkdir(parents=True)
    (drivable_dir / "val").mkdir(parents=True)

    image = np.full((8, 12, 3), 127, dtype=np.uint8)
    seg = np.zeros((8, 12), dtype=np.uint8)
    drivable = np.ones((8, 12), dtype=np.uint8)

    cv2.imwrite(str(images_dir / "val" / f"{image_id}.jpg"), image)
    cv2.imwrite(str(seg_dir / "val" / f"{image_id}_val_id.png"), seg)
    cv2.imwrite(str(drivable_dir / "val" / f"{image_id}_drivable_id.png"), drivable)

    dataset = BDD100KDataset(
        {
            "images_dir": str(images_dir),
            "detection_annotations_dir": "",
            "segmentation_annotations_dir": str(seg_dir),
            "drivable_annotations_dir": str(drivable_dir),
            "preprocessor_kwargs": {
                "image_resize": (16, 16),
                "original_image_size": (8, 12),
            },
        },
        dataset_type="val",
        mode=DatasetMode.EVAL,
    )

    sample = dataset[0]

    assert sample["detection_targets"].shape == (0, 5)
    assert sample["scene_attributes"] is None
    assert sample["segmentation_mask"] is not None
    assert sample["drivable_mask"] is not None

