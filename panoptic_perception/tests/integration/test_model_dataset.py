import pytest
import torch
from torch.utils.data import DataLoader
from collections import defaultdict

from panoptic_perception.dataset.bdd100k_dataset import BDD100KDataset, BDDPreprocessor
from panoptic_perception.models.model_factory import ModelFactory
from panoptic_perception.losses.multi_task_loss import MultiTaskLoss


@pytest.fixture
def dataset_type():
    return "train"

@pytest.fixture
def batch_size():
    return 8

@pytest.fixture
def dataset_kwargs():
    return {
        "images_dir": "data/100k/100k",
        "detection_annotations_dir": "data/bdd100k_labels/100k",
        "segmentation_annotations_dir": "data/bdd100k_seg_maps/labels",
        "drivable_annotations_dir": "data/drivable_maps/labels",
        "preprocessor_kwargs": {
            "image_resize": (640, 640),
            "original_image_size": (720, 1280)
        }
    }

@pytest.fixture
def dataloader(dataset_kwargs, dataset_type, batch_size):
    try:
        dataset = BDD100KDataset(dataset_kwargs, dataset_type=dataset_type)
        print(f"✓ Dataset created successfully")
        print(f"  Dataset type: {dataset.dataset_type}")
        print(f"  Dataset length: {len(dataset)}")
        print(f"  Images directory: {dataset.images_dir}")
        print(f"  First few image IDs: {dataset.image_ids[:5]}")

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Use 0 for debugging, increase for training
            collate_fn=BDDPreprocessor.collate_fn
        )

        print(f"✓ DataLoader created successfully")
        print(f"  Batch size: {dataloader.batch_size}")
        print(f"  Number of batches: {len(dataloader)}")
        print(f"  Shuffle: {dataloader.shuffle if hasattr(dataloader, 'shuffle') else 'N/A'}")

        return dataloader
        
    except Exception as e:
        print(f"⚠ Dataset creation failed (expected if data not present)")
        print(f"  Error: {e}")
        print(f"  Note: Update paths in test_dataset_creation() to point to actual BDD100K data")
        
        exit(1)

@pytest.fixture
def yolo_detect_drivable_model_kwargs():
    return {
        "model_type":"yolop",
        "cfg_path":"panoptic_perception/configs/models/yolo-detection-drivable.cfg",
        "device":"cuda:0"        
    }

@pytest.fixture
def yolo_detect_lane_model_kwargs():
    return {
        "model_type":"yolop",
        "cfg_path":"panoptic_perception/configs/models/yolo-lane-detect.cfg",
        "device":"cuda:0"
    }

@pytest.fixture
def yolov8_detect_anchor_free():
    return {
        "model_type":"yolov8p",
        "cfg_path":"panoptic_perception/configs/models/yolov8-768-1280-detection-drivable-anchorFree.cfg",
        "device":"cuda:0"
    }

@pytest.fixture
def yolov8_detect_drivable():
    return {
        "model_type":"yolov8p",
        "cfg_path":"panoptic_perception/configs/models/yolov8-768-1280-detection-drivable.cfg",
        "device":"cuda:0"
    }

@pytest.fixture
def multi_task_loss_kwargs():
    return {
        "detection":{
            "_type":"detection-loss-ATSS",
            # "_type":"detection-loss-anchor-free",
            "kwargs":{}
        },
        "drivable_segmentation":{
            "_type":"segmentation-loss",
            "kwargs":{}
        },
        "lane_detection":{
            "_type":"lane-detection-loss",
            "kwargs":{}
        }
    }

@pytest.fixture
def multi_task_loss_function(multi_task_loss_kwargs):
    return MultiTaskLoss(multi_task_loss_kwargs)

def test_yolop_drivable(yolo_detect_drivable_model_kwargs, multi_task_loss_function, dataloader):
    device = yolo_detect_drivable_model_kwargs.get("device", "cuda")
    model = ModelFactory.from_config(yolo_detect_drivable_model_kwargs)

    device = torch.device(device) if torch.cuda.is_available() and "cuda" in device else torch.device("cpu")
    model.to(device)

    model.loss_function = multi_task_loss_function
    assert model.loss_function is multi_task_loss_function, \
        f"Setter didn't persist. Got: {model.loss_function}"

    for _, data_items in enumerate(dataloader):
        for k, v in data_items.items():
            if torch.is_tensor(v):
                data_items[k] = v.to(device)

        outputs = model(
            data_items["images"],
            targets={
                "drivable_area_seg": data_items.get("drivable_area_seg"),
                "lane_seg": data_items.get("segmentation_masks"),
                "detections": data_items["detections"],
                "lanes_detections": data_items.get("lanes_detections"),
                "lane_seg_masks": data_items.get("lane_seg_masks"),
                "clean_images": data_items.get("clean_images")
            }
        )

        print(f"\nModel outputs: {outputs.keys()}\n")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    print(f"  {key}: {value.shape}")
                else:
                    print(f"  {key}: shape {value.shape}")
            else:
                print(f"  {key}: {type(value)}")

        loss = torch.zeros(1, device=device)

        loss_items = defaultdict()

        if outputs.detection_loss is not None:
            loss += outputs.detection_loss
            loss_items["detection_loss"] = outputs.detection_loss
        if outputs.drivable_segmentation_loss is not None:
            loss += outputs.drivable_segmentation_loss
            loss_items["drivable_segmentation_loss"] = outputs.drivable_segmentation_loss
        if outputs.lane_segmentation_loss is not None:
            loss += outputs.lane_segmentation_loss
            loss_items["lane_segmentation_loss"] = outputs.lane_segmentation_loss
        if outputs.lane_detection_loss is not None:
            loss += outputs.lane_detection_loss
            loss_items["lane_detection_loss"] = outputs.lane_detection_loss            

        print(f'Total Loss: {loss.item()} - Loss Items: {loss_items}')

        break