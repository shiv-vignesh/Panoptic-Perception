"""
Simple functional tests for BDD100K dataset.

Tests basic dataset and dataloader functionality with real data paths.
"""

import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset.bdd100k_dataset import BDDPreprocessor, BDD100KDataset, visualize_batch
from dataset.enums import BDD100KClasses

def test_bdd_classes():
    """Test BDD100K class enumeration."""
    print("\n" + "="*60)
    print("TEST: BDD100K Classes")
    print("="*60)

    # Test class values
    assert BDD100KClasses.PERSON.value == 0
    assert BDD100KClasses.CAR.value == 2
    assert BDD100KClasses.TRUCK.value == 3

    # Test from_label
    assert BDD100KClasses.from_label("car") == 2
    assert BDD100KClasses.from_label("person") == 0
    assert BDD100KClasses.from_label("invalid") is None

    # Test from_id
    assert BDD100KClasses.from_id(0) == "person"
    assert BDD100KClasses.from_id(2) == "car"
    assert BDD100KClasses.from_id(99) is None

    print("✓ All BDD100K class tests passed")
    print(f"✓ Total classes: {len(BDD100KClasses)}")
    for cls in BDD100KClasses:
        print(f"  {cls.name}: {cls.value}")


def test_preprocessor():
    """Test BDD preprocessor initialization."""
    print("\n" + "="*60)
    print("TEST: BDD Preprocessor")
    print("="*60)

    preprocessor_kwargs = {
        "image_resize": (640, 640),
        "original_image_size": (720, 1280)
    }

    preprocessor = BDDPreprocessor(preprocessor_kwargs)

    assert preprocessor.image_resize == (640, 640)
    assert preprocessor.original_size == (720, 1280)
    assert preprocessor.resized_width == 640
    assert preprocessor.resized_height == 640
    assert preprocessor.original_width == 1280
    assert preprocessor.original_height == 720

    print(f"✓ Preprocessor initialized")
    print(f"  Original size: {preprocessor.original_size}")
    print(f"  Resize to: {preprocessor.image_resize}")
    print(f"  Transformation pipeline: {preprocessor.transformation is not None}")


def test_prepare_targets():
    """Test target preparation for YOLO format."""
    print("\n" + "="*60)
    print("TEST: Prepare Targets")
    print("="*60)

    preprocessor_kwargs = {
        "image_resize": (640, 640),
        "original_image_size": (720, 1280)
    }
    preprocessor = BDDPreprocessor(preprocessor_kwargs)

    # Test with sample bboxes
    boxes = [
        [100, 200, 300, 400],  # [left, top, right, bottom]
        [500, 100, 600, 300]
    ]
    labels = [2, 0]  # CAR, PERSON

    targets = preprocessor.prepare_targets_2d(boxes, labels)

    assert targets.shape == (2, 5)
    assert targets[0, 0] == 2  # class id
    assert targets[1, 0] == 0  # class id

    # Check normalized coordinates are in [0, 1]
    assert torch.all(targets[:, 1:] >= 0)
    assert torch.all(targets[:, 1:] <= 1)

    print(f"✓ Prepared {len(boxes)} targets")
    print(f"  Target shape: {targets.shape}")
    print(f"  First target: class={targets[0, 0]}, x={targets[0, 1]:.4f}, y={targets[0, 2]:.4f}, w={targets[0, 3]:.4f}, h={targets[0, 4]:.4f}")


def test_collate_function():
    """Test batch collation."""
    print("\n" + "="*60)
    print("TEST: Collate Function")
    print("="*60)

    # Create sample batch
    batch = [
        {
            "image": torch.randn(3, 640, 640),
            "segmentation_mask": torch.randint(0, 10, (640, 640)),
            "drivable_mask": torch.randint(0, 2, (640, 640)),
            "detection_targets": torch.tensor([[0, 0.5, 0.5, 0.1, 0.1]])
        },
        {
            "image": torch.randn(3, 640, 640),
            "segmentation_mask": torch.randint(0, 10, (640, 640)),
            "drivable_mask": torch.randint(0, 2, (640, 640)),
            "detection_targets": torch.tensor([
                [1, 0.3, 0.3, 0.2, 0.2],
                [0, 0.7, 0.7, 0.15, 0.15]
            ])
        }
    ]

    result = BDDPreprocessor.collate_fn(batch)

    assert result["images"].shape == (2, 3, 640, 640)
    assert result["segmentation_masks"].shape == (2, 640, 640)
    assert result["drivable_masks"].shape == (2, 640, 640)
    assert result["detection_targets"].shape == (3, 6)  # 1 + 2 targets, 6 cols (batch_idx added)

    # Check batch indices were added correctly
    assert result["detection_targets"][0, 0] == 0  # From batch 0
    assert result["detection_targets"][1, 0] == 1  # From batch 1
    assert result["detection_targets"][2, 0] == 1  # From batch 1

    print(f"✓ Collated batch of {len(batch)} samples")
    print(f"  Images shape: {result['images'].shape}")
    print(f"  Segmentation masks shape: {result['segmentation_masks'].shape}")
    print(f"  Drivable masks shape: {result['drivable_masks'].shape}")
    print(f"  Detection targets shape: {result['detection_targets'].shape}")
    print(f"  Batch indices: {result['detection_targets'][:, 0].tolist()}")


def test_dataset_creation():
    """Test creating BDD100K dataset object."""
    print("\n" + "="*60)
    print("TEST: Dataset Creation")
    print("="*60)

    # Define paths (adjust these to your actual data paths)
    dataset_kwargs = {
        "images_dir": "panoptic_perception/BDD100k/100k",
        "detection_annotations_dir": "panoptic_perception/BDD100k/bdd100k_labels/100k",
        "segmentation_annotations_dir": "panoptic_perception/BDD100k/bdd100k_seg_maps/labels",
        "drivable_annotations_dir": "panoptic_perception/BDD100k/bdd100k_drivable_maps/labels",
        "preprocessor_kwargs": {
            "image_resize": (640, 640),
            "original_image_size": (720, 1280)
        }
    }        

    try:
        dataset = BDD100KDataset(dataset_kwargs, dataset_type='train', perform_augmentation=False)
        print(f"✓ Dataset created successfully")
        print(f"  Dataset type: {dataset.dataset_type}")
        print(f"  Dataset length: {len(dataset)}")
        print(f"  Images directory: {dataset.images_dir}")
        print(f"  First few image IDs: {dataset.image_ids[:5]}")

        return dataset
    except Exception as e:
        print(f"⚠ Dataset creation failed (expected if data not present)")
        print(f"  Error: {e}")
        print(f"  Note: Update paths in test_dataset_creation() to point to actual BDD100K data")
        return None


def test_dataset_getitem(dataset=None):
    """Test getting item from dataset."""
    print("\n" + "="*60)
    print("TEST: Dataset Get Item")
    print("="*60)

    if dataset is None:
        print("⚠ Skipping test - no dataset provided")
        return
    
    
    sample = dataset[0]

    try:
        # Get first sample
        sample = dataset[0]

        print(f"✓ Successfully loaded sample 0")
        print(f"  Keys: {sample.keys()}")
        print(f"  Image shape: {sample['image'].shape}")
        print(f"  Image dtype: {sample['image'].dtype}")

        if sample['segmentation_mask'] is not None:
            print(f"  Segmentation mask shape: {sample['segmentation_mask'].shape}")

        if sample['drivable_mask'] is not None:
            print(f"  Drivable mask shape: {sample['drivable_mask'].shape}")

        if sample['detection_targets'] is not None:
            print(f"  Detection targets shape: {sample['detection_targets'].shape}")
            print(f"  Number of objects: {sample['detection_targets'].shape[0]}")

    except Exception as e:
        print(f"✗ Failed to get item from dataset")
        print(f"  Error: {e}")


def test_dataloader_creation(dataset=None):
    """Test creating DataLoader."""
    print("\n" + "="*60)
    print("TEST: DataLoader Creation")
    print("="*60)

    if dataset is None:
        print("⚠ Skipping test - no dataset provided")
        return

    try:
        dataloader = DataLoader(
            dataset,
            batch_size=16,
            shuffle=False,
            num_workers=0,  # Use 0 for debugging, increase for training
            collate_fn=BDDPreprocessor.collate_fn
        )

        print(f"✓ DataLoader created successfully")
        print(f"  Batch size: {dataloader.batch_size}")
        print(f"  Number of batches: {len(dataloader)}")
        print(f"  Shuffle: {dataloader.shuffle if hasattr(dataloader, 'shuffle') else 'N/A'}")

        return dataloader

    except Exception as e:
        print(f"✗ Failed to create DataLoader")
        print(f"  Error: {e}")
        return None


def test_dataloader_iteration(dataloader=None):
    """Test iterating through DataLoader."""
    print("\n" + "="*60)
    print("TEST: DataLoader Iteration")
    print("="*60)

    if dataloader is None:
        print("⚠ Skipping test - no dataloader provided")
        return

    batch = next(iter(dataloader))
    print(f"  Batch keys: {batch.keys()}")
    print(f"  Images shape: {batch['images'].shape}")
    print(f"  Images dtype: {batch['images'].dtype}")
    print(f"  Images min/max: {batch['images'].min():.2f} / {batch['images'].max():.2f}")    
    
    try:
        # Get one batch
        batch = next(iter(dataloader))

        print(f"✓ Successfully loaded one batch")
        print(f"  Batch keys: {batch.keys()}")
        print(f"  Images shape: {batch['images'].shape}")
        print(f"  Images dtype: {batch['images'].dtype}")
        print(f"  Images min/max: {batch['images'].min():.2f} / {batch['images'].max():.2f}")

        if batch['segmentation_masks'] is not None:
            print(f"  Segmentation masks shape: {batch['segmentation_masks'].shape}")

        if batch['drivable_area_seg'] is not None:
            print(f"  Drivable masks shape: {batch['drivable_area_seg'].shape}")

        if batch['detections'] is not None:
            print(f"  Detection targets shape: {batch['detections'].shape}")
            print(f"  Total objects in batch: {batch['detections'].shape[0]}")
            print(f"  Batch indices: {batch['detections'][:, 0].unique().tolist()}")
            
        save_dir = "panoptic_perception/BDD100k/sample_target_visualizations-2"
        for batch_idx in range(batch["images"].shape[0]):
            print(f"  Visualizing: {batch_idx}")
            visualize_batch(batch["images"], batch["segmentation_masks"],
                            batch["drivable_area_seg"], batch["detections"], 
                            save_dir=save_dir, batch_index=batch_idx)

    except Exception as e:
        print(f"✗ Failed to iterate DataLoader")
        print(f"  Error: {e}")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("BDD100K DATASET TESTS")
    print("="*60)

    # Basic tests (always run)
    # test_bdd_classes()
    # test_preprocessor()
    # test_prepare_targets()
    # test_collate_function()

    # Dataset tests (require actual data)
    dataset = test_dataset_creation()

    if dataset is not None:
        test_dataset_getitem(dataset)
        dataloader = test_dataloader_creation(dataset)

        if dataloader is not None:
            test_dataloader_iteration(dataloader)

    print("\n" + "="*60)
    print("TESTS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    run_all_tests()
