import torch
import pytest

from panoptic_perception.models.denet import LapPyramidConv
from panoptic_perception.models.denet import TransLow
from panoptic_perception.models.denet.net import DENet
from panoptic_perception.models.models import DENetYolo, YOLOv8P

IMAGE_H = 768
IMAGE_W = 1280
NC = 3
BS = 2

# ---------------
KERNEL_SIZE = 5
NUM_HIGH_FREQ = 3
CH_BLOCKS = 64

@pytest.fixture(scope="session")
def rand_x():
    return torch.rand(BS, NC, IMAGE_H, IMAGE_W)

@pytest.fixture(scope="session")
def lap_pyramid():
    return LapPyramidConv(
        num_high=NUM_HIGH_FREQ,
        kernel_size=KERNEL_SIZE,
        channels=NC
    )

@pytest.fixture(scope="session")
def trans_low():
    return TransLow(num_channels=NC, channel_blocks=CH_BLOCKS)
    
def test_laplacian_pyramid(lap_pyramid, rand_x):
    pyr = lap_pyramid.pyramid_decom(rand_x)

    assert len(pyr) == NUM_HIGH_FREQ + 1

    # Each level should halve spatial dims
    for i, level in enumerate(pyr[:-1]):
        expected_h = IMAGE_H // (2 ** i)
        expected_w = IMAGE_W // (2 ** i)
        assert level.shape == (BS, NC, expected_h, expected_w), \
            f"Detail band {i}: expected {(BS, NC, expected_h, expected_w)}, got {level.shape}"

    # Base should be smallest
    base = pyr[-1]
    assert base.shape == (BS, NC, IMAGE_H // (2 ** NUM_HIGH_FREQ), IMAGE_W // (2 ** NUM_HIGH_FREQ))

    # Reconstruction should recover original (within floating point tolerance)
    recon_order = [pyr[-1]] + list(reversed(pyr[:-1]))
    recon = lap_pyramid.pyramid_recons(recon_order)
    assert recon.shape == rand_x.shape
    assert torch.allclose(recon, rand_x, atol=1e-4), \
        f"Reconstruction error: max diff = {(recon - rand_x).abs().max().item()}"


def test_trans_low(trans_low, lap_pyramid, rand_x):
    pyr = lap_pyramid.pyramid_decom(rand_x)
    base = pyr[-1]  # coarsest level, input to TransLow

    out, guide = trans_low(base)

    # Output should match base shape (residual connection preserves dims)
    assert out.shape == base.shape, \
        f"TransLow output: expected {base.shape}, got {out.shape}"

    # Guide should be (B, 3, H, W) — same spatial dims as base
    assert guide.shape == base.shape, \
        f"TransLow guide: expected {base.shape}, got {guide.shape}"

    # Output should be non-negative (relu applied)
    assert (out >= 0).all(), "TransLow output should be non-negative (relu)"

    # Output should differ from input (network should transform)
    assert not torch.allclose(out, base, atol=1e-6), \
        "TransLow output should differ from input"


# --- DENet (full forward) ---

CH_MASK = 16

@pytest.fixture(scope="session")
def denet():
    return DENet(
        num_high=NUM_HIGH_FREQ,
        gaussian_kernel_size=KERNEL_SIZE,
        num_channels=NC,
        channel_blocks=CH_BLOCKS,
        channel_mask=CH_MASK,
    )

def test_denet_forward(denet, rand_x):
    """Test full DENet forward: decompose → trans_low → up_guide + trans_high → reconstruct."""
    with torch.no_grad():
        out = denet(rand_x)

    # Output should match input shape
    assert out.shape == rand_x.shape, \
        f"DENet output: expected {rand_x.shape}, got {out.shape}"

    # Output should differ from input (network should transform)
    assert not torch.allclose(out, rand_x, atol=1e-6), \
        "DENet output should differ from input"


def test_denet_guide_upsampling(denet, rand_x):
    """Verify guide is progressively upsampled to match each detail band."""
    with torch.no_grad():
        pyrs = denet.lap_pyramid.pyramid_decom(rand_x)
        _, guide = denet.trans_low(pyrs[-1])

        for i in range(denet.num_high):
            guide = denet.up_guide[i](guide)
            detail = pyrs[-2 - i]
            assert guide.shape == detail.shape, \
                f"Level {i}: guide {guide.shape} != detail {detail.shape}"


def test_denet_submodule_consistency(denet, lap_pyramid, rand_x):
    """DENet's pyramid should produce same results as standalone instance."""
    with torch.no_grad():
        standalone_pyrs = lap_pyramid.pyramid_decom(rand_x)
        denet_pyrs = denet.lap_pyramid.pyramid_decom(rand_x)

    for i, (s, d) in enumerate(zip(standalone_pyrs, denet_pyrs)):
        assert torch.allclose(s, d, atol=1e-6), \
            f"Pyramid level {i} mismatch between standalone and DENet"


def test_denet_gradient_flow(denet, rand_x):
    """Verify gradients flow through the full DENet pipeline."""
    x = rand_x.clone().requires_grad_(True)
    out = denet(x)
    loss = out.mean()
    loss.backward()

    assert x.grad is not None, "No gradient on input"
    assert x.grad.abs().sum() > 0, "Gradient is all zeros"


# --- DENetYolo (end-to-end with task network) ---

YOLOV8_CFG = "panoptic_perception/configs/models/yolov8-detection-anchorFree.cfg"
DENET_KWARGS = {
    "num_high": NUM_HIGH_FREQ,
    "gaussian_kernel_size": KERNEL_SIZE,
    "num_channels": NC,
    "channel_blocks": CH_BLOCKS,
    "channel_mask": CH_MASK,
    "high_channels": 32,
    "high_kernel_size": 3,
    "visualize_intermediate": True,
}

@pytest.fixture(scope="session")
def denet_yolo():
    task_network = YOLOv8P(cfg=YOLOV8_CFG)
    model = DENetYolo(task_network=task_network, denet_kwargs=DENET_KWARGS)
    model.train()
    return model


def test_denetyolo_forward_training(denet_yolo):
    """DENetYolo training forward: DENet enhances → YOLOv8P computes loss."""
    from panoptic_perception.dataset.bdd100k_dataset import BDD100KDataset, BDDPreprocessor
    from torch.utils.data import DataLoader

    dataset_kwargs = {
        "images_dir": "panoptic_perception/BDD100k/100k/100k",
        "detection_annotations_dir": "panoptic_perception/BDD100k/bdd100k_labels/100k",
        "segmentation_annotations_dir": "panoptic_perception/BDD100k/bdd100k_seg_maps/labels",
        "drivable_annotations_dir": "panoptic_perception/BDD100k/bdd100k_drivable_maps/labels",
        "preprocessor_kwargs": {
            "image_resize": (640, 640),
            "original_image_size": (720, 1280)
        }
    }

    dataset = BDD100KDataset(dataset_kwargs, dataset_type='train')
    dataloader = DataLoader(
        dataset, batch_size=2, shuffle=True,
        num_workers=0, collate_fn=BDDPreprocessor.collate_fn
    )

    batch = next(iter(dataloader))
    images = batch["images"]
    targets = {
        "detections": batch["detections"],
        "drivable_area_seg": batch.get("drivable_area_seg"),
        "lane_seg": batch.get("segmentation_masks"),
    }

    outputs = denet_yolo(images, targets)

    # Detection loss should be computed
    assert outputs.detection_loss is not None, "Detection loss is None"
    assert torch.isfinite(outputs.detection_loss), \
        f"Detection loss is not finite: {outputs.detection_loss}"

    # Enhanced image should be stored
    assert denet_yolo.enhanced_image is not None, "Enhanced image not stored"
    assert denet_yolo.enhanced_image.shape == images.shape, \
        f"Enhanced image shape mismatch: {denet_yolo.enhanced_image.shape} vs {images.shape}"


def test_denetyolo_gradient_flow(denet_yolo):
    """Gradients flow through DENet → YOLOv8P end-to-end."""
    from panoptic_perception.dataset.bdd100k_dataset import BDD100KDataset, BDDPreprocessor
    from torch.utils.data import DataLoader

    dataset_kwargs = {
        "images_dir": "panoptic_perception/BDD100k/100k/100k",
        "detection_annotations_dir": "panoptic_perception/BDD100k/bdd100k_labels/100k",
        "segmentation_annotations_dir": "panoptic_perception/BDD100k/bdd100k_seg_maps/labels",
        "drivable_annotations_dir": "panoptic_perception/BDD100k/bdd100k_drivable_maps/labels",
        "preprocessor_kwargs": {
            "image_resize": (640, 640),
            "original_image_size": (720, 1280)
        }
    }

    dataset = BDD100KDataset(dataset_kwargs, dataset_type='train')
    dataloader = DataLoader(
        dataset, batch_size=2, shuffle=True,
        num_workers=0, collate_fn=BDDPreprocessor.collate_fn
    )

    denet_yolo.zero_grad()
    batch = next(iter(dataloader))
    images = batch["images"]
    targets = {
        "detections": batch["detections"],
        "drivable_area_seg": batch.get("drivable_area_seg"),
        "lane_seg": batch.get("segmentation_masks"),
    }

    outputs = denet_yolo(images, targets)
    outputs.detection_loss.backward()

    # DENet components should receive gradients
    denet_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in denet_yolo.denet.parameters()
    )
    yolo_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in denet_yolo.task_network.parameters()
    )

    assert denet_has_grad, "No gradients in DENet"
    assert yolo_has_grad, "No gradients in YOLOv8P task network"


def test_denetyolo_defogging_loss(denet_yolo):
    """SSIM defogging loss is computed when clean_images are provided."""
    from panoptic_perception.dataset.bdd100k_dataset import BDD100KDataset, BDDPreprocessor
    from torch.utils.data import DataLoader

    dataset_kwargs = {
        "images_dir": "panoptic_perception/BDD100k/100k/100k",
        "detection_annotations_dir": "panoptic_perception/BDD100k/bdd100k_labels/100k",
        "segmentation_annotations_dir": "panoptic_perception/BDD100k/bdd100k_seg_maps/labels",
        "drivable_annotations_dir": "panoptic_perception/BDD100k/bdd100k_drivable_maps/labels",
        "preprocessor_kwargs": {
            "image_resize": (640, 640),
            "original_image_size": (720, 1280)
        }
    }

    dataset = BDD100KDataset(dataset_kwargs, dataset_type='train')
    dataloader = DataLoader(
        dataset, batch_size=2, shuffle=True,
        num_workers=0, collate_fn=BDDPreprocessor.collate_fn
    )

    batch = next(iter(dataloader))
    images = batch["images"]
    clean_images = images.clone()  # simulate clean reference
    targets = {
        "detections": batch["detections"],
        "drivable_area_seg": batch.get("drivable_area_seg"),
        "lane_seg": batch.get("segmentation_masks"),
        "clean_images": clean_images,
    }

    outputs = denet_yolo(images, targets, compute_ssim=True)

    assert outputs.defogging_loss is not None, "Defogging loss is None when clean_images provided"
    assert torch.isfinite(outputs.defogging_loss), \
        f"Defogging loss not finite: {outputs.defogging_loss}"
    assert 0.0 <= outputs.defogging_loss.item() <= 1.0, \
        f"Defogging loss out of [0,1] range: {outputs.defogging_loss.item()}"


def test_denetyolo_no_defogging_loss_without_clean(denet_yolo):
    """Defogging loss is None when clean_images not provided."""
    from panoptic_perception.dataset.bdd100k_dataset import BDD100KDataset, BDDPreprocessor
    from torch.utils.data import DataLoader

    dataset_kwargs = {
        "images_dir": "panoptic_perception/BDD100k/100k/100k",
        "detection_annotations_dir": "panoptic_perception/BDD100k/bdd100k_labels/100k",
        "segmentation_annotations_dir": "panoptic_perception/BDD100k/bdd100k_seg_maps/labels",
        "drivable_annotations_dir": "panoptic_perception/BDD100k/bdd100k_drivable_maps/labels",
        "preprocessor_kwargs": {
            "image_resize": (640, 640),
            "original_image_size": (720, 1280)
        }
    }

    dataset = BDD100KDataset(dataset_kwargs, dataset_type='train')
    dataloader = DataLoader(
        dataset, batch_size=2, shuffle=True,
        num_workers=0, collate_fn=BDDPreprocessor.collate_fn
    )

    batch = next(iter(dataloader))
    images = batch["images"]
    targets = {
        "detections": batch["detections"],
        "drivable_area_seg": batch.get("drivable_area_seg"),
        "lane_seg": batch.get("segmentation_masks"),
    }

    outputs = denet_yolo(images, targets, compute_ssim=True)
    assert outputs.defogging_loss is None, \
        "Defogging loss should be None without clean_images"


def test_denetyolo_inference(denet_yolo):
    """DENetYolo inference mode: no targets, no loss, produces predictions."""
    from panoptic_perception.dataset.bdd100k_dataset import BDD100KDataset, BDDPreprocessor
    from torch.utils.data import DataLoader

    dataset_kwargs = {
        "images_dir": "panoptic_perception/BDD100k/100k/100k",
        "detection_annotations_dir": "panoptic_perception/BDD100k/bdd100k_labels/100k",
        "segmentation_annotations_dir": "panoptic_perception/BDD100k/bdd100k_seg_maps/labels",
        "drivable_annotations_dir": "panoptic_perception/BDD100k/bdd100k_drivable_maps/labels",
        "preprocessor_kwargs": {
            "image_resize": (640, 640),
            "original_image_size": (720, 1280)
        }
    }

    dataset = BDD100KDataset(dataset_kwargs, dataset_type='val')
    dataloader = DataLoader(
        dataset, batch_size=2, shuffle=False,
        num_workers=0, collate_fn=BDDPreprocessor.collate_fn
    )

    denet_yolo.eval()
    batch = next(iter(dataloader))

    with torch.no_grad():
        outputs = denet_yolo(batch["images"])

    assert outputs.detection_loss is None, "Loss should be None in inference"
    assert outputs.defogging_loss is None, "Defogging loss should be None in inference"
    assert denet_yolo.enhanced_image is not None, "Enhanced image should be stored"

    denet_yolo.train()  # restore for other tests


def test_denetyolo_nan_check(denet_yolo):
    """No NaN in any DENetYolo parameter after forward pass."""
    for name, param in denet_yolo.named_parameters():
        assert not torch.isnan(param).any(), f"NaN in parameter: {name}"