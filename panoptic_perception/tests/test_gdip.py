import torch
import pytest
from torch.utils.data import DataLoader

from panoptic_perception.models.gdip.gdip import GDIP, MultiLevelGDIP
from panoptic_perception.models.gdip.vision_encoder import build_vision_encoder, TorchVisionEncoder
from panoptic_perception.models.models import GDIPYolo

# ─── Vision Encoder Tests ────────────────────────────────────────────

def test_build_vision_encoder_vgg16():
    """Factory builds a VGG16 encoder with correct output dim."""
    encoder_cfg = {
        "type": "vgg16",
        "latent_dim": 256,
        "pretrained": False,
        "tap_layers": None
    }
    encoder = build_vision_encoder(encoder_cfg)
    assert isinstance(encoder, TorchVisionEncoder)

    x = torch.rand(2, 3, 224, 224)
    out = encoder(x)
    assert out.shape == (2, 256), f"Expected (2, 256), got {out.shape}"
    assert torch.isfinite(out).all(), "Encoder output contains NaN/Inf"

def test_build_vision_encoder_resnet18():
    """Factory builds a ResNet18 encoder with correct output dim."""
    encoder_cfg = {
        "type": "resnet18",
        "latent_dim": 128,
        "pretrained": False,
        "tap_layers": None
    }
    encoder = build_vision_encoder(encoder_cfg)
    assert isinstance(encoder, TorchVisionEncoder)

    x = torch.rand(2, 3, 224, 224)
    out = encoder(x)
    assert out.shape == (2, 128), f"Expected (2, 128), got {out.shape}"

def test_build_vision_encoder_invalid():
    """Factory raises ValueError for unknown encoder type."""
    encoder_cfg = {
        "type": "nonexistent_model_xyz",
        "latent_dim": 256,
        "pretrained": False,
    }
    with pytest.raises(ValueError, match="Unknown encoder"):
        build_vision_encoder(encoder_cfg)

def test_vision_encoder_with_tap_layers():
    """VGG16 encoder with tap_layers returns latent + list of feature maps."""
    encoder_cfg = {
        "type": "vgg16",
        "latent_dim": 256,
        "pretrained": False,
        "tap_layers": [4, 9, 16]
    }
    encoder = build_vision_encoder(encoder_cfg)

    x = torch.rand(2, 3, 224, 224)
    result = encoder(x)

    assert isinstance(result, tuple), "With tap_layers, encoder should return (latent, features)"
    latent, features = result
    assert latent.shape == (2, 256)
    assert len(features) == 3, f"Expected 3 tapped features, got {len(features)}"

def test_vision_encoder_no_tap_layers_returns_tensor():
    """Without tap_layers, encoder returns a single tensor."""
    encoder_cfg = {
        "type": "vgg16",
        "latent_dim": 256,
        "pretrained": False,
        "tap_layers": None
    }
    encoder = build_vision_encoder(encoder_cfg)

    x = torch.rand(1, 3, 224, 224)
    result = encoder(x)
    assert isinstance(result, torch.Tensor), "Without tap_layers, should return tensor"

def test_vision_encoder_gradient_flow():
    """Gradients flow through the vision encoder."""
    encoder_cfg = {
        "type": "resnet18",
        "latent_dim": 128,
        "pretrained": False,
    }
    encoder = build_vision_encoder(encoder_cfg)
    x = torch.rand(1, 3, 224, 224, requires_grad=True)
    out = encoder(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "No gradient on input"
    assert x.grad.shape == x.shape

# ─── GDIP Module Tests ───────────────────────────────────────────────

def test_gdip_forward_shape():
    """GDIP forward produces correct output shapes."""
    latent_dim = 256
    batch_size = 2
    h, w = 128, 128

    gdip = GDIP(latent_dim=latent_dim, num_gates=7)
    x = torch.rand(batch_size, 3, h, w)
    latent_out = torch.rand(batch_size, latent_dim)

    enhanced, gate = gdip(x, latent_out)

    assert enhanced.shape == x.shape, f"Shape mismatch: {enhanced.shape} vs {x.shape}"
    assert gate.shape == (batch_size, 7), f"Gate shape: {gate.shape}"

def test_gdip_gate_range():
    """Gate values are in [0.01, 1.0] range (tanh_range constraint)."""
    latent_dim = 256
    gdip = GDIP(latent_dim=latent_dim, num_gates=7)
    x = torch.rand(4, 3, 64, 64)
    latent_out = torch.rand(4, latent_dim)

    _, gate = gdip(x, latent_out)

    assert gate.min() >= 0.01, f"Gate min {gate.min():.4f} below 0.01"
    assert gate.max() <= 1.0, f"Gate max {gate.max():.4f} above 1.0"

def test_gdip_output_finite():
    """GDIP output contains no NaN or Inf values."""
    latent_dim = 128
    gdip = GDIP(latent_dim=latent_dim, num_gates=7)
    x = torch.rand(2, 3, 64, 64)
    latent_out = torch.rand(2, latent_dim)

    enhanced, gate = gdip(x, latent_out)

    assert torch.isfinite(enhanced).all(), "Enhanced image contains NaN/Inf"
    assert torch.isfinite(gate).all(), "Gate contains NaN/Inf"

def test_gdip_output_normalized():
    """GDIP output is in [0, 1] range (due to normalize + identity blend)."""
    latent_dim = 128
    gdip = GDIP(latent_dim=latent_dim, num_gates=7)
    x = torch.rand(2, 3, 64, 64)
    latent_out = torch.rand(2, latent_dim)

    enhanced, _ = gdip(x, latent_out)

    assert enhanced.min() >= 0.0, f"Output min {enhanced.min():.4f} below 0"
    assert enhanced.max() <= 1.0 + 1e-5, f"Output max {enhanced.max():.4f} above 1"

def test_gdip_gradient_flow():
    """Gradients flow through GDIP for end-to-end training."""
    latent_dim = 128
    gdip = GDIP(latent_dim=latent_dim, num_gates=7)
    x = torch.rand(1, 3, 64, 64, requires_grad=True)
    latent_out = torch.rand(1, latent_dim, requires_grad=True)

    enhanced, gate = gdip(x, latent_out)
    loss = enhanced.sum()
    loss.backward()

    assert x.grad is not None, "No gradient on image input"
    assert latent_out.grad is not None, "No gradient on latent input"

def test_gdip_different_resolutions():
    """GDIP handles different spatial resolutions."""
    latent_dim = 128
    gdip = GDIP(latent_dim=latent_dim, num_gates=7)
    latent_out = torch.rand(1, latent_dim)

    for h, w in [(64, 64), (128, 256), (768, 1280)]:
        x = torch.rand(1, 3, h, w)
        enhanced, gate = gdip(x, latent_out)
        assert enhanced.shape == (1, 3, h, w), f"Failed for resolution {h}x{w}"

def test_gdip_batch_independence():
    """Each image in a batch is enhanced independently (per-image normalization)."""
    latent_dim = 128
    gdip = GDIP(latent_dim=latent_dim, num_gates=7)
    gdip.eval()

    x_single = torch.rand(1, 3, 64, 64)
    latent_single = torch.rand(1, latent_dim)

    # Seed before each forward to make GaussianBlur (stochastic sigma) deterministic
    torch.manual_seed(42)
    with torch.no_grad():
        out_single, _ = gdip(x_single, latent_single)

    # Run same image in a batch with a different image
    x_batch = torch.cat([x_single, torch.rand(1, 3, 64, 64)], dim=0)
    latent_batch = torch.cat([latent_single, torch.rand(1, latent_dim)], dim=0)

    torch.manual_seed(42)
    with torch.no_grad():
        out_batch, _ = gdip(x_batch, latent_batch)

    # First image output should be identical regardless of batch composition
    assert torch.allclose(out_single, out_batch[:1], atol=1e-5), \
        "Per-image normalization violated: batch composition changed output"

# ─── End-to-End: Encoder + GDIP ──────────────────────────────────────

def test_encoder_gdip_end_to_end():
    """Full pipeline: encoder produces latent → GDIP enhances image."""
    latent_dim = 256
    encoder_cfg = {
        "type": "resnet18",
        "latent_dim": latent_dim,
        "pretrained": False,
    }
    encoder = build_vision_encoder(encoder_cfg)
    gdip = GDIP(latent_dim=latent_dim, num_gates=7)

    x = torch.rand(2, 3, 224, 224, requires_grad=True)

    # Encoder → latent
    latent_out = encoder(x)
    enhanced, gate = gdip(x, latent_out)

    assert enhanced.shape == x.shape
    assert gate.shape == (2, 7)

    # Gradient flows end-to-end
    loss = enhanced.sum()
    loss.backward()
    assert x.grad is not None, "No gradient through encoder+GDIP pipeline"

# ─── MultiLevelGDIP Tests ────────────────────────────────────────────

def test_mgdip_forward_shape():
    """MGDIP forward produces correct output shapes."""
    latent_dim = 128
    num_blocks = 3
    batch_size = 2
    h, w = 64, 64

    mgdip = MultiLevelGDIP(num_gdip_blocks=num_blocks, latent_dim=latent_dim)
    x = torch.rand(batch_size, 3, h, w)
    latent_features = [torch.rand(batch_size, latent_dim) for _ in range(num_blocks)]

    out, gates, intermediates = mgdip(x, latent_features)

    assert out.shape == (batch_size, 3, h, w), f"Output shape: {out.shape}"
    assert len(gates) == num_blocks, f"Expected {num_blocks} gates, got {len(gates)}"
    for i, g in enumerate(gates):
        assert g.shape == (batch_size, 7), f"Gate {i} shape: {g.shape}"
    assert intermediates == [], "Intermediates should be empty when return_intermediates=False"

def test_mgdip_return_intermediates():
    """MGDIP returns intermediate enhanced images when flag is True."""
    latent_dim = 128
    num_blocks = 3
    batch_size = 2
    h, w = 64, 64

    mgdip = MultiLevelGDIP(num_gdip_blocks=num_blocks, latent_dim=latent_dim)
    x = torch.rand(batch_size, 3, h, w)
    latent_features = [torch.rand(batch_size, latent_dim) for _ in range(num_blocks)]

    out, gates, intermediates = mgdip(x, latent_features, return_intermediates=True)

    assert len(intermediates) == num_blocks, f"Expected {num_blocks} intermediates, got {len(intermediates)}"
    for i, img in enumerate(intermediates):
        assert img.shape == (batch_size, 3, h, w), f"Intermediate {i} shape: {img.shape}"
    # Last intermediate should be the same as the final output
    assert torch.equal(out, intermediates[-1]), "Last intermediate should equal final output"

def test_mgdip_sequential_refinement():
    """Each GDIP block refines the output of the previous one (not the original input)."""
    latent_dim = 128
    num_blocks = 3

    mgdip = MultiLevelGDIP(num_gdip_blocks=num_blocks, latent_dim=latent_dim)
    x = torch.rand(1, 3, 64, 64)
    latent_features = [torch.rand(1, latent_dim) for _ in range(num_blocks)]

    _, _, intermediates = mgdip(x, latent_features, return_intermediates=True)

    # Intermediates should differ from each other (each block modifies the image)
    for i in range(1, num_blocks):
        assert not torch.equal(intermediates[i], intermediates[i-1]), \
            f"Block {i} output identical to block {i-1} — no refinement happening"

def test_mgdip_feature_count_mismatch():
    """MGDIP raises AssertionError when latent_features count != num_gdip_blocks."""
    latent_dim = 128
    mgdip = MultiLevelGDIP(num_gdip_blocks=3, latent_dim=latent_dim)
    x = torch.rand(1, 3, 64, 64)
    latent_features = [torch.rand(1, latent_dim) for _ in range(2)]  # 2 != 3

    with pytest.raises(AssertionError):
        mgdip(x, latent_features)

def test_mgdip_output_finite():
    """MGDIP output contains no NaN/Inf after chaining multiple blocks."""
    latent_dim = 128
    num_blocks = 4
    mgdip = MultiLevelGDIP(num_gdip_blocks=num_blocks, latent_dim=latent_dim)
    x = torch.rand(2, 3, 64, 64)
    latent_features = [torch.rand(2, latent_dim) for _ in range(num_blocks)]

    out, gates, _ = mgdip(x, latent_features)

    assert torch.isfinite(out).all(), "MGDIP output contains NaN/Inf"
    for i, g in enumerate(gates):
        assert torch.isfinite(g).all(), f"Gate {i} contains NaN/Inf"

def test_mgdip_gradient_flow():
    """Gradients flow through all GDIP blocks in MGDIP."""
    latent_dim = 128
    num_blocks = 3
    mgdip = MultiLevelGDIP(num_gdip_blocks=num_blocks, latent_dim=latent_dim)

    x = torch.rand(1, 3, 64, 64, requires_grad=True)
    latent_features = [torch.rand(1, latent_dim, requires_grad=True) for _ in range(num_blocks)]

    out, _, _ = mgdip(x, latent_features)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None, "No gradient on image input"
    for i, lf in enumerate(latent_features):
        assert lf.grad is not None, f"No gradient on latent_features[{i}]"

def test_mgdip_all_blocks_have_params():
    """Each GDIP block in MGDIP has independent learnable parameters."""
    latent_dim = 128
    num_blocks = 3
    mgdip = MultiLevelGDIP(num_gdip_blocks=num_blocks, latent_dim=latent_dim)

    assert len(mgdip.gdip_blocks) == num_blocks
    for i, block in enumerate(mgdip.gdip_blocks):
        params = list(block.parameters())
        assert len(params) > 0, f"Block {i} has no parameters"

def test_mgdip_encoder_end_to_end():
    """Full pipeline: encoder with tap_layers → MGDIP."""
    latent_dim = 256
    tap_layers = [4, 9, 16]
    encoder_cfg = {
        "type": "vgg16",
        "latent_dim": latent_dim,
        "pretrained": False,
        "tap_layers": tap_layers
    }
    encoder = build_vision_encoder(encoder_cfg)
    mgdip = MultiLevelGDIP(num_gdip_blocks=len(tap_layers), latent_dim=latent_dim)

    x = torch.rand(1, 3, 224, 224, requires_grad=True)
    latent, features = encoder(x)

    # Each tapped feature needs to be projected to latent_dim for MGDIP
    # For now, use the latent repeated (features have spatial dims, latent is [B, latent_dim])
    latent_features = [latent for _ in range(len(tap_layers))]

    out, gates, _ = mgdip(x, latent_features)

    assert out.shape == x.shape
    assert len(gates) == len(tap_layers)

    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "No gradient through encoder+MGDIP pipeline"

# ─── GDIPYolo with Real Data ─────────────────────────────────────────

def test_gdipyolo_with_real_data():
    """End-to-end GDIPYolo with actual BDD100K samples."""
    from panoptic_perception.dataset.bdd100k_dataset import BDD100KDataset, BDDPreprocessor

    # Dataset
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

    # GDIPYolo with GDIP mode
    gdip_kwargs = {
        "mode": "gdip",
        "encoder": {
            "type": "resnet18",
            "latent_dim": 256,
            "pretrained": False
        }
    }
    cfg_path = "panoptic_perception/configs/models/yolov8-detection-anchorFree.cfg"
    model = GDIPYolo(
        model_type="yolov8p",
        yolo_cfg=cfg_path,
        gdip_kwargs=gdip_kwargs
    )
    model.train()

    batch = next(iter(dataloader))
    images = batch["images"]
    targets = {
        "detections": batch["detections"],
        "drivable_area_seg": batch.get("drivable_area_seg"),
        "lane_seg": batch.get("segmentation_masks")
    }

    print(f"  Images: {images.shape}")
    print(f"  Detections: {batch['detections'].shape}")

    # Forward with targets (training mode — computes loss)
    outputs = model(images, targets)

    print(f"  Output keys: {list(outputs.keys())}")

    # Check outputs structure matches PanopticModelOutputs
    assert outputs.detection_loss is not None, "Detection loss is None"
    assert torch.isfinite(outputs.detection_loss), f"Detection loss is not finite: {outputs.detection_loss}"
    print(f"  Detection loss: {outputs.detection_loss.item():.4f}")

    # Backward pass — gradients flow through GDIP + YOLO
    total_loss = outputs.detection_loss
    total_loss.backward()

    # Verify gradients on GDIP components
    encoder_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                          for p in model.vision_encoder.parameters())
    gdip_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                        for p in model.gdip_module.parameters())
    yolo_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                        for p in model.model.parameters())

    assert encoder_has_grad, "No gradients in vision encoder"
    assert gdip_has_grad, "No gradients in GDIP module"
    assert yolo_has_grad, "No gradients in YOLO model"

    print(f"  Encoder grads: OK")
    print(f"  GDIP grads: OK")
    print(f"  YOLO grads: OK")

    # NaN check
    for name, param in model.named_parameters():
        assert not torch.isnan(param).any(), f"NaN in {name}"

    print(f"  NaN check: OK")

def test_mgdipyolo_with_real_data():
    """End-to-end MGDIPYolo (multi-level) with actual BDD100K samples."""
    from panoptic_perception.dataset.bdd100k_dataset import BDD100KDataset, BDDPreprocessor

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

    # MGDIP mode with VGG16 tap layers
    gdip_kwargs = {
        "mode": "mgdip",
        "encoder": {
            "type": "vgg16",
            "latent_dim": 256,
            "pretrained": False,
            "tap_layers": [4, 9, 16]
        },
        "visualize_intermediate": False
    }
    cfg_path = "panoptic_perception/configs/models/yolov8-detection-anchorFree.cfg"
    model = GDIPYolo(
        model_type="yolov8p",
        yolo_cfg=cfg_path,
        gdip_kwargs=gdip_kwargs
    )
    model.train()

    batch = next(iter(dataloader))
    images = batch["images"]
    targets = {
        "detections": batch["detections"],
        "drivable_area_seg": batch.get("drivable_area_seg"),
        "lane_seg": batch.get("segmentation_masks")
    }

    print(f"  Images: {images.shape}")
    print(f"  Detections: {batch['detections'].shape}")
    print(f"  MGDIP blocks: {len(model.gdip_module.gdip_blocks)}")

    # Forward with targets
    outputs = model(images, targets)

    assert outputs.detection_loss is not None, "Detection loss is None"
    assert torch.isfinite(outputs.detection_loss), f"Detection loss is not finite: {outputs.detection_loss}"
    print(f"  Detection loss: {outputs.detection_loss.item():.4f}")

    # Backward pass
    outputs.detection_loss.backward()

    # Verify gradients on all components
    encoder_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                          for p in model.vision_encoder.parameters())
    mgdip_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                         for p in model.gdip_module.parameters())
    yolo_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                        for p in model.model.parameters())

    assert encoder_has_grad, "No gradients in vision encoder"
    assert mgdip_has_grad, "No gradients in MGDIP module"
    assert yolo_has_grad, "No gradients in YOLO model"

    # Verify each GDIP block received gradients
    for i, block in enumerate(model.gdip_module.gdip_blocks):
        block_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                             for p in block.parameters())
        assert block_has_grad, f"No gradients in MGDIP block {i}"

    print(f"  Encoder grads: OK")
    print(f"  MGDIP grads (all {len(model.gdip_module.gdip_blocks)} blocks): OK")
    print(f"  YOLO grads: OK")

    # NaN check
    for name, param in model.named_parameters():
        assert not torch.isnan(param).any(), f"NaN in {name}"

    print(f"  NaN check: OK")

def test_gdipyolo_inference_with_real_data():
    """GDIPYolo inference (no targets) with actual BDD100K samples."""
    from panoptic_perception.dataset.bdd100k_dataset import BDD100KDataset, BDDPreprocessor

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

    gdip_kwargs = {
        "mode": "gdip",
        "encoder": {
            "type": "resnet18",
            "latent_dim": 256,
            "pretrained": False
        }
    }
    # cfg_path = "panoptic_perception/configs/models/yolov8-detection-anchorFree.cfg"
    cfg_path = "panoptic_perception/configs/models/yolo-768-1280-detection.cfg"    
    model = GDIPYolo(
        # model_type="yolov8p",
        model_type="yolop",
        yolo_cfg=cfg_path,
        gdip_kwargs=gdip_kwargs
    )
    model.eval()

    batch = next(iter(dataloader))

    with torch.no_grad():
        outputs = model(batch["images"])

    # In eval mode without targets: should have predictions, no losses
    assert outputs.detection_loss is None, "Loss should be None in eval without targets"
    assert outputs.detection_predictions is not None, "Detection predictions should not be None"

    print(f"  Detection predictions type: {type(outputs.detection_predictions)}")
    if isinstance(outputs.detection_predictions, torch.Tensor):
        print(f"  Detection predictions shape: {outputs.detection_predictions.shape}")
    else:
        print(f"  Detection predictions: {len(outputs.detection_predictions)} layers")

# ─── Run all tests ───────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("GDIP Unit Tests")
    print("=" * 60)

    # Vision encoder tests
    test_build_vision_encoder_vgg16()
    print("PASSED: build_vision_encoder vgg16")

    test_build_vision_encoder_resnet18()
    print("PASSED: build_vision_encoder resnet18")

    test_build_vision_encoder_invalid()
    print("PASSED: build_vision_encoder invalid type raises ValueError")

    test_vision_encoder_with_tap_layers()
    print("PASSED: vision_encoder with tap_layers")

    test_vision_encoder_no_tap_layers_returns_tensor()
    print("PASSED: vision_encoder without tap_layers returns tensor")

    test_vision_encoder_gradient_flow()
    print("PASSED: vision_encoder gradient flow")

    # GDIP tests
    test_gdip_forward_shape()
    print("PASSED: gdip forward shape")

    test_gdip_gate_range()
    print("PASSED: gdip gate range [0.01, 1.0]")

    test_gdip_output_finite()
    print("PASSED: gdip output finite")

    test_gdip_output_normalized()
    print("PASSED: gdip output normalized [0, 1]")

    test_gdip_gradient_flow()
    print("PASSED: gdip gradient flow")

    test_gdip_different_resolutions()
    print("PASSED: gdip different resolutions")

    test_gdip_batch_independence()
    print("PASSED: gdip batch independence (per-image normalization)")

    test_encoder_gdip_end_to_end()
    print("PASSED: encoder + gdip end-to-end")

    # MultiLevelGDIP tests
    print("=" * 60)
    print("MultiLevelGDIP Tests")
    print("=" * 60)

    test_mgdip_forward_shape()
    print("PASSED: mgdip forward shape")

    test_mgdip_return_intermediates()
    print("PASSED: mgdip return intermediates")

    test_mgdip_sequential_refinement()
    print("PASSED: mgdip sequential refinement")

    test_mgdip_feature_count_mismatch()
    print("PASSED: mgdip feature count mismatch raises AssertionError")

    test_mgdip_output_finite()
    print("PASSED: mgdip output finite")

    test_mgdip_gradient_flow()
    print("PASSED: mgdip gradient flow through all blocks")

    test_mgdip_all_blocks_have_params()
    print("PASSED: mgdip all blocks have independent params")

    test_mgdip_encoder_end_to_end()
    print("PASSED: encoder + mgdip end-to-end")

    # GDIPYolo with real data
    print("=" * 60)
    print("GDIPYolo Real Data Tests")
    print("=" * 60)

    test_gdipyolo_with_real_data()
    print("PASSED: gdipyolo (gdip) training forward+backward with real data")

    test_mgdipyolo_with_real_data()
    print("PASSED: gdipyolo (mgdip) training forward+backward with real data")

    test_gdipyolo_inference_with_real_data()
    print("PASSED: gdipyolo inference with real data")

    print("=" * 60)
    print("ALL TESTS PASSED")
