import pytest
import torch

from panoptic_perception.scripts.train.train_v2 import create_model
from panoptic_perception.models.teacher_model import TeacherFusion
from panoptic_perception.models.models import YOLOP
from panoptic_perception.losses.loss_factory import LossFactory
import panoptic_perception.losses  # registers all loss types


@pytest.fixture
def image_backbone_cfg():
    return "panoptic_perception/configs/models/teacher_model/image_backbone.cfg"


@pytest.fixture
def depth_backbone_cfg():
    return "panoptic_perception/configs/models/teacher_model/depth_backbone.cfg"


def test_depth_backbone_constructs(depth_backbone_cfg):
    model = YOLOP(cfg=depth_backbone_cfg, loss_function=None)

    assert model.detection_head_idx == -1
    assert model.segmentation_head_idx == -1
    assert model.lane_segmentation_head_idx == -1

    x = torch.rand(1, 3, 640, 640, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        taps, _ = model.forward_backbone(x, intercept_layers=[17, 20, 23])

    assert taps[17].shape == (1, 128, 80, 80)
    assert taps[20].shape == (1, 256, 40, 40)
    assert taps[23].shape == (1, 512, 20, 20)


def test_image_backbone_constructs(image_backbone_cfg):
    model = YOLOP(cfg=image_backbone_cfg, loss_function=None)
    assert model.detection_head_idx == 24


@pytest.fixture
def batch_size():
    return 1


@pytest.fixture
def image_size():
    return (640, 640)


@pytest.fixture
def num_classes():
    return 6


@pytest.fixture
def dummy_image(batch_size, image_size):
    h, w = image_size
    return torch.rand(batch_size, 3, h, w, dtype=torch.float32)


@pytest.fixture
def dummy_depth(batch_size, image_size):
    h, w = image_size
    return torch.rand(batch_size, 3, h, w, dtype=torch.float32)


@pytest.fixture
def dummy_detection_targets(batch_size, num_classes):
    # Row format: [batch_idx, cls, cx, cy, w, h] — post-collate detection target shape.
    rows = []
    for b in range(batch_size):
        for _ in range(3):
            cls = float(torch.randint(0, num_classes, (1,)).item())
            cx, cy = torch.rand(2).tolist()
            w, h = (torch.rand(2) * 0.5 + 0.05).tolist()
            rows.append([float(b), cls, cx, cy, w, h])
    return torch.tensor(rows, dtype=torch.float32)


@pytest.fixture
def dummy_targets(dummy_detection_targets):
    return {
        "detections": dummy_detection_targets,
        "drivable_area_seg": None,
        "lane_seg": None,
        "lanes_detections": None,
        "lane_seg_masks": None,
        "clean_images": None,
    }


@pytest.fixture
def image_model_kwargs(image_backbone_cfg):
    return {
        "model_type": "yolop",
        "cfg_path": image_backbone_cfg,
        "device": "cpu",
    }


@pytest.fixture
def depth_model_kwargs(depth_backbone_cfg):
    return {
        "model_type": "yolop",
        "cfg_path": depth_backbone_cfg,
        "device": "cpu",
    }


@pytest.fixture
def loss_function_kwargs():
    return {
        "detection": {"_type": "detection-loss-ATSS", "kwargs": {}},
        "drivable_segmentation": {"_type": "segmentation-loss", "kwargs": {}},
        "lane_detection": {"_type": "lane-detection-loss", "kwargs": {}},
        "depth_reconstruction": {
            "_type": "depth-reconstruction",
            "kwargs": {
                "loss_type": "smooth_l1",
                "aux_weights": {17: 0.3, 20: 0.4, 23: 0.5},
                "full_res_weight": 1.0,
            },
        },
        "loss_weights": {
            "detection": 1.0,
            "drivable_segmentation": 1.0,
            "lane_detection": 1.0,
            "lane_segmentation": 0.0,
            "depth_reconstruction": 0.5,
        },
    }


@pytest.fixture
def teacher_fusion_kwargs(image_model_kwargs, depth_model_kwargs):
    return {
        "image_model_kwargs": image_model_kwargs,
        "depth_model_kwargs": depth_model_kwargs,
        "fusion_kwargs": {
            "backbone_intercepts": {"detection": [(17, 128), (20, 256), (23, 512)]},
            "fusion_type": "attention",
            "weighted_fusion": False,
            # nb=1 while AttentionBasedFusion's depth-out path is non-symmetric.
            # See Fix-B TODO in teacher_model.py before bumping.
            "num_fusion_blocks": 1,
            "aux_depth_recon_cfg": "panoptic_perception/configs/models/teacher_model/depth_recon.cfg",
        },
    }


def test_teacher_fusion(teacher_fusion_kwargs, loss_function_kwargs,
                        dummy_image, dummy_depth, dummy_targets):
    image_model, _ = create_model(teacher_fusion_kwargs["image_model_kwargs"], loss_function_kwargs)
    depth_model, _ = create_model(teacher_fusion_kwargs["depth_model_kwargs"], loss_function_kwargs)

    assert image_model.detection_head_idx == 24
    assert depth_model.detection_head_idx == -1

    teacher_model = TeacherFusion(
        image_model=image_model,
        depth_model=depth_model,
        fusion_kwargs=teacher_fusion_kwargs["fusion_kwargs"],
    )

    teacher_model.eval()
    with torch.no_grad():
        outputs = teacher_model.forward(image=dummy_image, depth=dummy_depth, targets=dummy_targets)

    assert outputs is not None
    assert outputs.detection_logits is not None

    recon = outputs.depth_reconstruction
    assert recon is not None, "aux_depth_recon_cfg was set; expected depth_reconstruction in outputs"
    assert "detection" in recon.predictions
    preds = recon.predictions["detection"]

    h, w = dummy_image.shape[-2:]
    assert preds["full_res"].shape == (dummy_image.shape[0], 1, h, w)
    assert preds["tap_17"].shape == (dummy_image.shape[0], 1, h // 8, w // 8)
    assert preds["tap_20"].shape == (dummy_image.shape[0], 1, h // 16, w // 16)
    assert preds["tap_23"].shape == (dummy_image.shape[0], 1, h // 32, w // 32)

    assert recon.target.shape == (dummy_image.shape[0], 1, h, w)
    assert (preds["full_res"] >= 0).all() and (preds["full_res"] <= 1).all(), \
        "depth pred must be sigmoid-bounded to [0, 1]"

    assert outputs.depth_reconstruction_loss is not None, \
        "loss_function has depth_reconstruction registered; TeacherFusion should auto-compute"
    assert torch.isfinite(outputs.depth_reconstruction_loss)


def test_depth_reconstruction_loss_and_param_groups(
    teacher_fusion_kwargs, loss_function_kwargs, dummy_image, dummy_depth, dummy_targets,
):
    image_model, _ = create_model(teacher_fusion_kwargs["image_model_kwargs"], loss_function_kwargs)
    depth_model, _ = create_model(teacher_fusion_kwargs["depth_model_kwargs"], loss_function_kwargs)
    teacher_model = TeacherFusion(
        image_model=image_model, depth_model=depth_model,
        fusion_kwargs=teacher_fusion_kwargs["fusion_kwargs"],
    )

    groups = teacher_model.get_param_groups(optimizer_kwargs={})
    group_names = [g["name"] for g in groups]
    assert "fusion_blocks" in group_names
    assert "depth_decoders" in group_names, f"decoder params missing from optimizer; got {group_names}"
    decoder_group = next(g for g in groups if g["name"] == "depth_decoders")
    assert len(decoder_group["params"]) > 0, "depth_decoders group has no params"

    loss_cfg = {
        "_type": "depth-reconstruction",
        "kwargs": {"loss_type": "smooth_l1", "aux_weights": {17: 0.3, 20: 0.4, 23: 0.5}},
    }
    loss_fn = LossFactory.build(loss_cfg)

    teacher_model.train()
    outputs = teacher_model.forward(image=dummy_image, depth=dummy_depth, targets=dummy_targets)
    total, items = loss_fn(outputs.depth_reconstruction)

    assert torch.isfinite(total), f"loss is not finite: {total}"
    assert "depth_recon_total" in items
    for tap in (17, 20, 23):
        assert f"detection/tap_{tap}" in items
    assert "detection/full_res" in items

    # Backward should reach decoder params.
    decoder_param = decoder_group["params"][0]
    decoder_param.grad = None
    total.backward()
    assert decoder_param.grad is not None, "no gradient flowed back to depth_decoders"
