import pytest
import torch

from panoptic_perception.models.model_factory import ModelFactory
from panoptic_perception.models.models import YOLOP, YOLOv8P, GDIPYolo, DENetYolo

YOLOP_CFG = "panoptic_perception/configs/models/yolov8-768-1280-detection.cfg"
YOLOV8P_CFG = "panoptic_perception/configs/models/yolov8-768-1280-detection.cfg"

BS, NC, IMAGE_H, IMAGE_W = 2, 3, 768, 1280

DENET_KWARGS = {
    "num_high": 3,
    "gaussian_kernel_size": 5,
    "num_channels": 3,
    "channel_blocks": 64,
    "channel_mask": 16,
    "high_channels": 32,
    "high_kernel_size": 3,
    "visualize_intermediate": True,
}

GDIP_KWARGS = {
    "mode": "gdip",
    "encoder": {
        "type": "resnet18",
        "latent_dim": 256,
        "pretrained": True,
    },
    "visualize_intermediate": True,
}

MGDIP_KWARGS = {
    "mode": "mgdip",
    "encoder": {
        "type": "resnet18",
        "latent_dim": 256,
        "pretrained": True,
        "tap_layers": [5, 6, 7],
    },
    "visualize_intermediate": True,
}


# ---------------------------------------------------------------------------
# Registry state tests
# ---------------------------------------------------------------------------

class TestRegistryState:

    def test_task_models_registered(self):
        assert "yolop" in ModelFactory._task_models
        assert "yolov8p" in ModelFactory._task_models

    def test_enhancements_registered(self):
        assert "gdip-yolo" in ModelFactory._enhancements
        assert "denet-yolo" in ModelFactory._enhancements

    def test_registered_classes(self):
        assert ModelFactory._task_models["yolop"] is YOLOP
        assert ModelFactory._task_models["yolov8p"] is YOLOv8P
        assert ModelFactory._enhancements["gdip-yolo"] is GDIPYolo
        assert ModelFactory._enhancements["denet-yolo"] is DENetYolo


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------

class TestFactoryErrors:

    def test_unknown_model_type_raises(self):
        with pytest.raises(KeyError, match="Unknown Model Class"):
            ModelFactory.from_config({"model_type": "invalid", "cfg_path": YOLOP_CFG})

    def test_unknown_enhancement_raises(self):
        with pytest.raises(KeyError, match="Unknown enhancement"):
            ModelFactory.from_config({
                "model_type": "yolov8p",
                "cfg_path": YOLOV8P_CFG,
                "enhancement": "nonexistent",
            })

    def test_missing_cfg_path_raises(self):
        with pytest.raises(AssertionError):
            ModelFactory.from_config({
                "model_type": "yolop",
                "cfg_path": "/does/not/exist.cfg",
            })

    def test_denet_missing_kwargs_raises(self):
        with pytest.raises(KeyError, match="denet_kwargs"):
            ModelFactory.from_config({
                "model_type": "yolov8p",
                "cfg_path": YOLOV8P_CFG,
                "enhancement": "denet-yolo",
            })

    def test_gdip_missing_kwargs_raises(self):
        with pytest.raises(KeyError, match="gdip_kwargs"):
            ModelFactory.from_config({
                "model_type": "yolov8p",
                "cfg_path": YOLOV8P_CFG,
                "enhancement": "gdip-yolo",
            })


# ---------------------------------------------------------------------------
# Task model creation tests
# ---------------------------------------------------------------------------

class TestTaskModelCreation:

    def test_create_yolop(self):
        model = ModelFactory.from_config({
            "model_type": "yolop",
            "cfg_path": YOLOP_CFG,
        })
        assert isinstance(model, YOLOP)

    def test_create_yolov8p(self):
        model = ModelFactory.from_config({
            "model_type": "yolov8p",
            "cfg_path": YOLOV8P_CFG,
        })
        assert isinstance(model, YOLOv8P)

    def test_create_with_loss_weights(self):
        weights = {"detection": 1.0, "drivable_segmentation": 0.5, "lane_segmentation": 0.0}
        model = ModelFactory.from_config(
            {"model_type": "yolov8p", "cfg_path": YOLOV8P_CFG},
            loss_weights=weights,
        )
        assert model.loss_weights == weights

    def test_no_enhancement_returns_base(self):
        model = ModelFactory.from_config({
            "model_type": "yolov8p",
            "cfg_path": YOLOV8P_CFG,
            "enhancement": None,
        })
        assert isinstance(model, YOLOv8P)


# ---------------------------------------------------------------------------
# Enhancement model creation tests
# ---------------------------------------------------------------------------

class TestEnhancementCreation:

    def test_create_gdip_yolo(self):
        model = ModelFactory.from_config({
            "model_type": "yolov8p",
            "cfg_path": YOLOV8P_CFG,
            "enhancement": "gdip-yolo",
            "gdip_kwargs": GDIP_KWARGS,
        })
        assert isinstance(model, GDIPYolo)
        assert isinstance(model.task_network, YOLOv8P)

    def test_create_mgdip_yolo(self):
        model = ModelFactory.from_config({
            "model_type": "yolov8p",
            "cfg_path": YOLOV8P_CFG,
            "enhancement": "gdip-yolo",
            "gdip_kwargs": MGDIP_KWARGS,
        })
        assert isinstance(model, GDIPYolo)
        assert model.gdip_mode == "mgdip"

    def test_create_denet_yolo(self):
        model = ModelFactory.from_config({
            "model_type": "yolov8p",
            "cfg_path": YOLOV8P_CFG,
            "enhancement": "denet-yolo",
            "denet_kwargs": DENET_KWARGS,
        })
        assert isinstance(model, DENetYolo)
        assert isinstance(model.task_network, YOLOv8P)

    def test_denet_wraps_yolop(self):
        model = ModelFactory.from_config({
            "model_type": "yolop",
            "cfg_path": YOLOP_CFG,
            "enhancement": "denet-yolo",
            "denet_kwargs": DENET_KWARGS,
        })
        assert isinstance(model, DENetYolo)
        assert isinstance(model.task_network, YOLOP)

    def test_gdip_wraps_yolop(self):
        model = ModelFactory.from_config({
            "model_type": "yolop",
            "cfg_path": YOLOP_CFG,
            "enhancement": "gdip-yolo",
            "gdip_kwargs": GDIP_KWARGS,
        })
        assert isinstance(model, GDIPYolo)
        assert isinstance(model.task_network, YOLOP)


# ---------------------------------------------------------------------------
# Forward pass smoke tests
# ---------------------------------------------------------------------------

class TestForwardPass:

    @pytest.fixture
    def rand_x(self):
        return torch.rand(BS, NC, IMAGE_H, IMAGE_W)

    def test_yolov8p_forward(self, rand_x):
        model = ModelFactory.from_config({
            "model_type": "yolov8p",
            "cfg_path": YOLOV8P_CFG,
        })
        model.eval()
        with torch.no_grad():
            outputs = model(rand_x)
        assert outputs is not None

    def test_denet_yolo_forward(self, rand_x):
        model = ModelFactory.from_config({
            "model_type": "yolov8p",
            "cfg_path": YOLOV8P_CFG,
            "enhancement": "denet-yolo",
            "denet_kwargs": DENET_KWARGS,
        })
        model.eval()
        with torch.no_grad():
            outputs = model(rand_x, store_enhanced_image=True, compute_ssim=False)
        assert outputs is not None
        assert model.enhanced_image is not None
        assert model.enhanced_image.shape == rand_x.shape

    def test_gdip_yolo_forward(self, rand_x):
        model = ModelFactory.from_config({
            "model_type": "yolov8p",
            "cfg_path": YOLOV8P_CFG,
            "enhancement": "gdip-yolo",
            "gdip_kwargs": GDIP_KWARGS,
        })
        model.eval()
        with torch.no_grad():
            outputs = model(rand_x, store_enhanced_image=True, compute_ssim=False)
        assert outputs is not None
        assert model.enhanced_image is not None
        assert model.enhanced_image.shape == rand_x.shape
