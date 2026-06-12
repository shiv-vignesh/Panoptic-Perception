import io
import cProfile
import pstats
from pathlib import Path
import pytest
import torch
import cv2

from torch.utils.data.dataloader import DataLoader

from panoptic_perception.dataset.bdd100k_dataset import (
    BDD100KDataset, FoggyBDD100KDataset, BDDPreprocessor
)
from panoptic_perception.dataset.adverse_weather.depth_estimators import (
    HeuristicDepthEstimator,
    DepthAnythingEstimator,
    TorchCompiledDepthEstimator,
    ONNXDepthEstimator,
    TensorRTDepthEstimator,
    RadialDistance
)
from panoptic_perception.dataset.types import DatasetMode


PROJECT_ROOT = Path(__file__).parent.parent
BDD_ROOT = PROJECT_ROOT / "BDD100k"

DEV_IMAGES_DIR = str(BDD_ROOT / "100k" / "dev_set")
DETECTION_DIR = str(BDD_ROOT / "bdd100k_labels" / "100k")
DRIVABLE_DIR = str(BDD_ROOT / "bdd100k_drivable_maps" / "labels")

VISUALIZE_DIR = 'visualizations/foggy_variations_test-noAug-3'

# Set these on the vast.ai instance before running
ONNX_MODEL_PATH = "depth_anything_small_fp16.onnx"
TRT_ENGINE_PATH = "/workspace/depth_anything_small.engine"

PREPROCESSOR_KWARGS = {
    "image_resize": (768, 1280),
    "original_image_size": (720, 1280),
    "perform_augmentation": True,
    "augment_params": {
        "degrees": 10,
        "translate": 0.1,
        "scale": 0.25,
        "shear": 5,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "salt_prob": 0.005,
        "pepper_prob": 0.005,
        "flip_prob": 0.5,
        "img_size": [768, 1280]
    }
}

ADVERSE_PARAMS = {
    "fog_betas": [0.008, 0.015, 0.025, 0.040],
    "darkness_gammas": [1.3, 1.5, 2.0],
    "enable_fog_only": True,
    "enable_darkness_only": True,
    "enable_compound": True,
    "max_depth_meters": 70.0
}

BATCH_SIZE = 4
SHUFFLE = False
NUM_WORKERS = 0


# ---------------------------------------------------------------------------
# Estimator factories
# ---------------------------------------------------------------------------

def _make_heuristic():
    return HeuristicDepthEstimator(
        vertical_weight=0.7, intensity_weight=0.2,
        edge_weight=0.1, edge_epsilon=1e-8, uint8_max=255.0,
    )

def _make_depth_anything():
    return DepthAnythingEstimator(
        model_name="LiheYoung/depth-anything-small-hf",
        device="cuda", normalization_epsilon=1e-8,
    )

def _make_torch_compile():
    return TorchCompiledDepthEstimator(
        model_name="LiheYoung/depth-anything-small-hf",
        device="cuda", normalization_epsilon=1e-8,
        compile_mode="reduce-overhead",
    )

def _make_onnx():
    return ONNXDepthEstimator(
        onnx_path=ONNX_MODEL_PATH, device="cuda",
        input_size=518, normalization_epsilon=1e-8,
    )

def _make_tensorrt():
    return TensorRTDepthEstimator(
        engine_path=TRT_ENGINE_PATH, input_size=518,
        normalization_epsilon=1e-8,
    )

def _make_radial_distance():
    return RadialDistance(
        radial_depth_decay_rate=-0.04
    )

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def base_dataset_kwargs():
    return {
        "images_dir": DEV_IMAGES_DIR,
        "detection_annotations_dir": DETECTION_DIR,
        "segmentation_annotations_dir": "",
        "drivable_annotations_dir": DRIVABLE_DIR,
        "preprocessor_kwargs": PREPROCESSOR_KWARGS,
    }

@pytest.fixture(scope="session")
def dataloader_kwargs():
    return {
        "batch_size": BATCH_SIZE,
        "shuffle": SHUFFLE,
        "num_workers": NUM_WORKERS,
        "collate_fn": BDDPreprocessor.collate_fn,
    }

@pytest.fixture(scope="session")
def clean_dataset(base_dataset_kwargs):
    return BDD100KDataset(
        dataset_kwargs=base_dataset_kwargs,
        dataset_type="train",
        perform_augmentation=base_dataset_kwargs.get("perform_augmentation", True),
        mode=DatasetMode.TRAIN,
    )

def _make_foggy_dataset(base_dataset_kwargs, depth_estimator):
    kwargs = {**base_dataset_kwargs, "adverse_params": ADVERSE_PARAMS}
    return FoggyBDD100KDataset(
        dataset_kwargs=kwargs,
        dataset_type="train",
        perform_augmentation=base_dataset_kwargs.get("perform_augmentation", True),
        mode=DatasetMode.TRAIN,
        apply_fog_prob=1.0,
        depth_estimator=depth_estimator,
    )

@pytest.fixture(scope="session")
def foggy_heuristic(base_dataset_kwargs):
    return _make_foggy_dataset(base_dataset_kwargs, _make_heuristic())

@pytest.fixture(scope="session")
def foggy_depth_anything(base_dataset_kwargs):
    return _make_foggy_dataset(base_dataset_kwargs, _make_depth_anything())

@pytest.fixture(scope="session")
def foggy_torch_compile(base_dataset_kwargs):
    return _make_foggy_dataset(base_dataset_kwargs, _make_torch_compile())

@pytest.fixture(scope="session")
def foggy_onnx(base_dataset_kwargs):
    return _make_foggy_dataset(base_dataset_kwargs, _make_onnx())

@pytest.fixture(scope="session")
def foggy_tensorrt(base_dataset_kwargs):
    return _make_foggy_dataset(base_dataset_kwargs, _make_tensorrt())

@pytest.fixture(scope="session")
def foggy_radial_distance(base_dataset_kwargs):
    return _make_foggy_dataset(base_dataset_kwargs, _make_radial_distance())

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_loader(dataset, dl_kwargs):
    return DataLoader(
        dataset,
        batch_size=dl_kwargs["batch_size"],
        shuffle=dl_kwargs["shuffle"],
        num_workers=dl_kwargs["num_workers"],
        collate_fn=dl_kwargs["collate_fn"],
    )

def profile_loader(loader, name, n_batches=50, warmup: bool = True):
    n_batches = min(n_batches, len(loader))

    profiler = cProfile.Profile()

    if not warmup:
        profiler.enable()
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
        profiler.disable()
    else:
        it = iter(loader)
        warmup_count = min(5, len(loader))
        for i in range(warmup_count):
            next(it)

        profiler.enable()
        for i in range(n_batches - warmup_count):
            try:
                next(it)
            except StopIteration:
                break
        profiler.disable()

    s = io.StringIO()
    stats = pstats.Stats(profiler, stream=s)
    stats.sort_stats("cumtime").print_stats(30)

    print(f"\n--- PROFILE: {name} ---")
    print(s.getvalue())

def test_visualize_foggy_samples(loader, save_dir, n_batches=5):
    
    import os
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for i, data_items in enumerate(loader):
        if isinstance(data_items, dict):                
            assert isinstance(data_items["images"], torch.Tensor), f'Expected data type: {torch.Tensor} got {type(data_items["images"])}'
            foggy_images = (data_items["images"].clamp(0, 1) * 255).to(torch.uint8)
            image_paths = data_items.get("image_paths")

            for idx, image in enumerate(foggy_images):
                image = image.detach().cpu().permute(1, 2, 0).numpy()

                if image_paths:
                    filename = Path(image_paths[idx]).name
                    save_path = f'{save_dir}/{filename}'
                else:
                    save_path = f'{save_dir}/image_{i}_{idx}.png'
                cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            
            if i > n_batches:
                break
            
        else:
            raise TypeError(f'Expected data type: {dict} got {type(data_items)}')
        
        
        

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_profile_clean(clean_dataset, dataloader_kwargs):
    loader = _make_loader(clean_dataset, dataloader_kwargs)
    profile_loader(loader, "CLEAN")

    test_visualize_foggy_samples(loader, f'{VISUALIZE_DIR}/CLEAN')

def test_profile_heuristic(foggy_heuristic, dataloader_kwargs):
    loader = _make_loader(foggy_heuristic, dataloader_kwargs)
    profile_loader(loader, "FOGGY (heuristic)")
    
    test_visualize_foggy_samples(loader, f'{VISUALIZE_DIR}/heuristic')

def test_profile_depth_anything(foggy_depth_anything, dataloader_kwargs):
    loader = _make_loader(foggy_depth_anything, dataloader_kwargs)
    profile_loader(loader, "FOGGY (depth_anything)")
    
    test_visualize_foggy_samples(loader, f'{VISUALIZE_DIR}/DepthAnything')

def test_profile_torch_compile(foggy_torch_compile, dataloader_kwargs):
    loader = _make_loader(foggy_torch_compile, dataloader_kwargs)
    profile_loader(loader, "FOGGY (torch_compile)")

def test_profile_onnx(foggy_onnx, dataloader_kwargs):
    loader = _make_loader(foggy_onnx, dataloader_kwargs)
    profile_loader(loader, "FOGGY (onnx)")
    
    test_visualize_foggy_samples(loader, f'{VISUALIZE_DIR}/DepthAnything-ONNX')

def test_profile_tensorrt(foggy_tensorrt, dataloader_kwargs):
    loader = _make_loader(foggy_tensorrt, dataloader_kwargs)
    profile_loader(loader, "FOGGY (tensorrt)")

def test_profile_radial_distance(foggy_radial_distance, dataloader_kwargs):
    loader = _make_loader(foggy_radial_distance, dataloader_kwargs)
    profile_loader(loader, "FOGGY (radial distance)")
    
    test_visualize_foggy_samples(loader, f'{VISUALIZE_DIR}/RadialDistance')
    
