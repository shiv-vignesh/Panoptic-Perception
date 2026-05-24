"""
Plumbing tests for ATSS (panoptic_perception/utils/detection_utils.py).

These tests verify the call chain runs without crashing on realistic-shaped
inputs. They do NOT verify matching correctness (that needs known IoU
patterns + ground-truth assignments and belongs in a separate file once
_match() is implemented).
"""

import pytest
import torch

from panoptic_perception.dataset.bdd100k_dataset import BDD100KDataset, BDDPreprocessor
from panoptic_perception.utils.detection_utils import ATSS, Matcher
from panoptic_perception.models.models import YOLOP


# ----- fixtures ---------------------------------------------------------------

@pytest.fixture
def img_size():
    """(H, W) of the input image. 640x640 gives clean grid divisions for s=8/16/32."""
    return 640, 640

@pytest.fixture()
def dataloader_batch_size():
    return 8

@pytest.fixture
def dataset_kwargs(img_size):
    DATASET_KWARGS = {
        "images_dir": "data/100k/100k",
        "detection_annotations_dir": "data/bdd100k_labels/100k",
        "segmentation_annotations_dir": "data/bdd100k_seg_maps/labels",
        "drivable_annotations_dir": "data/bdd100k_drivable_maps/labels",
        "preprocessor_kwargs": {
            "image_resize": (img_size[0], img_size[0]),
            "original_image_size": (720, 1280)
        }
    }    

    return DATASET_KWARGS

@pytest.fixture
def dataloader(dataset_kwargs, dataloader_batch_size):
    
    # --- Dataset ---
    dataset = BDD100KDataset(dataset_kwargs, dataset_type='val')
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=dataloader_batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=BDDPreprocessor.collate_fn
    )
    print(f"Dataset: {len(dataset)} images, batch_size={dataloader_batch_size}")

    return dataloader

@pytest.fixture
def anchor_proposals(img_size):
    """3-level FPN anchors as List[Tensor[N_l, 4]] in pixel xyxy.

    Mirrors YOLOP-style strides 32/16/8 with 3 anchors per cell:
      level 0: 20x20 grid * 3 = 1200 anchors
      level 1: 40x40 grid * 3 = 4800 anchors
      level 2: 80x80 grid * 3 = 19200 anchors
    Anchor sizes are plausible defaults; values aren't algorithmically meaningful
    here — the goal is shape/dtype/coord-range realism for plumbing exercise.
    """
    H, W = img_size
    strides = [32, 16, 8]
    sizes_per_level = [
        [(128, 128), (180, 128), (256, 256)],
        [(64, 64),   (90, 64),   (128, 128)],
        [(32, 32),   (45, 32),   (64, 64)],
    ]

    levels = []
    for stride, anchor_sizes in zip(strides, sizes_per_level):
        gh, gw = H // stride, W // stride
        ys = (torch.arange(gh, dtype=torch.float32) + 0.5) * stride
        xs = (torch.arange(gw, dtype=torch.float32) + 0.5) * stride
        cy, cx = torch.meshgrid(ys, xs, indexing="ij")
        centers = torch.stack([cx.flatten(), cy.flatten()], dim=-1)  # (gh*gw, 2)

        boxes_per_anchor = []
        for aw, ah in anchor_sizes:
            x1 = centers[:, 0] - aw / 2
            y1 = centers[:, 1] - ah / 2
            x2 = centers[:, 0] + aw / 2
            y2 = centers[:, 1] + ah / 2
            boxes_per_anchor.append(torch.stack([x1, y1, x2, y2], dim=-1))

        # (gh*gw, A, 4) -> flatten anchors: (gh*gw*A, 4)
        per_cell = torch.stack(boxes_per_anchor, dim=1)
        levels.append(per_cell.reshape(-1, 4))

    return levels


@pytest.fixture
def targets():
    """3 GT boxes across a batch of 2. YOLO format: (batch_idx, class_id, cx, cy, w, h) normalized."""
    return torch.tensor(
        [
            [0, 0, 0.50, 0.50, 0.30, 0.30],
            [0, 1, 0.20, 0.30, 0.10, 0.15],
            [1, 0, 0.70, 0.40, 0.20, 0.25],
            [1, 3, 0.45, 0.35, 0.12, 0.18],
            [2, 2, 0.50, 0.40, 0.12, 0.18]
        ],
        dtype=torch.float32,
    )

@pytest.fixture
def empty_targets():
    return torch.tensor((0, 6), dtype=torch.float32)

@pytest.fixture
def image(img_size):
    H, W = img_size
    return torch.rand((4, 3, H, W))

@pytest.fixture
def matcher():
    return Matcher(low_threshold=0.2, high_threshold=0.7, allow_low_quality_matches=True)

@pytest.fixture
def atss():
    return ATSS()

@pytest.fixture
def model_cfg():
    return "panoptic_perception/configs/models/yolov8-768-1280-detection-drivable.cfg"

@pytest.fixture
def yolop_model(model_cfg):
    import os
    if not os.path.exists(model_cfg):
        raise FileNotFoundError

    return YOLOP(model_cfg, use_atss=True).eval()

def test_subsample_runs_end_to_end(atss, anchor_proposals, targets, img_size):
    """Plumbing test: subsample() executes without raising on realistic inputs.

    Does not assert on the return value — that contract isn't frozen yet.
    Once _match() and the return shape are defined, extend this with
    shape/dtype/range assertions.
    """

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    H, W = img_size
    atss.subsample(
        anchor_proposals=anchor_proposals,
        targets=targets.to(device),
        img_w=W,
        img_h=H,
        batch_size=3,
        device=device
    )

def test_atss_model_end_to_end(yolop_model, targets, image):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    yolop_model.to(device)

    model_outputs = yolop_model(
        image,
        {
            "detections": targets.to(device)
        }
    )

    print(model_outputs.detection_loss)

def test_atss_model_dataset(yolop_model, targets, dataloader):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    for data_items in dataloader:
        outputs = yolop_model(
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

        break

def test_compute_loss_empty_targets(yolop_model, image, empty_targets):
    """Zero-row targets: lbox=lcls=0, lobj>0 (pushes background everywhere), no NaN."""
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    yolop_model.to(device)

    yolop_model(
        image,
        {
            "detections": empty_targets.to(device)
        }
    )

# def test_subsample_infers_batch_size_when_none(atss, anchor_proposals, targets, img_size):
#     """batch_size=None should derive from max(targets[:, 0]) without crashing."""
#     H, W = img_size
#     atss.subsample(
#         anchor_proposals=anchor_proposals,
#         targets=targets,
#         img_w=W,
#         img_h=H,
#         batch_size=None,
#     )


# def test_subsample_rejects_wrong_target_shape(atss, anchor_proposals, img_size):
#     """Targets must be (N, 6); other shapes should fail the assert."""
#     H, W = img_size
#     bad_targets = torch.zeros((3, 5), dtype=torch.float32)  # missing one column
#     with pytest.raises(AssertionError):
#         atss.subsample(
#             anchor_proposals=anchor_proposals,
#             targets=bad_targets,
#             img_w=W,
#             img_h=H,
#             batch_size=2,
#         )


# def test_anchor_fixture_shapes(anchor_proposals):
#     """Sanity check on the fixture itself so failures upstream are obvious."""
#     expected_counts = [20 * 20 * 3, 40 * 40 * 3, 80 * 80 * 3]
#     assert len(anchor_proposals) == 3
#     for level, expected in zip(anchor_proposals, expected_counts):
#         assert level.shape == (expected, 4)
#         # xyxy invariant: x2 > x1, y2 > y1
#         assert torch.all(level[:, 2] > level[:, 0])
#         assert torch.all(level[:, 3] > level[:, 1])
