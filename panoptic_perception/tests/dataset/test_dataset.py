"""
Tests for BDD100K dataset with lane detection integration.

Unit tests use synthetic data; integration tests require real BDD100K data paths.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import torch
import json
import tempfile
import os
import random

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from panoptic_perception.dataset.bdd100k_dataset import BDDPreprocessor, BDD100KDataset, visualize_batch, LANE_VIS_COLORS
from panoptic_perception.dataset.enums import BDD100KClasses, BDD100KClassesReduced
from panoptic_perception.utils.lane_utils import BDD100KLaneCategories
from panoptic_perception.dataset.augmentations import (
    random_perspective, flip_horizontal, letterbox_with_masks,
    apply_augmentations, mixup_augmentation
)
from panoptic_perception.dataset.mosaic_augmentation import mosaic_augmentation
from panoptic_perception.utils.lane_utils import (
    polyline_to_lane_target, build_lane_targets, NUM_LANE_POINTS, MAX_LANES
)

# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _make_lane_polys():
    """Create synthetic lane polylines in pixel coordinates for a 720x1280 image."""
    # Left lane: roughly vertical, slight curve
    left_lane = np.array([
        [400, 700], [420, 600], [450, 500], [490, 400],
        [530, 300], [570, 200], [600, 100]
    ], dtype=np.float32)

    # Right lane
    right_lane = np.array([
        [800, 700], [780, 600], [760, 500], [740, 400],
        [720, 300], [700, 200], [680, 100]
    ], dtype=np.float32)

    return [
        {"points": left_lane, "category": "lane/single white"},
        {"points": right_lane, "category": "lane/single yellow"},
    ]


def _make_image(h=720, w=1280):
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


def _make_labels(n=3, h=720, w=1280):
    """Normalized xywh labels: [cls, cx, cy, w, h]."""
    labels = np.zeros((n, 5), dtype=np.float32)
    for i in range(n):
        labels[i] = [i % 3, 0.3 + i * 0.2, 0.4 + i * 0.1, 0.1, 0.15]
    return labels


def _make_bdd_json(tmp_dir, image_id="test_img", num_boxes=2, num_lanes=2):
    """Write a minimal BDD100K-format JSON and return its path."""
    objects = []
    for i in range(num_boxes):
        objects.append({
            "category": "car",
            "box2d": {"x1": 100 + i * 50, "y1": 200, "x2": 300 + i * 50, "y2": 400}
        })
    for i in range(num_lanes):
        cat = "lane/single white" if i % 2 == 0 else "lane/single yellow"
        objects.append({
            "category": cat,
            "poly2d": [
                [300 + i * 200, 700, "L"],
                [350 + i * 180, 500, "L"],
                [400 + i * 160, 300, "L"],
                [430 + i * 140, 100, "L"],
            ]
        })
    data = {
        "attributes": {"weather": "clear", "timeofday": "daytime"},
        "frames": [{"objects": objects}]
    }
    path = os.path.join(tmp_dir, f"{image_id}.json")
    with open(path, "w") as f:
        json.dump(data, f)
    return path


def _make_batch_item(h=640, w=640, n_det=2, n_lanes=2):
    """Create a single sample dict matching prepare_training_sample output."""
    lane_polys = _make_lane_polys()[:n_lanes] if n_lanes > 0 else []
    lane_targets, lane_categories = build_lane_targets(lane_polys, h, w)
    return {
        "image": torch.rand(3, h, w),
        "segmentation_mask": torch.randint(0, 10, (h, w)),
        "drivable_mask": torch.randint(0, 2, (h, w)),
        "detection_targets": torch.tensor(
            [[i % 3, 0.5, 0.5, 0.1, 0.1] for i in range(n_det)], dtype=torch.float32
        ) if n_det > 0 else torch.zeros((0, 5), dtype=torch.float32),
        "lane_targets": lane_targets,
        "lane_categories": lane_categories,
        "image_path": "/fake/path.jpg",
        "scene_attributes": {"weather": "clear", "timeofday": "daytime"},
    }


# ===================================================================
#  1. Enum Tests
# ===================================================================

class TestBDD100KLaneCategories:

    def test_values(self):
        assert BDD100KLaneCategories.SINGLE_WHITE.value == 0
        assert BDD100KLaneCategories.SINGLE_YELLOW.value == 1
        assert BDD100KLaneCategories.DOUBLE_YELLOW.value == 2
        assert BDD100KLaneCategories.ROAD_CURB.value == 3

    def test_from_label_bdd_format(self):
        assert BDD100KLaneCategories.from_label("lane/single white") == 0
        assert BDD100KLaneCategories.from_label("lane/single yellow") == 1
        assert BDD100KLaneCategories.from_label("lane/double yellow") == 2
        assert BDD100KLaneCategories.from_label("lane/road curb") == 3

    def test_from_label_enum_format(self):
        assert BDD100KLaneCategories.from_label("SINGLE_WHITE") == 0
        assert BDD100KLaneCategories.from_label("ROAD_CURB") == 3

    def test_from_id(self):
        assert BDD100KLaneCategories.from_id(0) == "single_white"
        assert BDD100KLaneCategories.from_id(3) == "road_curb"
        assert BDD100KLaneCategories.from_id(99) is None

    def test_from_label_invalid_raises(self):
        with pytest.raises(KeyError):
            BDD100KLaneCategories.from_label("lane/unknown type")


# ===================================================================
#  2. lane_utils Tests
# ===================================================================

class TestPolylineToLaneTarget:

    def test_basic_vertical_lane(self):
        # Near-vertical lane in a 720x1280 image
        pts = np.array([
            [640, 700], [640, 500], [640, 300], [640, 100]
        ], dtype=np.float32)

        target = polyline_to_lane_target(pts, img_h=720, img_w=1280)

        assert target is not None
        assert target.shape == (4 + NUM_LANE_POINTS,)
        # start_y should be close to 1.0 (bottom)
        assert target[0] > 0.8
        # start_x ~ 640/1279 ~ 0.5
        assert 0.4 < target[1] < 0.6
        # theta should be close to 0.5 (vertical = pi/2)
        assert 0.3 < target[2] < 0.7
        # length > 0
        assert target[3] > 0

    def test_short_lane(self):
        pts = np.array([[640, 700], [640, 695]], dtype=np.float32)
        target = polyline_to_lane_target(pts, img_h=720, img_w=1280)
        # Very short lane — may still return a target since it spans 2 y positions
        # but might produce < 2 valid sample points depending on y-grid alignment
        # Either None or a valid target is acceptable
        if target is not None:
            assert target.shape == (4 + NUM_LANE_POINTS,)

    def test_single_point_returns_none(self):
        pts = np.array([[640, 500]], dtype=np.float32)
        assert polyline_to_lane_target(pts, 720, 1280) is None

    def test_empty_returns_none(self):
        pts = np.array([], dtype=np.float32).reshape(0, 2)
        assert polyline_to_lane_target(pts, 720, 1280) is None

    def test_out_of_bounds_x_filtered(self):
        # Lane entirely outside image width
        pts = np.array([
            [1500, 700], [1500, 500], [1500, 300]
        ], dtype=np.float32)
        target = polyline_to_lane_target(pts, 720, 1280)
        # All x > img_w, so valid_count == 0 -> None
        assert target is None

    def test_invalid_positions_marked(self):
        # Lane only covers bottom half of image
        pts = np.array([
            [640, 700], [640, 600], [640, 500], [640, 400]
        ], dtype=np.float32)
        target = polyline_to_lane_target(pts, 720, 1280)
        assert target is not None
        # Top positions (y < 400) should be -1e5
        xs = target[4:]
        assert (xs < -1e4).any(), "Some positions should be invalid"
        assert (xs > -1e4).any(), "Some positions should be valid"

    def test_custom_n_offsets(self):
        pts = np.array([
            [640, 700], [640, 500], [640, 300], [640, 100]
        ], dtype=np.float32)
        target = polyline_to_lane_target(pts, 720, 1280, n_offsets=36)
        assert target is not None
        assert target.shape == (4 + 36,)


class TestBuildLaneTargets:

    def test_basic(self):
        lane_polys = _make_lane_polys()
        targets, categories = build_lane_targets(lane_polys, 720, 1280)

        assert targets.shape == (MAX_LANES, 6 + NUM_LANE_POINTS)
        assert categories.shape == (MAX_LANES,)

        # First 2 lanes should be valid
        assert targets[0, 0] == 1.0  # valid
        assert targets[1, 0] == 1.0
        # Rest should be zeros
        assert targets[2, 0] == 0.0

        # Categories
        assert categories[0].item() == 0  # single white
        assert categories[1].item() == 1  # single yellow
        assert categories[2].item() == -1  # empty

    def test_none_input(self):
        targets, categories = build_lane_targets(None, 720, 1280)
        assert targets.shape == (MAX_LANES, 6 + NUM_LANE_POINTS)
        assert (targets == 0).all()
        assert (categories == -1).all()

    def test_empty_list(self):
        targets, categories = build_lane_targets([], 720, 1280)
        assert (targets == 0).all()

    def test_crosswalk_skipped(self):
        polys = [{"points": np.array([[640, 700], [640, 100]], dtype=np.float32),
                  "category": "lane/crosswalk"}]
        targets, categories = build_lane_targets(polys, 720, 1280)
        assert (targets[:, 0] == 0).all(), "Crosswalk should be skipped"

    def test_max_lanes_truncation(self):
        # Create more lanes than MAX_LANES
        polys = []
        for i in range(MAX_LANES + 3):
            pts = np.array([
                [100 + i * 50, 700], [100 + i * 50, 100]
            ], dtype=np.float32)
            polys.append({"points": pts, "category": "lane/single white"})

        targets, categories = build_lane_targets(polys, 720, 1280)
        assert targets.shape[0] == MAX_LANES
        # At most MAX_LANES should be valid
        assert (targets[:, 0] == 1.0).sum() <= MAX_LANES

    def test_output_dtype(self):
        targets, categories = build_lane_targets(_make_lane_polys(), 720, 1280)
        assert targets.dtype == torch.float32
        assert categories.dtype == torch.int64


# ===================================================================
#  3. Augmentation Tests
# ===================================================================

class TestRandomPerspectiveLanePolys:

    def test_returns_five_values(self):
        img = _make_image(640, 640)
        labels = _make_labels()
        lane_polys = _make_lane_polys()

        result = random_perspective(img, None, None, labels, lane_polys=lane_polys,
                                    degrees=0, translate=0, scale=0, shear=0)
        assert len(result) == 5
        img_out, seg_out, drv_out, labels_out, lp_out = result
        assert lp_out is not None
        assert len(lp_out) == 2

    def test_identity_transform_preserves_points(self):
        img = _make_image(720, 1280)
        labels = np.zeros((0, 5))
        lane_polys = _make_lane_polys()
        orig_pts = [p["points"].copy() for p in lane_polys]

        _, _, _, _, lp_out = random_perspective(
            img, None, None, labels, lane_polys=lane_polys,
            degrees=0, translate=0, scale=0, shear=0
        )

        for i, poly in enumerate(lp_out):
            np.testing.assert_allclose(poly["points"], orig_pts[i], atol=1.0)

    def test_none_lane_polys_passthrough(self):
        img = _make_image(640, 640)
        labels = _make_labels()
        _, _, _, _, lp_out = random_perspective(
            img, None, None, labels, lane_polys=None,
            degrees=5, translate=0.1, scale=0.1, shear=5
        )
        assert lp_out is None

    def test_points_clipped_to_bounds(self):
        img = _make_image(640, 640)
        labels = np.zeros((0, 5))
        lane_polys = _make_lane_polys()

        random.seed(42)
        np.random.seed(42)
        _, _, _, _, lp_out = random_perspective(
            img, None, None, labels, lane_polys=lane_polys,
            degrees=30, translate=0.3, scale=0.3, shear=20
        )

        for poly in lp_out:
            assert (poly["points"][:, 0] >= 0).all()
            assert (poly["points"][:, 0] <= 639).all()
            assert (poly["points"][:, 1] >= 0).all()
            assert (poly["points"][:, 1] <= 639).all()

    def test_category_preserved(self):
        img = _make_image(720, 1280)
        labels = np.zeros((0, 5))
        lane_polys = _make_lane_polys()

        _, _, _, _, lp_out = random_perspective(
            img, None, None, labels, lane_polys=lane_polys,
            degrees=0, translate=0, scale=0, shear=0
        )
        assert lp_out[0]["category"] == "lane/single white"
        assert lp_out[1]["category"] == "lane/single yellow"


class TestFlipHorizontalLanePolys:

    def test_returns_five_values(self):
        img = _make_image(640, 640)
        labels = _make_labels()
        lane_polys = _make_lane_polys()
        result = flip_horizontal(img, None, None, labels, lane_polys=lane_polys)
        assert len(result) == 5

    def test_x_coords_mirrored(self):
        w = 1280
        img = _make_image(720, w)
        labels = np.zeros((0, 5))
        lane_polys = [{"points": np.array([[100, 500], [200, 300]], dtype=np.float32),
                       "category": "lane/single white"}]

        _, _, _, _, lp_out = flip_horizontal(img, None, None, labels, lane_polys=lane_polys)

        assert lp_out[0]["points"][0, 0] == pytest.approx(w - 1 - 100)
        assert lp_out[0]["points"][1, 0] == pytest.approx(w - 1 - 200)
        # y unchanged
        assert lp_out[0]["points"][0, 1] == 500
        assert lp_out[0]["points"][1, 1] == 300

    def test_none_passthrough(self):
        img = _make_image(640, 640)
        labels = _make_labels()
        _, _, _, _, lp_out = flip_horizontal(img, None, None, labels, lane_polys=None)
        assert lp_out is None


class TestLetterboxLanePolys:

    def test_returns_five_values(self):
        img = _make_image(720, 1280)
        labels = _make_labels()
        lane_polys = _make_lane_polys()
        result = letterbox_with_masks(img, None, None, labels,
                                      lane_polys=lane_polys, new_shape=(768, 1280))
        assert len(result) == 5

    def test_scale_and_offset_applied(self):
        img = _make_image(720, 1280)
        labels = np.zeros((0, 5))
        pts_orig = np.array([[640, 360]], dtype=np.float32)  # center of 1280x720
        lane_polys = [{"points": pts_orig.copy(), "category": "lane/single white"}]

        _, _, _, _, lp_out = letterbox_with_masks(
            img, None, None, labels, lane_polys=lane_polys, new_shape=(384, 640)
        )

        # 720x1280 -> letterbox to 384x640
        # r = min(384/720, 640/1280) = min(0.533, 0.5) = 0.5
        # new_unpad = (640, 360), pad_top = (384-360)/2 = 12
        r = 0.5
        expected_x = 640 * r  # + 0 left pad since width matches exactly
        expected_y = 360 * r + 12  # top pad
        assert lp_out[0]["points"][0, 0] == pytest.approx(expected_x, abs=1.0)
        assert lp_out[0]["points"][0, 1] == pytest.approx(expected_y, abs=1.0)


class TestApplyAugmentationsLanePolys:

    def test_returns_five_values(self):
        img = _make_image(640, 640)
        labels = _make_labels()
        lane_polys = _make_lane_polys()
        params = {"degrees": 0, "translate": 0, "scale": 0, "shear": 0,
                  "flip_prob": 0, "salt_prob": 0, "pepper_prob": 0}

        result = apply_augmentations(img, None, None, labels, params,
                                     img_size=(640, 640), lane_polys=lane_polys)
        assert len(result) == 5
        _, _, _, _, lp_out = result
        assert lp_out is not None

    def test_none_passthrough(self):
        img = _make_image(640, 640)
        labels = _make_labels()
        params = {"degrees": 5, "translate": 0.1, "scale": 0.1, "shear": 5,
                  "flip_prob": 0.5}
        _, _, _, _, lp_out = apply_augmentations(
            img, None, None, labels, params, img_size=(640, 640), lane_polys=None
        )
        assert lp_out is None


class TestMixupLanePolys:

    def test_concatenates_polys(self):
        img1 = _make_image(640, 640)
        img2 = _make_image(640, 640)
        labels1 = _make_labels(1)
        labels2 = _make_labels(1)
        lp1 = [{"points": np.array([[100, 500], [100, 100]], dtype=np.float32),
                "category": "lane/single white"}]
        lp2 = [{"points": np.array([[500, 600], [500, 200]], dtype=np.float32),
                "category": "lane/single yellow"}]

        _, _, _, _, mixed_lp = mixup_augmentation(
            img1, labels1, img2, labels2,
            lane_polys1=lp1, lane_polys2=lp2
        )

        assert len(mixed_lp) == 2
        assert mixed_lp[0]["category"] == "lane/single white"
        assert mixed_lp[1]["category"] == "lane/single yellow"

    def test_one_side_none(self):
        img1 = _make_image(640, 640)
        img2 = _make_image(640, 640)
        lp1 = [{"points": np.array([[100, 500], [100, 100]], dtype=np.float32),
                "category": "lane/single white"}]

        _, _, _, _, mixed_lp = mixup_augmentation(
            img1, _make_labels(1), img2, _make_labels(1),
            lane_polys1=lp1, lane_polys2=None
        )
        assert len(mixed_lp) == 1

    def test_both_none(self):
        img1, img2 = _make_image(640, 640), _make_image(640, 640)
        _, _, _, _, mixed_lp = mixup_augmentation(
            img1, _make_labels(1), img2, _make_labels(1),
            lane_polys1=None, lane_polys2=None
        )
        assert mixed_lp is None


class TestMosaicLanePolys:

    def test_returns_five_values(self):
        images = [_make_image(720, 1280) for _ in range(4)]
        bboxes = [[[100, 200, 300, 400]] for _ in range(4)]
        class_labels = [[2] for _ in range(4)]
        segs = [None] * 4
        drivables = [None] * 4
        lp_list = [_make_lane_polys() for _ in range(4)]

        result = mosaic_augmentation(
            images, bboxes, class_labels, segs, drivables,
            lane_polys_list=lp_list, output_size=(640, 640)
        )
        assert len(result) == 5
        _, _, _, _, mosaic_lp = result
        assert mosaic_lp is not None
        assert len(mosaic_lp) > 0  # at least some lanes should survive

    def test_none_passthrough(self):
        images = [_make_image(720, 1280) for _ in range(4)]
        bboxes = [[] for _ in range(4)]
        class_labels = [[] for _ in range(4)]
        segs = [None] * 4
        drivables = [None] * 4

        _, _, _, _, mosaic_lp = mosaic_augmentation(
            images, bboxes, class_labels, segs, drivables,
            lane_polys_list=None, output_size=(640, 640)
        )
        assert mosaic_lp is None

    def test_points_within_output_bounds(self):
        h_out, w_out = 640, 640
        images = [_make_image(720, 1280) for _ in range(4)]
        bboxes = [[] for _ in range(4)]
        class_labels = [[] for _ in range(4)]
        segs = [None] * 4
        drivables = [None] * 4
        lp_list = [_make_lane_polys() for _ in range(4)]

        _, _, _, _, mosaic_lp = mosaic_augmentation(
            images, bboxes, class_labels, segs, drivables,
            lane_polys_list=lp_list, output_size=(h_out, w_out)
        )

        for poly in mosaic_lp:
            assert (poly["points"][:, 0] >= 0).all()
            assert (poly["points"][:, 0] < w_out).all()
            assert (poly["points"][:, 1] >= 0).all()
            assert (poly["points"][:, 1] < h_out).all()


# ===================================================================
#  4. BDDPreprocessor Tests
# ===================================================================

class TestPreprocessorLaneMethods:

    def setup_method(self):
        self.preprocessor = BDDPreprocessor({
            "image_resize": (768, 1280),
            "original_image_size": (720, 1280),
        })

    def test_load_lane_annotations(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _make_bdd_json(tmp, num_lanes=3)
            lane_polys = self.preprocessor.load_lane_annotations(path)

            assert isinstance(lane_polys, list)
            assert len(lane_polys) == 3
            for poly in lane_polys:
                assert "points" in poly
                assert "category" in poly
                assert poly["points"].shape[1] == 2
                assert poly["points"].dtype == np.float32
                assert poly["category"].startswith("lane/")

    def test_load_lane_annotations_no_lanes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _make_bdd_json(tmp, num_lanes=0)
            lane_polys = self.preprocessor.load_lane_annotations(path)
            assert lane_polys == []

    def test_load_lane_annotations_missing_file(self):
        lane_polys = self.preprocessor.load_lane_annotations("/nonexistent/path.json")
        assert lane_polys == []

    def test_transform_lane_points_resize(self):
        lane_polys = [{"points": np.array([[640, 360]], dtype=np.float32),
                       "category": "lane/single white"}]

        result = self.preprocessor.transform_lane_points_resize(lane_polys, 720, 1280)

        # 720x1280 -> LongestMaxSize(1280): scale=1.0, new_h=720, new_w=1280
        # PadIfNeeded(768, 1280): pad_top=(768-720)//2=24
        assert result[0]["points"][0, 0] == pytest.approx(640, abs=1)
        assert result[0]["points"][0, 1] == pytest.approx(360 + 24, abs=1)

    def test_transform_lane_points_resize_none(self):
        assert self.preprocessor.transform_lane_points_resize(None, 720, 1280) is None

    def test_standard_augmentations_returns_five(self):
        img = _make_image(720, 1280)
        bboxes = [[100, 200, 300, 400]]
        class_labels = [2]
        lane_polys = _make_lane_polys()

        result = self.preprocessor.standard_augmentations(
            img, None, None, bboxes, class_labels, lane_polys=lane_polys
        )
        assert len(result) == 5
        img_out, seg_out, drv_out, labels_out, lp_out = result
        assert lp_out is not None

    def test_mosaic_augmentation_returns_five(self):
        images = [_make_image(720, 1280) for _ in range(4)]
        bboxes = [[[100, 200, 300, 400]] for _ in range(4)]
        class_labels = [[2] for _ in range(4)]
        segs = [None] * 4
        drivables = [None] * 4
        lp_list = [_make_lane_polys() for _ in range(4)]

        result = self.preprocessor.mosaic_augmentation(
            images, bboxes, class_labels, segs, drivables,
            lane_polys_list=lp_list
        )
        assert len(result) == 5

    def test_mixup_augmentation_returns_five(self):
        img1 = _make_image(720, 1280)
        img2 = _make_image(720, 1280)
        bboxes = [[100, 200, 300, 400]]
        class_labels = [2]
        lp = _make_lane_polys()

        result = self.preprocessor.mixup_augmentation(
            img1, bboxes, class_labels, None, None,
            img2, bboxes, class_labels, None, None,
            lane_polys1=lp, lane_polys2=lp
        )
        assert len(result) == 5


# ===================================================================
#  5. Collate Function Tests
# ===================================================================

class TestCollateFnWithLanes:

    def test_basic_collation(self):
        batch = [_make_batch_item(n_lanes=2), _make_batch_item(n_lanes=1)]
        result = BDDPreprocessor.collate_fn(batch)

        assert result["images"].shape == (2, 3, 640, 640)
        assert result["lanes"].shape == (2, MAX_LANES, 6 + NUM_LANE_POINTS)
        assert result["lane_categories"].shape == (2, MAX_LANES)

    def test_no_lane_targets(self):
        item = _make_batch_item(n_lanes=0)
        # Even with no valid lanes, the tensors should exist (all zeros)
        batch = [item]
        result = BDDPreprocessor.collate_fn(batch)

        assert result["lanes"].shape == (1, MAX_LANES, 6 + NUM_LANE_POINTS)
        assert (result["lanes"][0, :, 0] == 0).all(), "No valid lanes expected"

    def test_batch_indices_in_detections(self):
        batch = [_make_batch_item(n_det=1), _make_batch_item(n_det=2)]
        result = BDDPreprocessor.collate_fn(batch)

        assert result["detections"][0, 0] == 0
        assert result["detections"][1, 0] == 1
        assert result["detections"][2, 0] == 1

    def test_all_keys_present(self):
        batch = [_make_batch_item()]
        result = BDDPreprocessor.collate_fn(batch)
        expected_keys = {
            "images", "clean_images", "detections",
            "segmentation_masks", "drivable_area_seg",
            "lanes", "lane_categories",
            "image_paths", "scene_attributes"
        }
        assert set(result.keys()) == expected_keys


# ===================================================================
#  6. Visualize Batch Tests
# ===================================================================

class TestVisualizeBatchWithLanes:

    def test_with_lane_targets(self, tmp_path):
        batch = [_make_batch_item(n_lanes=2), _make_batch_item(n_lanes=1)]
        result = BDDPreprocessor.collate_fn(batch)

        visualize_batch(
            result["images"],
            result["segmentation_masks"],
            result["drivable_area_seg"],
            result["detections"],
            save_dir=str(tmp_path),
            batch_index=0,
            lane_targets=result["lanes"],
            lane_categories=result["lane_categories"],
        )

        assert (tmp_path / "sample_batch_0.png").exists()
        assert (tmp_path / "sample_batch_seg_0.png").exists()
        assert (tmp_path / "sample_batch_drivable_0.png").exists()

    def test_without_lane_targets(self, tmp_path):
        """Backward compatibility: lane_targets=None should work fine."""
        batch = [_make_batch_item(n_lanes=0)]
        result = BDDPreprocessor.collate_fn(batch)

        visualize_batch(
            result["images"],
            result["segmentation_masks"],
            result["drivable_area_seg"],
            result["detections"],
            save_dir=str(tmp_path),
            batch_index=0,
        )
        assert (tmp_path / "sample_batch_0.png").exists()


# ===================================================================
#  7. End-to-End Pipeline Test (synthetic)
# ===================================================================

class TestEndToEndPipeline:
    """Validates the full flow: raw polys -> augmentation -> target building -> collation."""

    def test_standard_augmentation_pipeline(self):
        preprocessor = BDDPreprocessor({
            "image_resize": (640, 640),
            "original_image_size": (720, 1280),
        })

        img = _make_image(720, 1280)
        bboxes = [[100, 200, 300, 400], [500, 100, 700, 300]]
        class_labels = [2, 0]
        lane_polys = _make_lane_polys()

        img_out, seg_out, drv_out, labels_out, lp_out = preprocessor.standard_augmentations(
            img, None, None, bboxes, class_labels, lane_polys=lane_polys
        )

        # Build targets from augmented polylines
        h, w = img_out.shape[:2]
        lane_targets, lane_categories = build_lane_targets(lp_out, h, w)

        assert lane_targets.shape == (MAX_LANES, 6 + NUM_LANE_POINTS)
        assert lane_categories.shape == (MAX_LANES,)

        # At least 1 lane should survive augmentation
        n_valid = (lane_targets[:, 0] == 1.0).sum().item()
        assert n_valid >= 1, f"Expected at least 1 valid lane, got {n_valid}"

    def test_non_augmented_path(self):
        preprocessor = BDDPreprocessor({
            "image_resize": (768, 1280),
            "original_image_size": (720, 1280),
        })

        img = _make_image(720, 1280)
        lane_polys = _make_lane_polys()

        # Simulate the non-augmented path
        orig_h, orig_w = img.shape[:2]
        t = preprocessor.transformation(image=img, bboxes=[[100, 200, 300, 400]],
                                        class_labels=[2])
        img_out = t["image"]

        lp_out = preprocessor.transform_lane_points_resize(lane_polys, orig_h, orig_w)

        h, w = img_out.shape[:2]
        lane_targets, lane_categories = build_lane_targets(lp_out, h, w)

        assert lane_targets.shape == (MAX_LANES, 6 + NUM_LANE_POINTS)
        n_valid = (lane_targets[:, 0] == 1.0).sum().item()
        assert n_valid == 2


# ===================================================================
#  Integration tests (need real data)
# ===================================================================

BDD_DATA_ROOT = "panoptic_perception/BDD100k"
BDD_IMAGES = os.path.join(BDD_DATA_ROOT, "100k/100k")
BDD_LABELS = os.path.join(BDD_DATA_ROOT, "bdd100k_labels/100k")
BDD_DRIVABLE = os.path.join(BDD_DATA_ROOT, "bdd100k_drivable_maps/labels")

HAS_BDD_DATA = os.path.isdir(BDD_IMAGES) and os.path.isdir(BDD_LABELS)


@pytest.mark.skipif(not HAS_BDD_DATA, reason="BDD100K data not found")
class TestIntegrationWithRealData:

    @pytest.fixture
    def dataset(self):
        dataset_kwargs = {
            "images_dir": BDD_IMAGES,
            "detection_annotations_dir": BDD_LABELS,
            "segmentation_annotations_dir": "",
            "drivable_annotations_dir": BDD_DRIVABLE,
            "preprocessor_kwargs": {
                "image_resize": (640, 640),
                "original_image_size": (720, 1280),
            }
        }
        return BDD100KDataset(dataset_kwargs, dataset_type="train",
                              perform_augmentation=False)

    def test_getitem_has_lane_keys(self, dataset):
        sample = dataset[0]
        assert "lane_targets" in sample
        assert "lane_categories" in sample
        assert sample["lane_targets"].shape == (MAX_LANES, 6 + NUM_LANE_POINTS)
        assert sample["lane_categories"].shape == (MAX_LANES,)

    def test_dataloader_batch_has_lane_keys(self, dataset):
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=8, shuffle=False,
                            num_workers=0, collate_fn=BDDPreprocessor.collate_fn)
        batch = next(iter(loader))

        assert "lanes" in batch
        assert "lane_categories" in batch
        assert batch["lanes"].shape[0] == 4
        assert batch["lanes"].shape[1] == MAX_LANES
        assert batch["lanes"].shape[2] == 6 + NUM_LANE_POINTS

    def test_visualize_with_real_data(self, dataset, tmp_path):
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=8, shuffle=False,
                            num_workers=0, collate_fn=BDDPreprocessor.collate_fn)
        batch = next(iter(loader))

        visualize_batch(
            batch["images"], batch["segmentation_masks"],
            batch["drivable_area_seg"], batch["detections"],
            save_dir=str(tmp_path), batch_index=0,
            lane_targets=batch["lanes"],
            lane_categories=batch["lane_categories"],
        )
        assert (tmp_path / "sample_batch_0.png").exists()

    def test_visualize_full_batch(self, dataset):
        """Visualize entire batch: detections + drivable + lanes composited on one image per sample."""
        import cv2
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=8, shuffle=True,
                            num_workers=0, collate_fn=BDDPreprocessor.collate_fn)
        batch = next(iter(loader))

        save_dir = "visualizations/dataset_batch_vis"
        os.makedirs(save_dir, exist_ok=True)
        batch_size = batch["images"].shape[0]
        H, W = batch["images"].shape[2], batch["images"].shape[3]

        for b in range(batch_size):
            img = batch["images"][b].permute(1, 2, 0).cpu().numpy()
            img = (img * 255).astype(np.uint8).copy()

            # --- drivable overlay ---
            if batch.get("drivable_area_seg") is not None:
                drv = batch["drivable_area_seg"][b].cpu().numpy()
                overlay = img.copy()
                overlay[drv == 1] = [0, 200, 0]
                overlay[drv == 2] = [0, 200, 200]
                img = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)

            # --- detection boxes ---
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

            # --- lane polylines ---
            if batch.get("lanes") is not None:
                lt = batch["lanes"][b]
                lc = batch["lane_categories"][b] if batch.get("lane_categories") is not None else None
                n_offsets = lt.shape[1] - 6
                prior_ys = np.linspace(1.0, 0.0, n_offsets)

                for i in range(lt.shape[0]):
                    if lt[i, 0].item() < 0.5:
                        continue
                    xs = lt[i, 6:].cpu().numpy()
                    cat_idx = int(lc[i].item()) if lc is not None and lc[i].item() >= 0 else 0
                    color = LANE_VIS_COLORS[cat_idx % len(LANE_VIS_COLORS)]
                    pts = []
                    for j in range(n_offsets):
                        if xs[j] > -1e4:
                            pts.append((int(xs[j] * (W - 1)), int(prior_ys[j] * (H - 1))))
                    for k in range(len(pts) - 1):
                        cv2.line(img, pts[k], pts[k + 1], color, 2)

            cv2.imwrite(f"{save_dir}/sample_{b}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        for b in range(batch_size):
            assert os.path.exists(f"{save_dir}/sample_{b}.png")
