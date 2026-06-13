"""
Mosaic augmentation for YOLOP-style training.
Combines 4 images to create diverse scenes with varied scales.
Particularly effective for small objects and long-tail classes.

Public API takes List[FrameData]; legacy parallel-list math is contained
in the private _mosaic_arrays implementation.
"""

import cv2
import numpy as np
import random
from typing import List, Tuple

from panoptic_perception.dataset.types import FrameData


def _mosaic_arrays(images_list, bboxes_list, class_labels_list,
                   segs_list, drivables_list, lane_polys_list=None,
                   output_size=(640, 640)):
    """Internal: legacy parallel-list mosaic implementation.

    Args:
        images_list: list of 4 images (each HWC)
        bboxes_list: list of 4 bbox arrays in pascal_voc pixel coords
                     [[x1, y1, x2, y2], ...]
        class_labels_list: list of 4 class label arrays
        segs_list: list of 4 seg masks or None
        drivables_list: list of 4 drivable masks or None
        lane_polys_list: list of 4 lane-poly lists or None
        output_size: (h, w) of the output mosaic image

    Returns:
        mosaic_img, mosaic_labels (Nx5 normalized xywh),
        mosaic_seg, mosaic_drivable, mosaic_lane_polys
    """
    assert len(images_list) == 4, "Mosaic requires exactly 4 images"

    h_out, w_out = output_size

    mosaic_img = np.full((h_out, w_out, 3), 114, dtype=np.uint8)
    mosaic_labels = []

    has_seg = segs_list[0] is not None
    has_drivable = drivables_list[0] is not None

    mosaic_seg = np.zeros((h_out, w_out), dtype=np.uint8) if has_seg else None
    mosaic_drivable = np.zeros((h_out, w_out), dtype=np.uint8) if has_drivable else None
    mosaic_lane_polys = [] if lane_polys_list is not None else None

    yc = int(random.uniform(0.4 * h_out, 0.6 * h_out))
    xc = int(random.uniform(0.4 * w_out, 0.6 * w_out))

    placements = [
        (0, 0, xc, yc),           # top-left
        (xc, 0, w_out, yc),       # top-right
        (0, yc, xc, h_out),       # bottom-left
        (xc, yc, w_out, h_out),   # bottom-right
    ]

    for idx, (img, bboxes, class_labels, seg, drivable) in enumerate(
        zip(images_list, bboxes_list, class_labels_list, segs_list, drivables_list)
    ):
        h, w = img.shape[:2]

        x1_place, y1_place, x2_place, y2_place = placements[idx]
        place_h = y2_place - y1_place
        place_w = x2_place - x1_place

        scale = min(place_w / w, place_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        if has_seg and seg is not None:
            seg_resized = cv2.resize(seg, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        if has_drivable and drivable is not None:
            drivable_resized = cv2.resize(drivable, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        x1_actual, y1_actual = x1_place, y1_place
        x2_actual = min(x1_actual + new_w, w_out)
        y2_actual = min(y1_actual + new_h, h_out)

        mosaic_img[y1_actual:y2_actual, x1_actual:x2_actual] = img_resized[
            :y2_actual - y1_actual, :x2_actual - x1_actual
        ]

        if has_seg and seg is not None:
            mosaic_seg[y1_actual:y2_actual, x1_actual:x2_actual] = seg_resized[
                :y2_actual - y1_actual, :x2_actual - x1_actual
            ]
        if has_drivable and drivable is not None:
            mosaic_drivable[y1_actual:y2_actual, x1_actual:x2_actual] = drivable_resized[
                :y2_actual - y1_actual, :x2_actual - x1_actual
            ]

        if len(bboxes) > 0:
            for bbox, cls in zip(bboxes, class_labels):
                x1, y1, x2, y2 = bbox

                x1_scaled = x1 * scale
                y1_scaled = y1 * scale
                x2_scaled = x2 * scale
                y2_scaled = y2 * scale

                x1_out = x1_scaled + x1_place
                y1_out = y1_scaled + y1_place
                x2_out = x2_scaled + x1_place
                y2_out = y2_scaled + y1_place

                x1_out = max(0, min(x1_out, w_out))
                y1_out = max(0, min(y1_out, h_out))
                x2_out = max(0, min(x2_out, w_out))
                y2_out = max(0, min(y2_out, h_out))

                cx = (x1_out + x2_out) / 2 / w_out
                cy = (y1_out + y2_out) / 2 / h_out
                bw = (x2_out - x1_out) / w_out
                bh = (y2_out - y1_out) / h_out

                if 0 <= cx <= 1 and 0 <= cy <= 1 and bw > 0.001 and bh > 0.001:
                    mosaic_labels.append([cls, cx, cy, bw, bh])

        if lane_polys_list is not None and lane_polys_list[idx] is not None:
            for poly in lane_polys_list[idx]:
                pts = poly["points"].copy()
                pts[:, 0] = pts[:, 0] * scale + x1_place
                pts[:, 1] = pts[:, 1] * scale + y1_place

                valid = ((pts[:, 0] >= 0) & (pts[:, 0] < w_out) &
                         (pts[:, 1] >= 0) & (pts[:, 1] < h_out))
                pts_valid = pts[valid]

                if len(pts_valid) >= 2:
                    mosaic_lane_polys.append({
                        "points": pts_valid.astype(np.float32),
                        "category": poly["category"],
                    })

    mosaic_labels = np.array(mosaic_labels) if mosaic_labels else np.zeros((0, 5))

    return mosaic_img, mosaic_labels, mosaic_seg, mosaic_drivable, mosaic_lane_polys


# ============================================================================
# PUBLIC FRAMEDATA-NATIVE API
# ============================================================================

def mosaic_augmentation(items: List[FrameData],
                        output_size: Tuple[int, int] = (640, 640)) -> FrameData:
    """Build a 4-image mosaic from a list of FrameData.

    Args:
        items: exactly 4 FrameData. Each contributes its image, masks,
               detections, and lane_polys to one quadrant.
        output_size: (height, width) of the resulting mosaic image.

    Returns:
        A new FrameData whose image is the mosaic, with seg/drivable
        composited and detections / lane_polys offset into the mosaic
        coordinate space. image_path is inherited from items[0].
    """
    assert len(items) == 4, f"Mosaic requires exactly 4 frames, got {len(items)}"

    images_list = [fd.image for fd in items]
    segs_list = [fd.seg for fd in items]
    drivables_list = [fd.drivable for fd in items]

    # Build per-frame parallel detection arrays (pixel xyxy + class labels).
    bboxes_list, class_labels_list = [], []
    for fd in items:
        if fd.frame_detections is None or len(fd.frame_detections) == 0:
            bboxes_list.append([])
            class_labels_list.append([])
            continue
        bboxes_list.append([d.bbox.to_list() for d in fd.frame_detections.detections])
        class_labels_list.append([d.label_id for d in fd.frame_detections.detections])

    # Lane polys: legacy list-of-dict form. If ANY frame has lane_polys we
    # pass through; the internal mosaic handles per-frame None entries.
    any_lanes = any(fd.lane_polys is not None for fd in items)
    lane_polys_list = (
        [fd.lane_polys_legacy() for fd in items] if any_lanes else None
    )

    mosaic_img, mosaic_labels, mosaic_seg, mosaic_drivable, mosaic_lane_polys = \
        _mosaic_arrays(
            images_list, bboxes_list, class_labels_list,
            segs_list, drivables_list,
            lane_polys_list=lane_polys_list,
            output_size=output_size,
        )

    result = FrameData(
        image=mosaic_img,
        image_path=items[0].image_path,
        seg=mosaic_seg,
        drivable=mosaic_drivable,
    )
    result.set_labels_array(mosaic_labels)
    result.set_lane_polys_legacy(mosaic_lane_polys)
    return result
