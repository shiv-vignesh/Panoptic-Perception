"""
YOLOP-style data augmentations for panoptic perception tasks.

All public augmentation entry points operate on FrameData. The legacy
parallel-list math is preserved internally — converted at the boundary via
FrameData.labels_array / set_labels_array and lane_polys_legacy /
set_lane_polys_legacy.
"""

import cv2
import numpy as np
import random
import math
from copy import copy

from panoptic_perception.dataset.types import FrameData

def _random_perspective_arrays(
    img, seg, drivable, labels, lane_polys=None,
    degrees=10, translate=0.1, scale=0.1, shear=10
):
    h, w = img.shape[:2]

    # --- MATRIX BUILDING ---
    C = np.eye(3)
    C[0, 2] = -w / 2
    C[1, 2] = -h / 2

    a = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1 + scale)
    R = np.eye(3)
    R[:2] = cv2.getRotationMatrix2D((0, 0), a, s)

    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)

    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * w
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * h

    M = T @ S @ R @ C

    # --- WARP IMAGE/MASKS ---
    img = cv2.warpAffine(img, M[:2], (w, h), borderValue=114)
    if seg is not None:
        seg = cv2.warpAffine(seg, M[:2], (w, h), flags=cv2.INTER_NEAREST)
    if drivable is not None:
        drivable = cv2.warpAffine(drivable, M[:2], (w, h), flags=cv2.INTER_NEAREST)

    # --- WARP LABELS ---
    if len(labels):
        new_labels = []
        for cls, cx, cy, bw, bh in labels:
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w
            y2 = (cy + bh / 2) * h

            corners = np.array([
                [x1, y1, 1],
                [x2, y1, 1],
                [x2, y2, 1],
                [x1, y2, 1],
            ])
            tc = corners @ M.T
            xs, ys = tc[:, 0], tc[:, 1]

            x1n, x2n = xs.min(), xs.max()
            y1n, y2n = ys.min(), ys.max()

            bw_n = (x2n - x1n) / w
            bh_n = (y2n - y1n) / h
            cx_n = (x1n + x2n) / 2 / w
            cy_n = (y1n + y2n) / 2 / h

            if bw_n > 0.002 and bh_n > 0.002:
                new_labels.append([cls, cx_n, cy_n, bw_n, bh_n])

        labels = np.array(new_labels) if new_labels else np.zeros((0, 5))

    # --- WARP LANE POLYLINES ---
    if lane_polys is not None:
        new_lane_polys = []
        for poly in lane_polys:
            pts = poly["points"]
            if len(pts) < 2:
                continue
            ones = np.ones((pts.shape[0], 1))
            pts_homo = np.hstack([pts, ones])
            pts_t = (M @ pts_homo.T).T[:, :2]

            pts_t[:, 0] = np.clip(pts_t[:, 0], 0, w - 1)
            pts_t[:, 1] = np.clip(pts_t[:, 1], 0, h - 1)

            if len(pts_t) >= 2:
                new_lane_polys.append({
                    "points": pts_t.astype(np.float32),
                    "category": poly["category"],
                })
        lane_polys = new_lane_polys

    return img, seg, drivable, labels, lane_polys


def _flip_horizontal_arrays(img, seg, drivable, labels, lane_polys=None):
    w = img.shape[1]
    img = np.fliplr(img).copy()
    if seg is not None:
        seg = np.fliplr(seg).copy()
    if drivable is not None:
        drivable = np.fliplr(drivable).copy()
    if len(labels):
        labels[:, 1] = 1 - labels[:, 1]
    if lane_polys is not None:
        for poly in lane_polys:
            poly["points"][:, 0] = w - 1 - poly["points"][:, 0]
    return img, seg, drivable, labels, lane_polys


def _letterbox_arrays(img, seg, drivable, labels, lane_polys=None,
                      new_shape=(640, 640), color=(114, 114, 114)):
    h0, w0 = img.shape[:2]
    new_h, new_w = new_shape

    r = min(new_h / h0, new_w / w0)
    new_unpad = (int(w0 * r), int(h0 * r))

    dw = (new_w - new_unpad[0]) / 2
    dh = (new_h - new_unpad[1]) / 2

    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)

    if seg is not None:
        seg = cv2.resize(seg, new_unpad, interpolation=cv2.INTER_NEAREST)
        seg = cv2.copyMakeBorder(seg, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=0)

    if drivable is not None:
        drivable = cv2.resize(drivable, new_unpad, interpolation=cv2.INTER_NEAREST)
        drivable = cv2.copyMakeBorder(drivable, top, bottom, left, right,
                                      cv2.BORDER_CONSTANT, value=0)

    if len(labels):
        labels = labels.copy()
        labels[:, 1] *= w0
        labels[:, 2] *= h0
        labels[:, 3] *= w0
        labels[:, 4] *= h0

        labels[:, 1] = labels[:, 1] * r + left
        labels[:, 2] = labels[:, 2] * r + top
        labels[:, 3] *= r
        labels[:, 4] *= r

        labels[:, 1] /= new_w
        labels[:, 2] /= new_h
        labels[:, 3] /= new_w
        labels[:, 4] /= new_h

    if lane_polys is not None:
        for poly in lane_polys:
            poly["points"][:, 0] = poly["points"][:, 0] * r + left
            poly["points"][:, 1] = poly["points"][:, 1] * r + top

    return img, seg, drivable, labels, lane_polys


def _mixup_arrays(img1, labels1, img2, labels2, seg1=None, seg2=None,
                  drivable1=None, drivable2=None, alpha=0.5,
                  lane_polys1=None, lane_polys2=None):
    lam = np.random.beta(alpha, alpha)
    mixed_img = (lam * img1.astype(np.float32) +
                 (1 - lam) * img2.astype(np.float32)).astype(np.uint8)

    if len(labels1) > 0 and len(labels2) > 0:
        mixed_labels = np.concatenate([labels1, labels2], axis=0)
    elif len(labels1) > 0:
        mixed_labels = labels1
    elif len(labels2) > 0:
        mixed_labels = labels2
    else:
        mixed_labels = np.zeros((0, 5))

    if seg1 is not None and seg2 is not None:
        mixed_seg = seg1 if lam > 0.5 else seg2
    else:
        mixed_seg = seg1 if seg1 is not None else seg2

    if drivable1 is not None and drivable2 is not None:
        mixed_drivable = drivable1 if lam > 0.5 else drivable2
    else:
        mixed_drivable = drivable1 if drivable1 is not None else drivable2

    mixed_lane_polys = None
    if lane_polys1 is not None or lane_polys2 is not None:
        mixed_lane_polys = []
        if lane_polys1 is not None:
            mixed_lane_polys.extend(lane_polys1)
        if lane_polys2 is not None:
            mixed_lane_polys.extend(lane_polys2)

    return mixed_img, mixed_labels, mixed_seg, mixed_drivable, mixed_lane_polys


def _copy_paste_arrays(img, labels, source_img, source_labels,
                       target_classes=[1, 3], max_instances=3):
    if len(source_labels) == 0:
        return img, labels

    h, w = img.shape[:2]
    img = img.copy()
    labels_list = list(labels) if len(labels) > 0 else []

    paste_candidates = [
        label for label in source_labels if int(label[0]) in target_classes
    ]
    if len(paste_candidates) == 0:
        return img, labels

    num_to_paste = min(len(paste_candidates), max_instances)
    instances_to_paste = random.sample(paste_candidates, num_to_paste)

    sh, sw = source_img.shape[:2]

    for instance in instances_to_paste:
        cls, cx, cy, bw, bh = instance

        x1_src = int((cx - bw / 2) * sw)
        y1_src = int((cy - bh / 2) * sh)
        x2_src = int((cx + bw / 2) * sw)
        y2_src = int((cy + bh / 2) * sh)

        x1_src = max(0, x1_src)
        y1_src = max(0, y1_src)
        x2_src = min(sw, x2_src)
        y2_src = min(sh, y2_src)

        instance_crop = source_img[y1_src:y2_src, x1_src:x2_src]
        if instance_crop.size == 0:
            continue

        crop_h, crop_w = instance_crop.shape[:2]
        if crop_h >= h or crop_w >= w:
            continue

        paste_x = random.randint(0, w - crop_w)
        paste_y = random.randint(0, h - crop_h)

        img[paste_y:paste_y + crop_h, paste_x:paste_x + crop_w] = instance_crop

        new_cx = (paste_x + crop_w / 2) / w
        new_cy = (paste_y + crop_h / 2) / h
        new_w = crop_w / w
        new_h = crop_h / h

        labels_list.append([cls, new_cx, new_cy, new_w, new_h])

    return img, np.array(labels_list) if labels_list else np.zeros((0, 5))


# ============================================================================
# PUBLIC FRAMEDATA-NATIVE API
# ============================================================================

def random_perspective(frame: FrameData, degrees=10, translate=0.1,
                       scale=0.1, shear=10) -> FrameData:
    """Random perspective / affine warp. Mutates and returns the input FrameData
    (image/masks/detections/lane_polys all updated). Does not deep-copy."""
    labels = frame.labels_array()
    lane_polys = frame.lane_polys_legacy()

    img, seg, drivable, labels, lane_polys = _random_perspective_arrays(
        frame.image, frame.seg, frame.drivable, labels, lane_polys=lane_polys,
        degrees=degrees, translate=translate, scale=scale, shear=shear,
    )

    frame.image = img
    frame.seg = seg
    frame.drivable = drivable
    frame.set_labels_array(labels)
    frame.set_lane_polys_legacy(lane_polys)
    return frame


def augment_hsv(frame: FrameData, hgain=0.5, sgain=0.5, vgain=0.5) -> FrameData:
    """HSV color jitter. Mutates frame.image in place. Other fields untouched."""
    img = frame.image
    if hgain == 0 and sgain == 0 and vgain == 0:
        return frame

    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1

    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype

    x = np.arange(0, 256, dtype=r.dtype)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue),
                         cv2.LUT(sat, lut_sat),
                         cv2.LUT(val, lut_val)))

    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)
    return frame


def flip_horizontal(frame: FrameData) -> FrameData:
    """Horizontal flip. Updates image, masks, detections, lane_polys."""
    labels = frame.labels_array()
    lane_polys = frame.lane_polys_legacy()

    img, seg, drivable, labels, lane_polys = _flip_horizontal_arrays(
        frame.image, frame.seg, frame.drivable, labels, lane_polys=lane_polys,
    )

    frame.image = img
    frame.seg = seg
    frame.drivable = drivable
    frame.set_labels_array(labels)
    frame.set_lane_polys_legacy(lane_polys)
    return frame


def letterbox_with_masks(frame: FrameData,
                         new_shape=(640, 640),
                         color=(114, 114, 114)) -> FrameData:
    """Letterbox resize with aspect-ratio preservation. Pads with `color`."""
    labels = frame.labels_array()
    lane_polys = frame.lane_polys_legacy()

    img, seg, drivable, labels, lane_polys = _letterbox_arrays(
        frame.image, frame.seg, frame.drivable, labels,
        lane_polys=lane_polys, new_shape=new_shape, color=color,
    )

    frame.image = img
    frame.seg = seg
    frame.drivable = drivable
    frame.set_labels_array(labels)
    frame.set_lane_polys_legacy(lane_polys)
    return frame


def apply_salt_pepper(frame: FrameData, salt_prob=0.01,
                      pepper_prob=0.01) -> FrameData:
    """Salt-and-pepper noise. Mutates frame.image in place."""
    if salt_prob == 0 and pepper_prob == 0:
        return frame

    noisy_img = frame.image.copy()
    h, w, _ = frame.image.shape

    if salt_prob > 0:
        salt_mask = np.random.random((h, w)) < salt_prob
        noisy_img[salt_mask] = 255
    if pepper_prob > 0:
        pepper_mask = np.random.random((h, w)) < pepper_prob
        noisy_img[pepper_mask] = 0

    frame.image = noisy_img
    return frame


def mixup_augmentation(frame1: FrameData, frame2: FrameData,
                       alpha: float = 0.5) -> FrameData:
    """MixUp blend of two frames. Returns a new FrameData; inputs untouched.

    The returned frame inherits image_path from frame1.
    """
    labels1 = frame1.labels_array()
    labels2 = frame2.labels_array()
    lane_polys1 = frame1.lane_polys_legacy()
    lane_polys2 = frame2.lane_polys_legacy()

    mixed_img, mixed_labels, mixed_seg, mixed_drivable, mixed_lane_polys = \
        _mixup_arrays(
            frame1.image, labels1, frame2.image, labels2,
            seg1=frame1.seg, seg2=frame2.seg,
            drivable1=frame1.drivable, drivable2=frame2.drivable,
            alpha=alpha,
            lane_polys1=lane_polys1, lane_polys2=lane_polys2,
        )

    mixed = FrameData(
        image=mixed_img,
        image_path=frame1.image_path,
        seg=mixed_seg,
        drivable=mixed_drivable,
    )
    mixed.set_labels_array(mixed_labels)
    mixed.set_lane_polys_legacy(mixed_lane_polys)
    return mixed


def copy_paste_instances(target: FrameData, source: FrameData,
                         target_classes=[1, 3],
                         max_instances=3) -> FrameData:
    """Copy a few instances of target_classes from `source` onto `target`.
    Mutates and returns `target`. Only image and detections are touched.
    """
    target_labels = target.labels_array()
    source_labels = source.labels_array()

    img, labels = _copy_paste_arrays(
        target.image, target_labels, source.image, source_labels,
        target_classes=target_classes, max_instances=max_instances,
    )

    target.image = img
    target.set_labels_array(labels)
    return target


def apply_augmentations(frame: FrameData, params: dict,
                        img_size=(640, 640)) -> FrameData:
    """Standard training-time augmentation pipeline:
       1) random perspective  2) HSV  3) salt/pepper  4) flip  5) letterbox.

    All steps run on the same FrameData; the input is mutated and returned.
    """
    # 1) GEOMETRIC
    frame = random_perspective(
        frame,
        degrees=params.get("degrees", 10),
        translate=params.get("translate", 0.1),
        scale=params.get("scale", 0.1),
        shear=params.get("shear", 10),
    )

    # 2) HSV
    augment_hsv(
        frame,
        params.get("hsv_h", 0.015),
        params.get("hsv_s", 0.7),
        params.get("hsv_v", 0.4),
    )

    # 3) SALT & PEPPER
    salt_prob = params.get("salt_prob", 0.0)
    pepper_prob = params.get("pepper_prob", 0.0)
    if (salt_prob > 0 or pepper_prob > 0) and random.random() < 0.5:
        apply_salt_pepper(frame, salt_prob=salt_prob, pepper_prob=pepper_prob)

    # 4) FLIP
    if random.random() < params.get("flip_prob", 0.5):
        frame = flip_horizontal(frame)

    # 5) LETTERBOX
    frame = letterbox_with_masks(frame, new_shape=img_size)

    return frame
