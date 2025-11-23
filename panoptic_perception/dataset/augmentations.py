"""
YOLOP-style data augmentations for panoptic perception tasks.
Handles augmentation for images, segmentation masks, drivable area masks, and detection labels.
"""

import cv2
import numpy as np
import random
import math


def random_perspective(
    img, seg, drivable, labels,
    degrees=10, translate=0.1, scale=0.1, shear=10
):
    h, w = img.shape[:2]

    # --- MATRIX BUILDING ---
    # C: Center image at origin
    C = np.eye(3)
    C[0,2] = -w/2
    C[1,2] = -h/2

    # R: Rotation and scale
    a = random.uniform(-degrees, degrees)
    s = random.uniform(1-scale, 1+scale)
    R = np.eye(3)
    R[:2] = cv2.getRotationMatrix2D((0,0), a, s)

    # S: Shear
    S = np.eye(3)
    S[0,1] = math.tan(random.uniform(-shear, shear) * math.pi/180)
    S[1,0] = math.tan(random.uniform(-shear, shear) * math.pi/180)

    # T: Translate back to center + random offset
    # Formula: center (0.5) ± random translation
    T = np.eye(3)
    T[0,2] = random.uniform(0.5 - translate, 0.5 + translate) * w
    T[1,2] = random.uniform(0.5 - translate, 0.5 + translate) * h

    # Combined transformation: Translate to origin -> Rotate -> Shear -> Translate back
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

            # convert normalized → pixel
            x1 = (cx - bw/2) * w
            y1 = (cy - bh/2) * h
            x2 = (cx + bw/2) * w
            y2 = (cy + bh/2) * h

            corners = np.array([
                [x1, y1, 1],
                [x2, y1, 1],
                [x2, y2, 1],
                [x1, y2, 1]
            ])

            tc = corners @ M.T
            xs, ys = tc[:,0], tc[:,1]

            x1n, x2n = xs.min(), xs.max()
            y1n, y2n = ys.min(), ys.max()

            bw_n = (x2n - x1n) / w
            bh_n = (y2n - y1n) / h
            cx_n = (x1n + x2n) / 2 / w
            cy_n = (y1n + y2n) / 2 / h

            if bw_n > 0.002 and bh_n > 0.002:
                new_labels.append([cls, cx_n, cy_n, bw_n, bh_n])

        labels = np.array(new_labels) if new_labels else np.zeros((0,5))

    return img, seg, drivable, labels

def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    """
    Apply HSV color augmentation to image.

    Args:
        img: input image in BGR format (numpy array)
        hgain: hue gain (0-1), multiplied by random value
        sgain: saturation gain (0-1), multiplied by random value
        vgain: value/brightness gain (0-1), multiplied by random value
    """
    if hgain == 0 and sgain == 0 and vgain == 0:
        return

    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains

    # Convert BGR to HSV
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=r.dtype)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    # Apply LUT (Look-Up Table)
    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))

    # Convert back to BGR and update in-place
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)

def flip_horizontal(img, seg, drivable, labels):
    img = np.fliplr(img).copy()
    if seg is not None:
        seg = np.fliplr(seg).copy()
    if drivable is not None:
        drivable = np.fliplr(drivable).copy()
    if len(labels):
        labels[:,1] = 1 - labels[:,1]
    return img, seg, drivable, labels

def letterbox_with_masks(img, seg, drivable, labels, new_shape=(640,640), color=(114,114,114)):
    """
    Letterbox resize for image, segmentation mask, drivable mask, and bounding boxes.
    Maintains aspect ratio by padding.

    Args:
        img: input image (HWC)
        seg: segmentation mask (HW) or None
        drivable: drivable area mask (HW) or None
        labels: detection labels in normalized xywh format (N, 5) [class, cx, cy, w, h]
        new_shape: target size (height, width)
        color: padding color for image

    Returns:
        resized and padded img, seg, drivable, labels
    """
    h0, w0 = img.shape[:2]
    new_h, new_w = new_shape

    # Scale ratio
    r = min(new_h / h0, new_w / w0)
    new_unpad = (int(w0 * r), int(h0 * r))

    # Padding
    dw = new_w - new_unpad[0]
    dh = new_h - new_unpad[1]
    dw /= 2
    dh /= 2

    # Resize image
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    # Pad image
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    # Resize and pad segmentation mask
    if seg is not None:
        seg = cv2.resize(seg, new_unpad, interpolation=cv2.INTER_NEAREST)
        seg = cv2.copyMakeBorder(seg, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

    # Resize and pad drivable mask
    if drivable is not None:
        drivable = cv2.resize(drivable, new_unpad, interpolation=cv2.INTER_NEAREST)
        drivable = cv2.copyMakeBorder(drivable, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

    # Update labels (normalized xywh format)
    if len(labels):
        labels = labels.copy()

        # convert from normalized to absolute (original image size)
        labels[:,1] *= w0  # cx
        labels[:,2] *= h0  # cy
        labels[:,3] *= w0  # w
        labels[:,4] *= h0  # h

        # apply scale and pad
        labels[:,1] = labels[:,1] * r + left  # cx
        labels[:,2] = labels[:,2] * r + top   # cy
        labels[:,3] *= r  # w
        labels[:,4] *= r  # h

        # back to normalized (new image size)
        labels[:,1] /= new_w  # cx
        labels[:,2] /= new_h  # cy
        labels[:,3] /= new_w  # w
        labels[:,4] /= new_h  # h

    return img, seg, drivable, labels

def apply_salt_pepper(img, salt_prob=0.01, pepper_prob=0.01):
    """
    Apply salt and pepper noise to image.

    Args:
        img: input image in BGR format (numpy array, HWC, uint8)
        salt_prob: probability of salt noise (white pixels)
        pepper_prob: probability of pepper noise (black pixels)

    Returns:
        noisy image (same format as input)
    """
    if salt_prob == 0 and pepper_prob == 0:
        return img

    # Work with a copy to avoid modifying original
    noisy_img = img.copy()

    # Get image shape
    h, w, c = img.shape

    # Salt noise (set pixels to 255 - white)
    if salt_prob > 0:
        salt_mask = np.random.random((h, w)) < salt_prob
        noisy_img[salt_mask] = 255

    # Pepper noise (set pixels to 0 - black)
    if pepper_prob > 0:
        pepper_mask = np.random.random((h, w)) < pepper_prob
        noisy_img[pepper_mask] = 0

    return noisy_img    

def apply_augmentations(img, seg, drivable, labels, params):

    # -----------------------------------------
    # 1) GEOMETRIC AUG (on original size)
    # -----------------------------------------
    img, seg, drivable, labels = random_perspective(
        img, seg, drivable, labels,
        degrees=params.get("degrees", 10),
        translate=params.get("translate", 0.1),
        scale=params.get("scale", 0.1),
        shear=params.get("shear", 10)
    )

    # -----------------------------------------
    # 2) HSV
    # -----------------------------------------
    augment_hsv(img, params.get("hsv_h", 0.015), params.get("hsv_s", 0.7), params.get("hsv_v", 0.4))

    # -----------------------------------------
    # 3) SALT & PEPPER NOISE
    # -----------------------------------------
    salt_prob = params.get("salt_prob", 0.0)
    pepper_prob = params.get("pepper_prob", 0.0)
    if (salt_prob > 0 or pepper_prob > 0) and random.random() < 0.5:
        img = apply_salt_pepper(img, salt_prob=salt_prob, pepper_prob=pepper_prob)

    # -----------------------------------------
    # 4) HORIZONTAL FLIP
    # -----------------------------------------
    if random.random() < params.get("flip_prob", 0.5):
        img, seg, drivable, labels = flip_horizontal(img, seg, drivable, labels)

    # -----------------------------------------
    # 5) LETTERBOX RESIZE (after augmentations)
    # -----------------------------------------
    img_size = params.get("img_size", (640, 640))
    img, seg, drivable, labels = letterbox_with_masks(img, seg, drivable, labels, new_shape=img_size)

    return img, seg, drivable, labels
