"""
Mosaic augmentation for YOLOP-style training.
Combines 4 images to create diverse scenes with varied scales.
Particularly effective for small objects and long-tail classes.
"""

import cv2
import numpy as np
import random


def mosaic_augmentation(images_list, labels_list, segs_list, drivables_list, output_size=(640, 640)):
    """
    Create mosaic augmentation by combining 4 images.

    Args:
        images_list: list of 4 images (each HWC, BGR)
        labels_list: list of 4 label arrays (each Nx5: [class, cx, cy, w, h] normalized)
        segs_list: list of 4 segmentation masks or None
        drivables_list: list of 4 drivable masks or None
        output_size: output image size (height, width)

    Returns:
        mosaic_img, mosaic_labels, mosaic_seg, mosaic_drivable
    """
    assert len(images_list) == 4, "Mosaic requires exactly 4 images"

    h_out, w_out = output_size

    # Create output arrays
    mosaic_img = np.full((h_out, w_out, 3), 114, dtype=np.uint8)
    mosaic_labels = []

    # Initialize masks if provided
    has_seg = segs_list[0] is not None
    has_drivable = drivables_list[0] is not None

    if has_seg:
        mosaic_seg = np.zeros((h_out, w_out), dtype=np.uint8)
    else:
        mosaic_seg = None

    if has_drivable:
        mosaic_drivable = np.zeros((h_out, w_out), dtype=np.uint8)
    else:
        mosaic_drivable = None

    # Random center point for dividing the mosaic
    yc = int(random.uniform(0.4 * h_out, 0.6 * h_out))
    xc = int(random.uniform(0.4 * w_out, 0.6 * w_out))

    # Define placement regions for 4 images
    # [top-left, top-right, bottom-left, bottom-right]
    placements = [
        (0, 0, xc, yc),           # top-left
        (xc, 0, w_out, yc),       # top-right
        (0, yc, xc, h_out),       # bottom-left
        (xc, yc, w_out, h_out)    # bottom-right
    ]

    for idx, (img, labels, seg, drivable) in enumerate(zip(images_list, labels_list, segs_list, drivables_list)):
        h, w = img.shape[:2]

        # Get placement region
        x1_place, y1_place, x2_place, y2_place = placements[idx]
        place_h = y2_place - y1_place
        place_w = x2_place - x1_place

        # Resize image to fit placement region
        scale = min(place_w / w, place_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Calculate actual placement coordinates
        x1_actual = x1_place
        y1_actual = y1_place
        x2_actual = x1_actual + new_w
        y2_actual = y1_actual + new_h

        # Clip to output bounds
        x2_actual = min(x2_actual, w_out)
        y2_actual = min(y2_actual, h_out)

        # Place image
        mosaic_img[y1_actual:y2_actual, x1_actual:x2_actual] = img_resized[:y2_actual-y1_actual, :x2_actual-x1_actual]

        # Place segmentation mask
        if has_seg and seg is not None:
            seg_resized = cv2.resize(seg, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            mosaic_seg[y1_actual:y2_actual, x1_actual:x2_actual] = seg_resized[:y2_actual-y1_actual, :x2_actual-x1_actual]

        # Place drivable mask
        if has_drivable and drivable is not None:
            drivable_resized = cv2.resize(drivable, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            mosaic_drivable[y1_actual:y2_actual, x1_actual:x2_actual] = drivable_resized[:y2_actual-y1_actual, :x2_actual-x1_actual]

        # Transform labels
        if len(labels) > 0:
            labels_transformed = labels.copy()

            # Scale and translate bounding boxes
            # labels format: [class, cx, cy, w, h] (normalized 0-1)
            labels_transformed[:, 1] = labels[:, 1] * new_w / w_out + x1_actual / w_out  # cx
            labels_transformed[:, 2] = labels[:, 2] * new_h / h_out + y1_actual / h_out  # cy
            labels_transformed[:, 3] = labels[:, 3] * new_w / w_out  # w
            labels_transformed[:, 4] = labels[:, 4] * new_h / h_out  # h

            # Filter out boxes that are too small or out of bounds
            valid_boxes = []
            for box in labels_transformed:
                cx, cy, bw, bh = box[1], box[2], box[3], box[4]
                if 0 <= cx <= 1 and 0 <= cy <= 1 and bw > 0.001 and bh > 0.001:
                    valid_boxes.append(box)

            if valid_boxes:
                mosaic_labels.extend(valid_boxes)

    mosaic_labels = np.array(mosaic_labels) if mosaic_labels else np.zeros((0, 5))

    return mosaic_img, mosaic_labels, mosaic_seg, mosaic_drivable


def apply_mosaic_with_augmentations(dataset, indices, augment_params):
    """
    Helper function to apply mosaic + other augmentations.

    Args:
        dataset: BDD100KDataset instance
        indices: list of 4 indices to use for mosaic
        augment_params: augmentation parameters dict

    Returns:
        augmented image, labels, seg, drivable
    """
    from panoptic_perception.dataset.augmentations import apply_augmentations

    images_list = []
    labels_list = []
    segs_list = []
    drivables_list = []

    # Load 4 images without augmentation
    for idx in indices:
        sample = dataset[idx]
        # Note: This assumes dataset returns dict with 'image', 'detection_targets', etc.
        # You may need to modify based on your dataset structure
        images_list.append(sample['image'])
        labels_list.append(sample['detection_targets'])
        segs_list.append(sample['segmentation_mask'])
        drivables_list.append(sample['drivable_mask'])

    # Apply mosaic
    mosaic_img, mosaic_labels, mosaic_seg, mosaic_drivable = mosaic_augmentation(
        images_list, labels_list, segs_list, drivables_list,
        output_size=augment_params.get('img_size', (640, 640))
    )

    # Apply other augmentations on top of mosaic
    mosaic_img, mosaic_seg, mosaic_drivable, mosaic_labels = apply_augmentations(
        mosaic_img, mosaic_seg, mosaic_drivable, mosaic_labels, augment_params
    )

    return mosaic_img, mosaic_labels, mosaic_seg, mosaic_drivable
