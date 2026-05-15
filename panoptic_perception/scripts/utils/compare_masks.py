"""
Compare BDD100K source drivable masks vs polygon-rendered masks.
Checks alignment, resolution, and boundary differences.
"""

import cv2
import numpy as np
import os

# Paths
IMAGES_DIR = "panoptic_perception/BDD100k/100k/100k/train"
SOURCE_MASKS_DIR = "panoptic_perception/BDD100k/bdd100k_drivable_maps/labels/train"
POLY_MASKS_DIR = "panoptic_perception/BDD100k/bdd100k_drivable_polygonRender/train"
POLY_VISUAL_DIR = "panoptic_perception/BDD100k/bdd100k_drivable_polygonRender_visual/train"
OUTPUT_DIR = "panoptic_perception/BDD100k/mask_comparison"

NUM_SAMPLES = 10


def create_overlay(image, mask, color=(0, 255, 0)):
    """Overlay mask on image with given color."""
    overlay = image.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(image, 0.6, overlay, 0.4, 0)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Get samples from poly masks that are already generated
    poly_files = [f for f in os.listdir(POLY_MASKS_DIR) if f.endswith('.png')][:NUM_SAMPLES]
    image_files = [f.replace('_drivable_id.png', '.jpg') for f in poly_files]

    for img_file in image_files:
        img_name = img_file.replace('.jpg', '')

        # Load image
        img_path = os.path.join(IMAGES_DIR, img_file)
        image = cv2.imread(img_path)
        if image is None:
            print(f"Could not load image: {img_path}")
            continue

        h, w = image.shape[:2]
        print(f"\n{img_name}: Image size = {w}x{h}")

        # Load source mask (from BDD100K)
        source_path = os.path.join(SOURCE_MASKS_DIR, f"{img_name}_drivable_id.png")
        source_mask = cv2.imread(source_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(source_path) else None

        # Load polygon-rendered mask
        poly_path = os.path.join(POLY_MASKS_DIR, f"{img_name}_drivable_id.png")
        poly_mask = cv2.imread(poly_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(poly_path) else None

        # Load polygon visual mask (0/255)
        poly_vis_path = os.path.join(POLY_VISUAL_DIR, f"{img_name}_drivable_id.png")
        poly_vis_mask = cv2.imread(poly_vis_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(poly_vis_path) else None

        # Print mask info
        if source_mask is not None:
            print(f"  Source mask: {source_mask.shape}, unique={np.unique(source_mask)}")
        else:
            print(f"  Source mask: NOT FOUND")

        if poly_mask is not None:
            print(f"  Poly mask:   {poly_mask.shape}, unique={np.unique(poly_mask)}")
        else:
            print(f"  Poly mask:   NOT FOUND")

        # Create comparison visualization
        rows = []

        # Row 1: Original image
        row1 = image.copy()
        cv2.putText(row1, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Row 2: Source mask overlay (green) - merge class 2 to 1
        if source_mask is not None:
            source_binary = source_mask.copy()
            source_binary[source_binary == 2] = 1  # Merge alternative
            overlay_source = create_overlay(image, source_binary, (0, 255, 0))
            cv2.putText(overlay_source, "Source (BDD100K)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            overlay_source = np.zeros_like(image)
            cv2.putText(overlay_source, "Source: N/A", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Row 3: Poly mask overlay (blue)
        if poly_mask is not None:
            overlay_poly = create_overlay(image, poly_mask, (255, 0, 0))
            cv2.putText(overlay_poly, "PolyRender", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            overlay_poly = np.zeros_like(image)
            cv2.putText(overlay_poly, "Poly: N/A", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Row 4: Difference (where they disagree)
        if source_mask is not None and poly_mask is not None:
            source_binary = source_mask.copy()
            source_binary[source_binary == 2] = 1

            # Resize poly_mask if needed
            if poly_mask.shape != source_binary.shape:
                poly_resized = cv2.resize(poly_mask, (source_binary.shape[1], source_binary.shape[0]),
                                          interpolation=cv2.INTER_NEAREST)
                print(f"  Resized poly mask from {poly_mask.shape} to {poly_resized.shape}")
            else:
                poly_resized = poly_mask

            diff = np.zeros_like(image)
            # Green: only in source, Red: only in poly, Black: both or neither
            only_source = (source_binary > 0) & (poly_resized == 0)
            only_poly = (source_binary == 0) & (poly_resized > 0)
            diff[only_source] = [0, 255, 0]   # Green = source only
            diff[only_poly] = [0, 0, 255]     # Red = poly only

            # Calculate IoU
            intersection = ((source_binary > 0) & (poly_resized > 0)).sum()
            union = ((source_binary > 0) | (poly_resized > 0)).sum()
            iou = intersection / union if union > 0 else 0
            print(f"  IoU (source vs poly): {iou:.4f}")
            cv2.putText(diff, f"Diff (IoU={iou:.3f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(diff, "Green=Source only, Red=Poly only", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        else:
            diff = np.zeros_like(image)
            cv2.putText(diff, "Diff: N/A", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Stack into 2x2 grid
        top_row = np.hstack([row1, overlay_source])
        bottom_row = np.hstack([overlay_poly, diff])
        comparison = np.vstack([top_row, bottom_row])

        # Save
        out_path = os.path.join(OUTPUT_DIR, f"{img_name}_comparison.jpg")
        cv2.imwrite(out_path, comparison)
        print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
