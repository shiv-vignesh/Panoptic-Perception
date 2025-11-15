"""
Script to check file correspondence between images and annotations in BDD100K dataset.

Checks how many images from 10k/train have corresponding:
- Detection annotations (JSON files or entries in det_train.json)
- Segmentation masks
- Drivable area masks
"""

import os
import sys
import json
from pathlib import Path
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def get_image_ids(images_dir):
    """Get all image IDs from the images directory."""
    if not os.path.exists(images_dir):
        print(f"✗ Images directory does not exist: {images_dir}")
        return []

    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg') or f.endswith('.png')]
    image_ids = [f.rsplit('.', 1)[0] for f in image_files]

    return sorted(image_ids)


def check_detection_annotations(image_ids, det_dir, dataset_type='train'):
    """
    Check how many images have detection annotations.

    Supports both:
    - Single JSON file: det_train.json
    - Individual JSON files per image
    """
    matches = []
    missing = []

    for image_id in image_ids:
        # Check for individual JSON file
        det_path = os.path.join(det_dir, f"{image_id}.json")
        if os.path.exists(det_path):
            matches.append(image_id)
            continue

        # If neither found, mark as missing
        missing.append(image_id)

    return matches, missing


def check_mask_annotations(image_ids, mask_dir, suffix='_train_id.png'):
    """Check how many images have corresponding mask files."""
    matches = []
    missing = []

    masks_path = os.path.join(mask_dir)

    if not os.path.exists(masks_path):
        print(f"  ✗ Mask directory does not exist: {masks_path}")
        return matches, missing

    # Try different suffix patterns
    suffixes = [suffix, '_train_id.png', '_val_id.png', '_drivable_id.png', '.png']

    for image_id in image_ids:
        mask_path = os.path.join(masks_path, f"{image_id}{suffix}")
        if os.path.exists(mask_path):
            matches.append(image_id)                
        else:
            missing.append(image_id)
            
    return matches, missing


def print_statistics(label, total, matches, missing):
    """Print statistics in a formatted way."""
    match_count = len(matches)
    missing_count = len(missing)
    match_pct = (match_count / total * 100) if total > 0 else 0

    print(f"\n{label}:")
    print(f"  Total images:    {total:>6}")
    print(f"  Matches:         {match_count:>6} ({match_pct:.1f}%)")
    print(f"  Missing:         {missing_count:>6}")

    if missing_count > 0 and missing_count <= 5:
        print(f"  Missing IDs:     {missing[:5]}")


def main():
    print("\n" + "="*80)
    print("BDD100K FILE CORRESPONDENCE CHECKER")
    print("="*80)

    # ========================================================================
    # CONFIGURATION - UPDATE THESE PATHS
    # ========================================================================

    base_path = "panoptic_perception/BDD100K"  # UPDATE THIS
    dataset_type = "train"  # or "val"

    images_dir = os.path.join(base_path, "100k/100k", dataset_type)
    det_dir = os.path.join(base_path, "bdd100k_labels/100k", dataset_type)
    seg_dir = os.path.join(base_path, "bdd100k_seg_maps/labels", dataset_type)
    drivable_dir = os.path.join(base_path, "bdd100k_drivable_maps/labels", dataset_type)

    print(f"\nBase path: {base_path}")
    print(f"Dataset type: {dataset_type}")
    print(f"\nChecking directories:")
    print(f"  Images:      {images_dir}")
    print(f"  Detection:   {det_dir}")
    print(f"  Segmentation: {seg_dir}")
    print(f"  Drivable:    {drivable_dir}")

    # ========================================================================
    # GET IMAGE IDS
    # ========================================================================

    print("\n" + "-"*80)
    print("STEP 1: Getting image IDs from images directory")
    print("-"*80)

    image_ids = get_image_ids(images_dir)

    if not image_ids:
        print("✗ No images found. Please check the path.")
        return

    print(f"✓ Found {len(image_ids)} images")
    print(f"  First 5 IDs: {image_ids[:5]}")
    print(f"  Last 5 IDs:  {image_ids[-5:]}")

    total_images = len(image_ids)

    # ========================================================================
    # CHECK DETECTION ANNOTATIONS
    # ========================================================================

    print("\n" + "-"*80)
    print("STEP 2: Checking detection annotations")
    print("-"*80)

    det_matches, det_missing = check_detection_annotations(image_ids, det_dir)
    print_statistics("Detection Annotations", total_images, det_matches, det_missing)

    # ========================================================================
    # CHECK SEGMENTATION MASKS
    # ========================================================================

    print("\n" + "-"*80)
    print("STEP 3: Checking segmentation masks")
    print("-"*80)

    seg_suffix = f"_{dataset_type}_id.png"
    seg_matches, seg_missing = check_mask_annotations(image_ids, seg_dir, seg_suffix)
    print_statistics("Segmentation Masks", total_images, seg_matches, seg_missing)

    # ========================================================================
    # CHECK DRIVABLE MASKS
    # ========================================================================

    print("\n" + "-"*80)
    print("STEP 4: Checking drivable area masks")
    print("-"*80)

    drivable_suffix = "_drivable_id.png"
    drivable_matches, drivable_missing = check_mask_annotations(
        image_ids, drivable_dir, drivable_suffix
    )
    print_statistics("Drivable Area Masks", total_images, drivable_matches, drivable_missing)

    # ========================================================================
    # SUMMARY
    # ========================================================================

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    print(f"\nTotal images: {total_images}")
    print(f"\nAnnotation coverage:")
    print(f"  Detection:      {len(det_matches):>6} / {total_images} ({len(det_matches)/total_images*100:.1f}%)")
    print(f"  Segmentation:   {len(seg_matches):>6} / {total_images} ({len(seg_matches)/total_images*100:.1f}%)")
    print(f"  Drivable:       {len(drivable_matches):>6} / {total_images} ({len(drivable_matches)/total_images*100:.1f}%)")

    # Find images with all annotations
    det_set = set(det_matches)
    seg_set = set(seg_matches)
    drivable_set = set(drivable_matches)

    all_annotations = det_set & seg_set & drivable_set
    print(f"\nImages with ALL annotations: {len(all_annotations)} ({len(all_annotations)/total_images*100:.1f}%)")

    # Find images with at least one annotation
    any_annotation = det_set | seg_set | drivable_set
    print(f"Images with ANY annotation:  {len(any_annotation)} ({len(any_annotation)/total_images*100:.1f}%)")

    # Find images with no annotations
    no_annotations = set(image_ids) - any_annotation
    if no_annotations:
        print(f"\n⚠ Images with NO annotations: {len(no_annotations)}")
        if len(no_annotations) <= 10:
            print(f"  IDs: {sorted(no_annotations)[:10]}")

    # ========================================================================
    # SAVE REPORT
    # ========================================================================

    print("\n" + "-"*80)

    report_path = "bdd100k_match_report.txt"
    with open(report_path, 'w') as f:
        f.write("BDD100K FILE CORRESPONDENCE REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Base path: {base_path}\n")
        f.write(f"Dataset type: {dataset_type}\n")
        f.write(f"Total images: {total_images}\n\n")

        f.write("MATCHES:\n")
        f.write(f"  Detection:      {len(det_matches):>6} ({len(det_matches)/total_images*100:.1f}%)\n")
        f.write(f"  Segmentation:   {len(seg_matches):>6} ({len(seg_matches)/total_images*100:.1f}%)\n")
        f.write(f"  Drivable:       {len(drivable_matches):>6} ({len(drivable_matches)/total_images*100:.1f}%)\n\n")

        f.write(f"Complete coverage: {len(all_annotations)} images\n\n")

        if det_missing:
            f.write(f"\nMISSING DETECTION ({len(det_missing)}):\n")
            for img_id in det_missing[:20]:
                f.write(f"  {img_id}\n")

        if seg_missing:
            f.write(f"\nMISSING SEGMENTATION ({len(seg_missing)}):\n")
            for img_id in seg_missing[:20]:
                f.write(f"  {img_id}\n")

        if drivable_missing:
            f.write(f"\nMISSING DRIVABLE ({len(drivable_missing)}):\n")
            for img_id in drivable_missing[:20]:
                f.write(f"  {img_id}\n")

    print(f"✓ Report saved to: {report_path}")

    print("\n" + "="*80)
    print("CHECK COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
