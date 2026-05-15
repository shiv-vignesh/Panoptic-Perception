"""
Generate drivable area segmentation masks from BDD100K polygon annotations.
Renders polygons to binary masks (0=background, 1=drivable).
"""

from matplotlib.path import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
import os
import json
import numpy as np
from tqdm import tqdm


# Constants
DPI = 80
FIG_W, FIG_H = 16, 9  # 16*80=1280, 9*80=720
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720


def poly2patch(poly2d, closed=False, alpha=1., color=None):
    """Convert BDD100K poly2d format to matplotlib patch."""
    moves = {'L': Path.LINETO,
             'C': Path.CURVE4}
    points = [p[:2] for p in poly2d]
    codes = [moves[p[2]] for p in poly2d]
    codes[0] = Path.MOVETO

    if closed:
        points.append(points[0])
        codes.append(Path.CLOSEPOLY)

    return mpatches.PathPatch(
        Path(points, codes),
        facecolor=color if closed else 'none',
        edgecolor=color,
        lw=1 if closed else 2,
        alpha=alpha,
        antialiased=False,
        snap=True)


def get_drivable_areas(objects):
    """Filter objects to only drivable area polygons."""
    return [o for o in objects
            if 'poly2d' in o and o['category'].startswith('area')]


def has_drivable_areas(data):
    """Check if annotation has any drivable area polygons."""
    return any(obj['category'].startswith('area') for obj in data)


def draw_drivable(objects, ax, merge_alternative=True):
    """Draw drivable area polygons onto matplotlib axes."""
    objects = get_drivable_areas(objects)

    for obj in objects:
        if obj['category'] == 'area/drivable':
            color = (1, 1, 1)  # White = drivable
        elif obj['category'] == 'area/alternative':
            color = (1, 1, 1) if merge_alternative else (0, 0, 0)
        else:
            print(f"Unknown area category: {obj['category']}")
            color = (0, 0, 0)

        poly2d = obj['poly2d']
        ax.add_patch(poly2patch(poly2d, closed=True, alpha=1.0, color=color))

    ax.axis('off')


def render_mask(data, out_path, visual_path, merge_alternative=True):
    """Render drivable area mask and save both visual and training versions."""
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], frameon=False)

    ax.set_xlim(0, IMAGE_WIDTH - 1)
    ax.set_ylim(0, IMAGE_HEIGHT - 1)
    ax.invert_yaxis()

    # Draw black background
    ax.add_patch(poly2patch(
        [[0, 0, 'L'], [0, IMAGE_HEIGHT - 1, 'L'],
         [IMAGE_WIDTH - 1, IMAGE_HEIGHT - 1, 'L'],
         [IMAGE_WIDTH - 1, 0, 'L']],
        closed=True, alpha=1., color=(0, 0, 0)))

    # Draw drivable areas if present
    if has_drivable_areas(data):
        draw_drivable(data, ax, merge_alternative)

    fig.savefig(out_path, dpi=DPI)
    plt.close()

    mask = cv2.imread(out_path, cv2.IMREAD_GRAYSCALE)

    # Save visual version (0/255)
    cv2.imwrite(visual_path, mask)

    # Save training version (0/1)
    mask = (mask > 0).astype(np.uint8)
    cv2.imwrite(out_path, mask)


def main(mode, labels_dir, output_dir, visual_dir, merge_alternative=True):
    """Generate drivable masks for all images in the dataset."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(visual_dir, exist_ok=True)

    label_files = os.listdir(labels_dir)

    for label_json_fn in tqdm(label_files, desc=f"Generating {mode} masks"):
        label_path = os.path.join(labels_dir, label_json_fn)

        with open(label_path, 'r') as f:
            label_json = json.load(f)

        data = label_json["frames"][0]["objects"]
        img_name = label_json["name"]
        out_path = os.path.join(output_dir, f"{img_name}_drivable_id.png")
        visual_path = os.path.join(visual_dir, f"{img_name}_drivable_id.png")

        render_mask(data, out_path, visual_path, merge_alternative)

    print(f"Generated {len(label_files)} masks in {output_dir}")


if __name__ == "__main__":
    # Config
    MERGE_ALTERNATIVE = True  # True: alternative→drivable, False: alternative→background

    base_labels = "panoptic_perception/BDD100k/bdd100k_labels/100k"
    base_output = "panoptic_perception/BDD100k/bdd100k_drivable_polygonRender"
    base_visual = "panoptic_perception/BDD100k/bdd100k_drivable_polygonRender_visual"

    for mode in ["train", "val"]:
        labels_dir = f"{base_labels}/{mode}"
        output_dir = f"{base_output}/{mode}"
        visual_dir = f"{base_visual}/{mode}"
        main(mode, labels_dir, output_dir, visual_dir, MERGE_ALTERNATIVE)
    