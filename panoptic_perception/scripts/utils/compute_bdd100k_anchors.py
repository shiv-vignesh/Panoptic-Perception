"""
Compute custom anchor boxes for BDD100K dataset using K-means clustering.
This helps the YOLO model better match the object sizes in BDD100K.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.cluster import KMeans


def read_bdd100k_labels(labels_dir: str, dataset_type: str = "train"):
    """
    Read BDD100K detection labels and extract bounding box dimensions.

    Args:
        labels_dir: Path to BDD100K labels directory (e.g., 'bdd100k_labels/100k/train')
        dataset_type: 'train' or 'val'

    Returns:
        width_arr: numpy array of box widths
        height_arr: numpy array of box heights
    """
    objects_width = []
    objects_height = []

    # BDD100K has individual JSON files per image
    labels_path = os.path.join(labels_dir, dataset_type)

    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Could not find BDD100K labels at {labels_path}")

    print(f"Reading labels from: {labels_path}")

    # Get all JSON files
    json_files = [f for f in os.listdir(labels_path) if f.endswith('.json')]

    if len(json_files) == 0:
        raise FileNotFoundError(f"No JSON files found in {labels_path}")

    print(f"Found {len(json_files)} label files")

    # Process each JSON file
    for json_file in tqdm(json_files, desc=f"Processing {dataset_type} labels"):
        json_path = os.path.join(labels_path, json_file)

        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except:
            continue

        # Extract frames data
        if 'frames' not in data:
            continue

        for frame in data['frames']:
            if 'objects' not in frame:
                continue

            for obj in frame['objects']:
                if 'box2d' not in obj:
                    continue

                box = obj['box2d']

                # Extract coordinates
                x1 = float(box['x1'])
                y1 = float(box['y1'])
                x2 = float(box['x2'])
                y2 = float(box['y2'])

                # Compute width and height
                width = abs(x2 - x1)
                height = abs(y2 - y1)

                # Filter out invalid boxes
                if width > 0 and height > 0:
                    objects_width.append(width)
                    objects_height.append(height)

    width_arr = np.array(objects_width)
    height_arr = np.array(objects_height)

    print(f"  Found {len(width_arr)} valid bounding boxes")
    if len(width_arr) > 0:
        print(f"  Width  - min: {width_arr.min():.1f}, max: {width_arr.max():.1f}, mean: {width_arr.mean():.1f}")
        print(f"  Height - min: {height_arr.min():.1f}, max: {height_arr.max():.1f}, mean: {height_arr.mean():.1f}")

    return width_arr, height_arr


def compute_anchors(width_arr: np.array, height_arr: np.array,
                   img_width: int, img_height: int, yolo_input_size: int,
                   n_anchors: int = 9, save_path: str = None):
    """
    Compute anchor boxes using K-means clustering.

    Args:
        width_arr: Array of bounding box widths
        height_arr: Array of bounding box heights
        img_width: Original image width
        img_height: Original image height
        yolo_input_size: YOLO model input size (e.g., 640)
        n_anchors: Number of anchors to generate (default: 9 for 3 YOLO layers)
        save_path: Path to save visualization
    """
    # Stack width and height
    x = np.stack([width_arr, height_arr], axis=1)  # (N, 2)

    print(f"\nRunning K-Means clustering with {n_anchors} clusters...")

    # K-Means clustering to generate anchor boxes
    kmeans = KMeans(n_clusters=n_anchors, random_state=42, n_init=10)
    kmeans.fit(x)
    y_kmeans = kmeans.predict(x)

    # Compute average width/height for each cluster
    anchors = []
    for i in range(n_anchors):
        cluster_boxes = x[y_kmeans == i]
        if len(cluster_boxes) > 0:
            anchors.append(np.mean(cluster_boxes, axis=0))
        else:
            print(f"Warning: Cluster {i} is empty!")

    anchors = np.array(anchors)

    # Rescale anchors to YOLO input size
    # Original image: (img_width, img_height) -> YOLO input: (yolo_input_size, yolo_input_size)
    scaled_anchors = anchors.copy()
    scaled_anchors[:, 0] = (anchors[:, 0] / img_width) * yolo_input_size
    scaled_anchors[:, 1] = (anchors[:, 1] / img_height) * yolo_input_size
    scaled_anchors = np.rint(scaled_anchors).astype(int)

    # Sort anchors by area (small to large)
    areas = scaled_anchors[:, 0] * scaled_anchors[:, 1]
    sorted_indices = np.argsort(areas)
    scaled_anchors = scaled_anchors[sorted_indices]

    # Print results
    print("\n" + "="*60)
    print("COMPUTED ANCHOR BOXES (for YOLO config)")
    print("="*60)
    print(f"Image size: {img_width}x{img_height} -> YOLO input: {yolo_input_size}x{yolo_input_size}\n")

    # Format for YOLO config (3 anchors per layer)
    print("Format for yolop.cfg:")
    print("anchors=[", end="")

    for layer_idx in range(3):
        start_idx = layer_idx * 3
        end_idx = start_idx + 3
        layer_anchors = scaled_anchors[start_idx:end_idx]

        print("[", end="")
        anchor_list = []
        for anchor in layer_anchors:
            anchor_list.extend([int(anchor[0]), int(anchor[1])])
        print(",".join(map(str, anchor_list)), end="")
        print("]", end="")

        if layer_idx < 2:
            print(", ", end="")

    print("], [128, 256, 512]")
    print()

    # Print individual layer anchors
    for layer_idx in range(3):
        start_idx = layer_idx * 3
        end_idx = start_idx + 3
        layer_anchors = scaled_anchors[start_idx:end_idx]
        layer_name = ["Small", "Medium", "Large"][layer_idx]
        print(f"Layer {layer_idx} ({layer_name} objects):")
        for i, anchor in enumerate(layer_anchors):
            area = anchor[0] * anchor[1]
            print(f"  Anchor {i+1}: {anchor[0]:3d}x{anchor[1]:3d} (area: {area:5d})")
        print()

    print("="*60)

    # Visualize anchors
    if save_path:
        fig, ax = plt.subplots(figsize=(10, 10))

        colors = ['red', 'green', 'blue']
        layer_names = ['Small (P3)', 'Medium (P4)', 'Large (P5)']

        for layer_idx in range(3):
            start_idx = layer_idx * 3
            end_idx = start_idx + 3
            layer_anchors = scaled_anchors[start_idx:end_idx]

            for anchor in layer_anchors:
                rect = plt.Rectangle(
                    (yolo_input_size/2 - anchor[0]/2, yolo_input_size/2 - anchor[1]/2),
                    anchor[0], anchor[1],
                    edgecolor=colors[layer_idx],
                    facecolor='none',
                    linewidth=2,
                    label=layer_names[layer_idx] if anchor is layer_anchors[0] else ""
                )
                ax.add_patch(rect)

        # Add grid
        ax.grid(True, alpha=0.3)
        ax.set_aspect(1.0)
        ax.set_xlim([0, yolo_input_size])
        ax.set_ylim([0, yolo_input_size])
        ax.set_xlabel('Width (pixels)', fontsize=12)
        ax.set_ylabel('Height (pixels)', fontsize=12)
        ax.set_title(f'BDD100K Anchor Boxes for YOLO ({yolo_input_size}x{yolo_input_size})', fontsize=14)

        # Add legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ Visualization saved to: {save_path}")

    return scaled_anchors


def main():
    """Main function to compute BDD100K anchors."""

    # ===== BDD100K Configuration =====
    IMG_WIDTH = 1280   # BDD100K original image width
    IMG_HEIGHT = 720   # BDD100K original image height
    YOLO_INPUT_SIZE = 640  # Your YOLO model input size

    # Paths (adjust these to your setup)
    LABELS_DIR = "panoptic_perception/BDD100k/bdd100k_labels/100k"

    OUTPUT_DIR = "panoptic_perception/configs/anchors"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    PLT_SAVE_PATH = f"{OUTPUT_DIR}/bdd100k_anchors_{YOLO_INPUT_SIZE}.png"
    ANCHORS_TXT = f"{OUTPUT_DIR}/bdd100k_anchors_{YOLO_INPUT_SIZE}.txt"

    print("="*60)
    print("BDD100K Anchor Box Computation")
    print("="*60)
    print(f"Original image size: {IMG_WIDTH}x{IMG_HEIGHT}")
    print(f"YOLO input size: {YOLO_INPUT_SIZE}x{YOLO_INPUT_SIZE}")
    print(f"Labels directory: {LABELS_DIR}")
    print("="*60)
    print()

    # Read training labels
    print("📊 Reading training labels...")
    train_width, train_height = read_bdd100k_labels(LABELS_DIR, dataset_type="train")

    # Read validation labels
    print("\n📊 Reading validation labels...")
    val_width, val_height = read_bdd100k_labels(LABELS_DIR, dataset_type="val")

    # Combine train and val
    print("\n📊 Combining train and validation data...")
    all_width = np.concatenate([train_width, val_width])
    all_height = np.concatenate([train_height, val_height])

    print(f"  Total boxes: {len(all_width)}")
    print(f"  Width  - min: {all_width.min():.1f}, max: {all_width.max():.1f}, mean: {all_width.mean():.1f}")
    print(f"  Height - min: {all_height.min():.1f}, max: {all_height.max():.1f}, mean: {all_height.mean():.1f}")

    # Compute anchors
    anchors = compute_anchors(
        all_width, all_height,
        img_width=IMG_WIDTH,
        img_height=IMG_HEIGHT,
        yolo_input_size=YOLO_INPUT_SIZE,
        n_anchors=9,
        save_path=PLT_SAVE_PATH
    )

    # Save anchors to text file
    with open(ANCHORS_TXT, "w") as f:
        f.write("# BDD100K Custom Anchors\n")
        f.write(f"# Image size: {IMG_WIDTH}x{IMG_HEIGHT}\n")
        f.write(f"# YOLO input: {YOLO_INPUT_SIZE}x{YOLO_INPUT_SIZE}\n")
        f.write(f"# Total boxes analyzed: {len(all_width)}\n\n")

        # Write formatted for yolop.cfg
        f.write("anchors=[")
        for layer_idx in range(3):
            start_idx = layer_idx * 3
            end_idx = start_idx + 3
            layer_anchors = anchors[start_idx:end_idx]

            f.write("[")
            anchor_list = []
            for anchor in layer_anchors:
                anchor_list.extend([int(anchor[0]), int(anchor[1])])
            f.write(",".join(map(str, anchor_list)))
            f.write("]")

            if layer_idx < 2:
                f.write(", ")
        f.write("], [128, 256, 512]\n\n")

        # Write raw anchor values
        f.write("# Raw anchor values (width, height):\n")
        f.write(str(anchors.tolist()))

    print(f"✅ Anchors saved to: {ANCHORS_TXT}")
    print("\n🎯 Next steps:")
    print("  1. Copy the 'anchors=' line above")
    print("  2. Replace line 179 in panoptic_perception/configs/models/yolop.cfg")
    print("  3. Restart training to use new anchors")
    print()


if __name__ == "__main__":
    main()
