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


def read_bdd100k_labels(labels_dir: str, dataset_type: str = "train",
                        target_classes: list = None):
    """
    Read BDD100K detection labels and extract bounding box dimensions.

    Args:
        labels_dir: Path to BDD100K labels directory (e.g., 'bdd100k_labels/100k/train')
        dataset_type: 'train' or 'val'
        target_classes: List of class names to include. If None, includes all classes.
                       YOLOP uses: ['car', 'bus', 'truck', 'person', 'rider', 'bike',
                                    'motor', 'train', 'traffic light', 'traffic sign']

    Returns:
        width_arr: numpy array of box widths
        height_arr: numpy array of box heights
    """
    objects_width = []
    objects_height = []
    class_counts = {}

    # BDD100K has individual JSON files per image
    labels_path = os.path.join(labels_dir, dataset_type)

    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Could not find BDD100K labels at {labels_path}")

    print(f"Reading labels from: {labels_path}")
    if target_classes:
        print(f"  Filtering to classes: {target_classes}")

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

                # Filter by class if specified
                obj_class = obj.get('category', '').lower()
                if target_classes:
                    if obj_class not in [c.lower() for c in target_classes]:
                        continue

                # Track class distribution
                class_counts[obj_class] = class_counts.get(obj_class, 0) + 1

                box = obj['box2d']

                # Extract coordinates
                x1 = float(box['x1'])
                y1 = float(box['y1'])
                x2 = float(box['x2'])
                y2 = float(box['y2'])

                # Compute width and height
                width = abs(x2 - x1)
                height = abs(y2 - y1)

                # Filter out invalid boxes (too small)
                if width > 1 and height > 1:
                    objects_width.append(width)
                    objects_height.append(height)

    width_arr = np.array(objects_width)
    height_arr = np.array(objects_height)

    print(f"  Found {len(width_arr)} valid bounding boxes")
    if len(width_arr) > 0:
        print(f"  Width  - min: {width_arr.min():.1f}, max: {width_arr.max():.1f}, mean: {width_arr.mean():.1f}")
        print(f"  Height - min: {height_arr.min():.1f}, max: {height_arr.max():.1f}, mean: {height_arr.mean():.1f}")

    # Print class distribution
    if class_counts:
        print(f"  Class distribution:")
        for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
            print(f"    {cls}: {count}")

    return width_arr, height_arr


def iou_distance(boxes: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """
    Compute 1 - IoU between boxes and anchors (for IoU-based k-means).
    Boxes and anchors are (width, height) pairs centered at origin.

    Args:
        boxes: (N, 2) array of [width, height]
        anchors: (K, 2) array of [width, height]

    Returns:
        (N, K) distance matrix where distance = 1 - IoU
    """
    N = boxes.shape[0]
    K = anchors.shape[0]

    # Expand dims for broadcasting: boxes (N,1,2), anchors (1,K,2)
    boxes_exp = boxes[:, np.newaxis, :]  # (N, 1, 2)
    anchors_exp = anchors[np.newaxis, :, :]  # (1, K, 2)

    # Intersection (min of w, min of h since centered)
    inter_w = np.minimum(boxes_exp[:, :, 0], anchors_exp[:, :, 0])
    inter_h = np.minimum(boxes_exp[:, :, 1], anchors_exp[:, :, 1])
    inter_area = inter_w * inter_h  # (N, K)

    # Union
    box_area = boxes[:, 0] * boxes[:, 1]  # (N,)
    anchor_area = anchors[:, 0] * anchors[:, 1]  # (K,)
    union_area = box_area[:, np.newaxis] + anchor_area[np.newaxis, :] - inter_area  # (N, K)

    iou = inter_area / (union_area + 1e-9)
    return 1.0 - iou  # distance = 1 - IoU


def kmeans_iou(boxes: np.ndarray, n_clusters: int = 9, max_iter: int = 300) -> np.ndarray:
    """
    K-means clustering using IoU distance metric (like YOLOv5 autoanchor).

    Args:
        boxes: (N, 2) array of [width, height]
        n_clusters: Number of anchor clusters
        max_iter: Maximum iterations

    Returns:
        (n_clusters, 2) array of anchor [width, height]
    """
    N = boxes.shape[0]

    # Initialize anchors using k-means++ style (random sample)
    np.random.seed(42)
    indices = np.random.choice(N, n_clusters, replace=False)
    anchors = boxes[indices].copy()

    for iteration in range(max_iter):
        # Assign each box to nearest anchor (by IoU distance)
        distances = iou_distance(boxes, anchors)  # (N, K)
        assignments = np.argmin(distances, axis=1)  # (N,)

        # Update anchors as median of assigned boxes (more robust than mean)
        new_anchors = np.zeros_like(anchors)
        for k in range(n_clusters):
            mask = assignments == k
            if mask.sum() > 0:
                # Use median for robustness to outliers
                new_anchors[k] = np.median(boxes[mask], axis=0)
            else:
                # Keep old anchor if cluster is empty
                new_anchors[k] = anchors[k]

        # Check convergence
        if np.allclose(anchors, new_anchors, atol=0.1):
            print(f"  Converged at iteration {iteration + 1}")
            break

        anchors = new_anchors

    return anchors


def compute_anchors(width_arr: np.array, height_arr: np.array,
                   img_width: int, img_height: int,
                   yolo_input_width: int, yolo_input_height: int,
                   n_anchors: int = 9, save_path: str = None,
                   use_iou_kmeans: bool = True):
    """
    Compute anchor boxes using K-means clustering.

    Args:
        width_arr: Array of bounding box widths
        height_arr: Array of bounding box heights
        img_width: Original image width (1280 for BDD100K)
        img_height: Original image height (720 for BDD100K)
        yolo_input_width: YOLO model input width (e.g., 640)
        yolo_input_height: YOLO model input height (e.g., 384 for YOLOP, 640 for square)
        n_anchors: Number of anchors to generate (default: 9 for 3 YOLO layers)
        save_path: Path to save visualization
        use_iou_kmeans: Use IoU-based k-means (recommended) vs Euclidean k-means
    """
    # Stack width and height
    x = np.stack([width_arr, height_arr], axis=1)  # (N, 2)

    print(f"\nClustering {len(x)} boxes into {n_anchors} anchors...")
    print(f"  Method: {'IoU-based K-means' if use_iou_kmeans else 'Euclidean K-means'}")

    if use_iou_kmeans:
        # IoU-based k-means (like YOLOv5 autoanchor)
        anchors = kmeans_iou(x, n_clusters=n_anchors)
    else:
        # Standard Euclidean k-means
        kmeans = KMeans(n_clusters=n_anchors, random_state=42, n_init=10)
        kmeans.fit(x)
        anchors = kmeans.cluster_centers_  # Use centroids directly

    # Rescale anchors to YOLO input size
    # Original image: (img_width, img_height) -> YOLO input: (yolo_input_width, yolo_input_height)
    # IMPORTANT: Scale width and height separately!
    scaled_anchors = anchors.copy()
    scaled_anchors[:, 0] = (anchors[:, 0] / img_width) * yolo_input_width    # width scaling
    scaled_anchors[:, 1] = (anchors[:, 1] / img_height) * yolo_input_height  # height scaling
    scaled_anchors = np.rint(scaled_anchors).astype(int)

    # Ensure minimum anchor size of 2 pixels
    scaled_anchors = np.maximum(scaled_anchors, 2)

    # Sort anchors by area (small to large)
    areas = scaled_anchors[:, 0] * scaled_anchors[:, 1]
    sorted_indices = np.argsort(areas)
    scaled_anchors = scaled_anchors[sorted_indices]

    # Print results
    print("\n" + "="*60)
    print("COMPUTED ANCHOR BOXES (for YOLO config)")
    print("="*60)
    print(f"Image size: {img_width}x{img_height} -> YOLO input: {yolo_input_width}x{yolo_input_height}\n")

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
        fig, ax = plt.subplots(figsize=(10, 8))

        colors = ['red', 'green', 'blue']
        layer_names = ['Small (P3)', 'Medium (P4)', 'Large (P5)']

        for layer_idx in range(3):
            start_idx = layer_idx * 3
            end_idx = start_idx + 3
            layer_anchors = scaled_anchors[start_idx:end_idx]

            for anchor in layer_anchors:
                rect = plt.Rectangle(
                    (yolo_input_width/2 - anchor[0]/2, yolo_input_height/2 - anchor[1]/2),
                    anchor[0], anchor[1],
                    edgecolor=colors[layer_idx],
                    facecolor='none',
                    linewidth=2,
                    label=layer_names[layer_idx] if anchor is layer_anchors[0] else ""
                )
                ax.add_patch(rect)

        # Add grid
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.set_xlim([0, yolo_input_width])
        ax.set_ylim([0, yolo_input_height])
        ax.set_xlabel('Width (pixels)', fontsize=12)
        ax.set_ylabel('Height (pixels)', fontsize=12)
        ax.set_title(f'BDD100K Anchor Boxes for YOLO ({yolo_input_width}x{yolo_input_height})', fontsize=14)

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

    import argparse

    parser = argparse.ArgumentParser(description="Compute BDD100K anchor boxes using K-means")
    parser.add_argument("--input-width", type=int, default=640,
                        help="YOLO input width (default: 640)")
    parser.add_argument("--input-height", type=int, default=640,
                        help="YOLO input height (default: 640, YOLOP uses 384)")
    parser.add_argument("--use-yolop-size", action="store_true",
                        help="Use YOLOP's 640x384 input size")
    parser.add_argument("--euclidean", action="store_true",
                        help="Use Euclidean K-means instead of IoU-based")
    parser.add_argument("--all-classes", action="store_true",
                        help="Include all classes (default: filter to detection classes)")
    args = parser.parse_args()

    # ===== BDD100K Configuration =====
    IMG_WIDTH = 1280   # BDD100K original image width
    IMG_HEIGHT = 720   # BDD100K original image height

    # YOLO input size
    if args.use_yolop_size:
        YOLO_INPUT_WIDTH = 640
        YOLO_INPUT_HEIGHT = 384  # YOLOP's aspect ratio
    else:
        YOLO_INPUT_WIDTH = args.input_width
        YOLO_INPUT_HEIGHT = args.input_height

    # Target classes (YOLOP uses these 10 classes)
    # Your model uses 6: car/bus/truck -> VEHICLES, person, rider, motor, traffic_light, traffic_sign
    TARGET_CLASSES = None if args.all_classes else [
        'car', 'bus', 'truck',       # -> VEHICLES
        'person',                     # -> PERSON
        'rider',                      # -> RIDER
        'motor', 'bike',              # -> MOTOR (motorcycle + bicycle)
        'traffic light',              # -> TRAFFIC_LIGHT
        'traffic sign'                # -> TRAFFIC_SIGN
    ]

    # Paths (adjust these to your setup)
    LABELS_DIR = "panoptic_perception/BDD100k/bdd100k_labels/100k"

    OUTPUT_DIR = "panoptic_perception/configs/anchors"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    size_str = f"{YOLO_INPUT_WIDTH}x{YOLO_INPUT_HEIGHT}"
    PLT_SAVE_PATH = f"{OUTPUT_DIR}/bdd100k_anchors_{size_str}.png"
    ANCHORS_TXT = f"{OUTPUT_DIR}/bdd100k_anchors_{size_str}.txt"

    print("="*60)
    print("BDD100K Anchor Box Computation")
    print("="*60)
    print(f"Original image size: {IMG_WIDTH}x{IMG_HEIGHT}")
    print(f"YOLO input size: {YOLO_INPUT_WIDTH}x{YOLO_INPUT_HEIGHT}")
    print(f"K-means method: {'Euclidean' if args.euclidean else 'IoU-based (recommended)'}")
    print(f"Class filter: {'All classes' if args.all_classes else TARGET_CLASSES}")
    print(f"Labels directory: {LABELS_DIR}")
    print("="*60)
    print()

    # Read training labels
    print("Reading training labels...")
    train_width, train_height = read_bdd100k_labels(
        LABELS_DIR, dataset_type="train", target_classes=TARGET_CLASSES
    )

    # Read validation labels
    print("\nReading validation labels...")
    val_width, val_height = read_bdd100k_labels(
        LABELS_DIR, dataset_type="val", target_classes=TARGET_CLASSES
    )

    # Combine train and val
    print("\nCombining train and validation data...")
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
        yolo_input_width=YOLO_INPUT_WIDTH,
        yolo_input_height=YOLO_INPUT_HEIGHT,
        n_anchors=9,
        save_path=PLT_SAVE_PATH,
        use_iou_kmeans=not args.euclidean
    )

    # Save anchors to text file
    with open(ANCHORS_TXT, "w") as f:
        f.write("# BDD100K Custom Anchors\n")
        f.write(f"# Image size: {IMG_WIDTH}x{IMG_HEIGHT}\n")
        f.write(f"# YOLO input: {YOLO_INPUT_WIDTH}x{YOLO_INPUT_HEIGHT}\n")
        f.write(f"# K-means method: {'Euclidean' if args.euclidean else 'IoU-based'}\n")
        f.write(f"# Classes: {'All' if args.all_classes else str(TARGET_CLASSES)}\n")
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

    print(f"Anchors saved to: {ANCHORS_TXT}")

    # Compare with YOLOP's original anchors
    print("\n" + "="*60)
    print("COMPARISON WITH YOLOP ORIGINAL ANCHORS")
    print("="*60)
    yolop_anchors = np.array([
        [3, 9], [5, 11], [4, 20],      # P3 (small)
        [7, 18], [6, 39], [12, 31],    # P4 (medium)
        [19, 50], [38, 81], [68, 157]  # P5 (large)
    ])

    print("\nYOLOP Original (640x384 input, 10 classes):")
    for layer_idx in range(3):
        layer_name = ["P3 (small)", "P4 (medium)", "P5 (large)"][layer_idx]
        layer_anchors = yolop_anchors[layer_idx*3:(layer_idx+1)*3]
        areas = layer_anchors[:, 0] * layer_anchors[:, 1]
        print(f"  {layer_name}: {layer_anchors.tolist()} (areas: {areas.tolist()})")

    print(f"\nYour Computed ({YOLO_INPUT_WIDTH}x{YOLO_INPUT_HEIGHT} input):")
    for layer_idx in range(3):
        layer_name = ["P3 (small)", "P4 (medium)", "P5 (large)"][layer_idx]
        layer_anchors = anchors[layer_idx*3:(layer_idx+1)*3]
        areas = layer_anchors[:, 0] * layer_anchors[:, 1]
        print(f"  {layer_name}: {layer_anchors.tolist()} (areas: {areas.tolist()})")

    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("1. If your anchors are similar to YOLOP's -> use your computed anchors")
    print("2. If very different -> try --use-yolop-size flag or use YOLOP anchors directly")
    print("3. Update panoptic_perception/configs/models/yolo-detection-drivable.cfg line 179")
    print("4. Restart training with new anchors")
    print()


if __name__ == "__main__":
    main()
