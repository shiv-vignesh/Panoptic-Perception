def analyze_class_distribution(dataset_kwargs: dict, dataset_type: str = "train"):
    """
    Compute class distributions and CE weights for both detection and segmentation
    across the full dataset.

    Args:
        dataset_kwargs: Dataset configuration dict
        dataset_type: 'train' or 'val'
    """
    from panoptic_perception.dataset.bdd100k_dataset import BDD100KDataset, BDDPreprocessor
    from panoptic_perception.dataset.enums import BDD100KClassesReduced
    from collections import defaultdict
    from torch.utils.data import DataLoader
    
    import numpy as np
    from tqdm import tqdm

    dataset = BDD100KDataset(
        dataset_kwargs=dataset_kwargs,
        dataset_type=dataset_type,
        perform_augmentation=False
    )

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        collate_fn=BDDPreprocessor.collate_fn
    )

    num_det_classes = 6
    det_class_counts = np.zeros(num_det_classes, dtype=np.int64)
    total_instances = 0

    num_seg_classes = 2
    seg_pixel_counts = np.zeros(num_seg_classes, dtype=np.int64)
    total_pixels = 0

    train_iter = tqdm(dataloader, desc=f'Computing Class Distribution')
    for i, data_items in enumerate(train_iter):
        detections = data_items["detections"]
        if detections is not None and detections.shape[0] > 0:
            class_ids = detections[:, 1].long().numpy()
            for cls_id in class_ids:
                if 0 <= cls_id < num_det_classes:
                    det_class_counts[cls_id] += 1
            total_instances += len(class_ids)

        drivable_area_seg = data_items["drivable_area_seg"]
        if drivable_area_seg is not None:
            total_pixels += drivable_area_seg.numel()
            for c in range(num_seg_classes):
                seg_pixel_counts[c] += (drivable_area_seg == c).sum().item()

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1} batches...")

    # DETECTION CLASS WEIGHTS
    print("\n" + "=" * 70)
    print(f"DETECTION CLASS DISTRIBUTION ({dataset_type})")
    print("=" * 70)
    print(f"{'Class':<20} {'Count':<12} {'Ratio':<12} {'Inv Freq Wt':<12} {'Sqrt Wt':<12}")
    print("-" * 70)

    det_ratios = det_class_counts / total_instances
    # Inverse frequency: total / (num_classes * class_count)
    det_inv_freq = total_instances / (num_det_classes * det_class_counts + 1e-8)
    # Normalize so min weight = 1.0
    det_inv_freq_norm = det_inv_freq / det_inv_freq.min()
    det_sqrt_norm = np.sqrt(det_inv_freq_norm)

    for c in range(num_det_classes):
        name = BDD100KClassesReduced.from_id(c) or f"class_{c}"
        print(f"{name:<20} {det_class_counts[c]:<12} {det_ratios[c]:<12.4f} {det_inv_freq_norm[c]:<12.2f} {det_sqrt_norm[c]:<12.2f}")

    print("-" * 70)
    print(f"{'Total instances:':<32} {total_instances}")
    print(f"\nInverse freq weights: [{', '.join(f'{w:.2f}' for w in det_inv_freq_norm)}]")
    print(f"Sqrt weights:         [{', '.join(f'{w:.2f}' for w in det_sqrt_norm)}]")

    # SEGMENTATION CLASS WEIGHTS
    print("\n" + "=" * 70)
    print(f"SEGMENTATION CLASS DISTRIBUTION ({dataset_type})")
    print("=" * 70)

    seg_ratios = seg_pixel_counts / total_pixels
    seg_inv_freq = total_pixels / (num_seg_classes * seg_pixel_counts + 1e-8)
    seg_inv_freq_norm = seg_inv_freq / seg_inv_freq.min()
    seg_sqrt_norm = np.sqrt(seg_inv_freq_norm)

    seg_class_names = ["background", "drivable"]
    print(f"{'Class':<20} {'Pixels':<16} {'Ratio':<12} {'Inv Freq Wt':<12} {'Sqrt Wt':<12}")
    print("-" * 70)
    for c in range(num_seg_classes):
        print(f"{seg_class_names[c]:<20} {seg_pixel_counts[c]:<16,} {seg_ratios[c]:<12.4f} {seg_inv_freq_norm[c]:<12.2f} {seg_sqrt_norm[c]:<12.2f}")

    print("-" * 70)
    print(f"{'Total pixels:':<36} {total_pixels:,}")
    print(f"\nInverse freq weights: [{', '.join(f'{w:.2f}' for w in seg_inv_freq_norm)}]")
    print(f"Sqrt weights:         [{', '.join(f'{w:.2f}' for w in seg_sqrt_norm)}]")

    return {
        "detection": {
            "class_counts": det_class_counts,
            "ratios": det_ratios,
            "inv_freq_weights": det_inv_freq_norm,
            "sqrt_weights": det_sqrt_norm,
        },
        "segmentation": {
            "pixel_counts": seg_pixel_counts,
            "ratios": seg_ratios,
            "inv_freq_weights": seg_inv_freq_norm,
            "sqrt_weights": seg_sqrt_norm,
        }
    }
        

if __name__ == "__main__":

    dataset_kwargs = {
        "images_dir": "/workspace/data/100k",
        "detection_annotations_dir": "/workspace/data/bdd100k_labels",
        "segmentation_annotations_dir": "",
        "drivable_annotations_dir": "/workspace/data/bdd100k_drivable_maps/labels",
        "preprocessor_kwargs": {
            "image_resize": (640, 640),
            "original_image_size": (720, 1280)
        }
    }

    results = analyze_class_distribution(dataset_kwargs, dataset_type="train")
    
"""
======================================================================                                                                                                                                        
  DETECTION CLASS DISTRIBUTION (train)                                                                                                                                                                          
  ======================================================================                                                                                                                                        
  Class                Count        Ratio        Inv Freq Wt  Sqrt Wt                                                                                                                                           
  ----------------------------------------------------------------------                                                                                                                                        
  person               91405        0.0710       8.27         2.88                                                                                                                                              
  rider                4521         0.0035       167.16       12.93                                                                                                                                             
  vehicles             755740       0.5868       1.00         1.00                                                                                                                                              
  motor                10227        0.0079       73.90        8.60                                                                                                                                              
  traffic_light        186224       0.1446       4.06         2.01                                                                                                                                              
  traffic_sign         239893       0.1863       3.15         1.77                                                                                                                                              
  ----------------------------------------------------------------------                                                                                                                                        
  Total instances:                 1288010                                                                                                                                                                      
                                                                                                                                                                                                                
  Inverse freq weights: [8.27, 167.16, 1.00, 73.90, 4.06, 3.15]                                                                                                                                                 
  Sqrt weights:         [2.88, 12.93, 1.00, 8.60, 2.01, 1.77]                                                                                                                                                   
                                                                                                                                                                                                                
  ======================================================================                                                                                                                                        
  SEGMENTATION CLASS DISTRIBUTION (train)                                                                                                                                                                       
  ======================================================================                                                                                                                                        
  Class                Pixels           Ratio        Inv Freq Wt  Sqrt Wt                                                                                                                                       
  ----------------------------------------------------------------------                                                                                                                                        
  background           23,817,923,244   0.8307       1.00         1.00                                                                                                                                          
  drivable             4,854,076,756    0.1693       4.91         2.22                                                                                                                                          
  ----------------------------------------------------------------------                                                                                                                                        
  Total pixels:                        28,672,000,000                                                                                                                                                           
                                                                                                                                                                                                                
  Inverse freq weights: [1.00, 4.91]
  Sqrt weights:         [1.00, 2.22]
  
"""
