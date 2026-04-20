import torch
from torch.utils.data import Dataset

import numpy as np
import os
import json
import cv2
import hashlib
import random
import albumentations as A
from collections import defaultdict

from panoptic_perception.dataset.enums import BDD100KClasses, BDD100KClassesReduced
from panoptic_perception.dataset.augmentations import (
    apply_augmentations, copy_paste_instances, random_perspective,
    mixup_augmentation, augment_hsv, flip_horizontal,
    letterbox_with_masks
)
from panoptic_perception.dataset.mosaic_augmentation import mosaic_augmentation
from panoptic_perception.dataset.adverse_weather import (
    apply_nighttime_fog,
    FogParameters, SyntheticFogGenerator, SyntheticLowLightGenerator,
    HeuristicDepthEstimator, DepthAnythingEstimator, ONNXDepthEstimator,
    RadialDistance
)

from panoptic_perception.utils.lane_utils import build_lane_targets

from enum import Enum

LANE_VIS_COLORS = [
    (255, 255, 255),  # single white
    (0, 255, 255),    # single yellow
    (0, 200, 255),    # double yellow
    (255, 128, 0),    # road curb
]

def visualize_batch(images, seg, drivable, targets, save_dir, batch_index=0,
                    lane_targets=None, lane_categories=None):
    """
    images: tensor (B,3,H,W) normalized to [0,1]
    targets: tensor (N,6) -> (batch_idx, class, x_center, y_center, w, h)
    lane_targets: tensor (B, max_lanes, 78) or None
    lane_categories: tensor (B, max_lanes) or None
    batch_index: index of image in batch to visualize
    """
    img = images[batch_index].permute(1,2,0).cpu().numpy()
    img = (img * 255).astype(np.uint8)

    H, W, _ = img.shape

    # collect targets for this specific image
    t = targets[targets[:,0] == batch_index]

    # convert xywh normalized -> pixel PascalVOC (x1,y1,x2,y2)
    boxes = []
    classes = []
    for row in t:
        _, cls, xc, yc, w, h = row

        xc *= W
        yc *= H
        w *= W
        h *= H

        x1 = xc - w/2
        y1 = yc - h/2
        x2 = xc + w/2
        y2 = yc + h/2

        boxes.append([x1, y1, x2, y2])
        classes.append(int(cls))

    # draw bounding boxes on the image
    img_draw = img.copy()
    for (x1,y1,x2,y2), cls in zip(boxes, classes):
        cv2.rectangle(img_draw,
                      (int(x1), int(y1)),
                      (int(x2), int(y2)),
                      (0,255,0), 2)
        cv2.putText(img_draw, str(cls),
                    (int(x1), int(y1)-5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0,255,0), 2)

    # draw lane targets
    if lane_targets is not None:
        lt = lane_targets[batch_index]  # (max_lanes, 78)
        lc = lane_categories[batch_index] if lane_categories is not None else None
        n_offsets = lt.shape[1] - 6  # 72

        # y-positions: 1.0 (bottom) to 0.0 (top), matching CLRNet prior_ys
        prior_ys = np.linspace(1.0, 0.0, n_offsets)

        for i in range(lt.shape[0]):
            if lt[i, 0].item() < 0.5:  # not valid
                continue

            xs = lt[i, 6:].cpu().numpy()  # (72,) normalized x-coords
            cat_idx = int(lc[i].item()) if lc is not None and lc[i].item() >= 0 else 0
            color = LANE_VIS_COLORS[cat_idx % len(LANE_VIS_COLORS)]

            pts = []
            for j in range(n_offsets):
                if xs[j] > -1e4:
                    x_pix = int(xs[j] * (W - 1))
                    y_pix = int(prior_ys[j] * (H - 1))
                    pts.append((x_pix, y_pix))

            for k in range(len(pts) - 1):
                cv2.line(img_draw, pts[k], pts[k + 1], color, 2)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cv2.imwrite(f'{save_dir}/sample_batch_{batch_index}.png', cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR))

    if seg is not None:
        if seg[batch_index].ndim == 3:
            seg_vis = seg[batch_index].permute(1, 2, 0).cpu().numpy()
        else:
            seg_vis = seg[batch_index].cpu().numpy()

        # Scale segmentation class IDs for visibility (multiply by factor)
        # BDD100K has ~19 semantic classes, scale to spread across 0-255
        seg_scaled = (seg_vis * 12).clip(0, 255).astype(np.uint8)
        cv2.imwrite(f'{save_dir}/sample_batch_seg_{batch_index}.png', cv2.cvtColor(seg_scaled, cv2.COLOR_RGB2BGR))

    if drivable is not None:
        if drivable[batch_index].ndim == 3:
            drivable_vis = drivable[batch_index].permute(1, 2, 0).cpu().numpy()
        else:
            drivable_vis = drivable[batch_index].cpu().numpy()

        # BDD100K drivable area classes: 0=background, 1=direct, 2=alternative
        # Create color visualization for proper visibility
        drivable_colored = np.zeros((*drivable_vis.shape, 3), dtype=np.uint8)
        drivable_colored[drivable_vis == 1] = [0, 255, 0]    # Direct drivable: green
        drivable_colored[drivable_vis == 2] = [0, 255, 255]  # Alternative: yellow

        cv2.imwrite(f'{save_dir}/sample_batch_drivable_{batch_index}.png', drivable_colored)
    


class DatasetMode(Enum):
    TRAIN = "train"
    EVAL = "eval"
    INFER = "infer"

class BDDPreprocessor:
    def __init__(self, preprocess_kwargs:dict):
        self.preprocess_kwargs = preprocess_kwargs

        self.image_resize = preprocess_kwargs.get("image_resize", (640, 640))
        self.original_size = preprocess_kwargs.get("original_image_size", (720, 1280))

        self.original_width = self.original_size[1]
        self.original_height = self.original_size[0]

        self.resized_width = self.image_resize[1]
        self.resized_height = self.image_resize[0]

        # YOLOP-style augmentation parameters
        # Optimized for small objects and long-tail classes
        self.augment_params = preprocess_kwargs.get("augment_params", {
            # Geometric augmentations (reduced to preserve small objects)
            'degrees': 10,          # Reduced from 15 - less rotation preserves small objects
            'translate': 0.1,     # Reduced from 0.1 - keep objects in frame
            'scale': 0.25,         # Increased from 0.1 - multi-scale helps small objects
            'shear': 5,            # Reduced from 10 - less distortion

            # Color augmentations (kept aggressive for robustness)
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,

            # Noise (reduced - hurts small objects)
            'salt_prob': 0.005,    # Reduced from 0.1 - too much noise hurts small objects
            'pepper_prob': 0.005,  # Reduced from 0.1

            # Flip
            "flip_prob": 0.5,

            # Output size
            "img_size": (self.resized_height, self.resized_width)
        })
        
        max_size = max(self.resized_height, self.resized_width)

        base_resize_image = [A.LongestMaxSize(max_size=max_size, interpolation=cv2.INTER_LINEAR), 
                       A.PadIfNeeded(self.resized_height, self.resized_width, 
                                    border_mode=cv2.BORDER_CONSTANT, fill=(114, 114, 114))]

        base_resize_mask = [A.LongestMaxSize(max_size=max_size, interpolation=cv2.INTER_NEAREST), 
                       A.PadIfNeeded(self.resized_height, self.resized_width, 
                                    border_mode=cv2.BORDER_CONSTANT, fill_mask=0)]

        self.transformation = A.Compose(
            base_resize_image,
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"])
            )

        # self.image_only_transformation = A.Compose(
        #         [A.Resize(height=self.resized_height, width=self.resized_width)]
        #     )

        # self.mask_only_transformation = A.Compose(
        #     [A.Resize(height=self.resized_height, width=self.resized_width, interpolation=cv2.INTER_NEAREST)]
        # )
        
        self.image_only_transformation = A.Compose(base_resize_image)        
        self.mask_only_transformation = A.Compose(base_resize_mask)

    def load_detection(self, json_path, filter_by_area=False):
        """
        Load 2D bbox annotations from a BDD100K detection JSON file.

        Returns:
            bboxes: list of [x1, y1, x2, y2] in original pixel coordinates
            class_labels - list of label_id
            attributes - dict of occluded, truncated, trafficLightColor
        """        
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        bboxes = []
        class_labels = []
        
        MIN_AREA = 15 * 15
        
        for frames in data["frames"]:
            for item in frames["objects"]:
                if "box2d" in item:
                    box = item["box2d"]
                    label = item["category"]
                    label = '_'.join(label.split())
                    # label_id = BDD100KClasses.from_label(label)
                    label_id = BDD100KClassesReduced.from_label(label)
                    
                    if box["x2"] > box["x1"] and box["y2"] > box["y1"]: 

                        if filter_by_area:
                            w = box["x2"] - box["x1"]
                            h = box["y2"] - box["y1"]
                            area = w * h

                            # Drop only class 2 if smaller than threshold
                            if label_id == 2 and area < MIN_AREA:
                                continue                            

                        bboxes.append([box["x1"], box["y1"], box["x2"], box["y2"]])
                        class_labels.append(label_id)

        attributes = data.get("attributes", None)
        if attributes is not None:
            assert type(attributes) == dict, f'Expected Attributes to be a dict, got: {type(attributes)}'

        return bboxes, class_labels, attributes

    def load_lane_annotations(self, json_path):
        """
        Load lane polyline annotations from a BDD100K detection JSON file.

        Returns:
            lane_polys: list of dicts {"points": np.array(N,2), "category": str}
                        points in original pixel coordinates
        """
        lane_polys = []

        if not os.path.exists(json_path):
            return lane_polys

        with open(json_path, 'r') as f:
            data = json.load(f)

        for frames in data["frames"]:
            for item in frames["objects"]:
                if "poly2d" not in item:
                    continue
                category = item.get("category", "")
                if not category.startswith("lane/"):
                    continue

                pts = np.array(
                    [[p[0], p[1]] for p in item["poly2d"] if p[2] == "L"],
                    dtype=np.float32
                )
                if len(pts) >= 2:
                    lane_polys.append({
                        "points": pts,
                        "category": category
                    })

        return lane_polys

    def transform_lane_points_resize(self, lane_polys, orig_h, orig_w):
        """
        Apply the same LongestMaxSize + PadIfNeeded transform to lane points.
        Mirrors what self.transformation does to the image geometry.
        """
        if lane_polys is None:
            return None

        max_size = max(self.resized_height, self.resized_width)

        # LongestMaxSize: scale so longest side = max_size
        scale = max_size / max(orig_h, orig_w)
        new_h = int(orig_h * scale)
        new_w = int(orig_w * scale)

        # PadIfNeeded: symmetric padding to reach target dims
        pad_top = (self.resized_height - new_h) // 2
        pad_left = (self.resized_width - new_w) // 2

        for poly in lane_polys:
            poly["points"] = poly["points"].astype(np.float32)
            poly["points"][:, 0] = poly["points"][:, 0] * scale + pad_left
            poly["points"][:, 1] = poly["points"][:, 1] * scale + pad_top

        return lane_polys

    def prepare_targets_2d(self, boxes, labels):
        if len(boxes) == 0:
            return np.zeros((0, 5), dtype=np.float32)

        targets = []
        for bbox, class_id in zip(boxes, labels):
            left, top, right, bottom = bbox

            x_center = (left + right) / 2 / self.image_resize[1]
            y_center = (top + bottom) / 2 / self.image_resize[0]
            width = (right - left) / self.image_resize[1]
            height = (bottom - top) / self.image_resize[0]

            targets.append([class_id, x_center, y_center, width, height])

        return np.array(targets, dtype=np.float32)

    def to_pascal_voc(self, labels_xywh, h, w):
        bboxes, class_labels = [], []
        for label in labels_xywh:
            cls, cx, cy, bw, bh = label
            x1 = (cx - bw/2) * w
            y1 = (cy - bh/2) * h
            x2 = (cx + bw/2) * w
            y2 = (cy + bh/2) * h
            
            bboxes.append([x1, y1, x2, y2])
            class_labels.append(int(cls))
            
        return bboxes, class_labels

    def to_normalize_xywh(self, bboxes, labels, h, w):
        
        source_labels = []
        for (x1, y1, x2, y2), cls in zip(bboxes, labels):
            cx = (x1 + x2)/2/w
            cy = (y1 + y2)/2/h
            bw = (x2 - x1)/w
            bh = (y2 - y1)/h
            
            source_labels.append([cls, cx, cy, bw, bh])
        
        source_labels = np.array(source_labels) if source_labels else np.zeros((0, 5))
        return source_labels
    
    def normalize_tensor(self, tensor:torch.Tensor):

        tensor = tensor.float() / 255.0 
        return tensor
    
    def mosaic_augmentation(self, images_list, bboxes_list, class_labels_list,
                            segs_list, drivables_list, lane_polys_list=None):

        image, labels_xywh, seg, drivable, lane_polys = mosaic_augmentation(
            images_list, bboxes_list, class_labels_list, segs_list, drivables_list,
            lane_polys_list=lane_polys_list,
            output_size=self.image_resize
        )

        image, seg, drivable, labels_xywh, lane_polys = random_perspective(
            image, seg, drivable, labels_xywh, lane_polys=lane_polys,
            degrees=self.augment_params["degrees"],
            translate=self.augment_params["translate"],
            scale=self.augment_params["scale"],
            shear=self.augment_params["shear"],
        )

        # Apply remaining augmentations (HSV, flip, etc. but skip letterbox since mosaic outputs correct size)
        augment_hsv(
            image,
            self.augment_params.get("hsv_h", 0.015),
            self.augment_params.get("hsv_s", 0.7),
            self.augment_params.get("hsv_v", 0.4)
        )

        if random.random() < 0.5:
            image, seg, drivable, labels_xywh, lane_polys = flip_horizontal(
                image, seg, drivable, labels_xywh, lane_polys=lane_polys
            )

        return image, seg, drivable, labels_xywh, lane_polys
    
    def mixup_augmentation(self, img1, bboxes1, class_labels1, seg1, drivable1,
                           img2, bboxes2, class_labels2, seg2, drivable2,
                           lane_polys1=None, lane_polys2=None):

        orig_h1, orig_w1 = img1.shape[:2]
        orig_h2, orig_w2 = img2.shape[:2]

        # Transform image 1
        t1 = self.transformation(
            image=img1, bboxes=bboxes1, class_labels=class_labels1
        )
        img1, bboxes1, class_labels1 = t1['image'], t1['bboxes'], t1['class_labels']

        if seg1 is not None:
            seg1 = self.mask_only_transformation(image=seg1)['image']
        if drivable1 is not None:
            drivable1 = self.mask_only_transformation(image=drivable1)['image']
        if lane_polys1 is not None:
            lane_polys1 = self.transform_lane_points_resize(lane_polys1, orig_h1, orig_w1)

        # Transform image 2
        t2 = self.transformation(
            image=img2, bboxes=bboxes2, class_labels=class_labels2
        )
        img2, bboxes2, class_labels2 = t2['image'], t2['bboxes'], t2['class_labels']

        if seg2 is not None:
            seg2 = self.mask_only_transformation(image=seg2)['image']
        if drivable2 is not None:
            drivable2 = self.mask_only_transformation(image=drivable2)['image']
        if lane_polys2 is not None:
            lane_polys2 = self.transform_lane_points_resize(lane_polys2, orig_h2, orig_w2)

        labels1 = self.prepare_targets_2d(boxes=bboxes1, labels=class_labels1)
        labels2 = self.prepare_targets_2d(boxes=bboxes2, labels=class_labels2)

        image, labels_xywh, seg, drivable, lane_polys = mixup_augmentation(
            img1=img1, img2=img2, seg1=seg1, seg2=seg2,
            drivable1=drivable1, drivable2=drivable2,
            labels1=labels1, labels2=labels2, alpha=0.5,
            lane_polys1=lane_polys1, lane_polys2=lane_polys2
        )

        return image, labels_xywh, seg, drivable, lane_polys

    def standard_augmentations(self, image, seg, drivable, bboxes, class_labels,
                               lane_polys=None):

        orig_h, orig_w = image.shape[:2]

        t = self.transformation(image=image,
                            bboxes=bboxes,
                            class_labels=class_labels)

        image, bboxes, class_labels = t['image'], t['bboxes'], t['class_labels']
        if seg is not None:
            seg = self.mask_only_transformation(image=seg)['image']
        if drivable is not None:
            drivable = self.mask_only_transformation(image=drivable)['image']
        if lane_polys is not None:
            lane_polys = self.transform_lane_points_resize(lane_polys, orig_h, orig_w)

        labels_xywh = self.prepare_targets_2d(bboxes, class_labels)

        image, seg, drivable, labels_xywh, lane_polys = apply_augmentations(
            img=image,
            seg=seg,
            drivable=drivable,
            labels=labels_xywh,
            params=self.augment_params,
            img_size=self.image_resize,
            lane_polys=lane_polys
        )

        return image, seg, drivable, labels_xywh, lane_polys
        
    @staticmethod
    def collate_fn(batch:dict):

        batch_images = []
        batch_clean_images = []
        batch_targets = []
        batch_segmentation_masks = []
        batch_drivable_masks = []
        batch_lane_targets = []
        batch_lane_categories = []
        batch_image_paths = []
        batch_scene_attributes = []

        for batch_idx, batch_items in enumerate(batch):
            image = batch_items['image']
            image_path = batch_items["image_path"]
            scene_attributes = batch_items["scene_attributes"]
            clean_image = batch_items.get("clean_image")

            assert image is not None, f"Image tensor at batch index {batch_idx} is None."
            assert image.ndim == 3, "Image tensor must have 3 dimensions (C, H, W)."

            batch_images.append(image)
            batch_image_paths.append(image_path)
            batch_scene_attributes.append(scene_attributes)

            if clean_image is not None:
                batch_clean_images.append(clean_image)

            if batch_items['detection_targets'] is not None and batch_items['detection_targets'].shape[0] > 0:
                nt = batch_items['detection_targets'].shape[0]
                batch_targets.append(
                    torch.cat(
                        [
                            torch.full(size=(nt,1),
                                       fill_value=batch_idx,
                                       dtype=batch_items['detection_targets'].dtype),
                            batch_items['detection_targets']
                        ], dim=1
                    )
                )

            if batch_items['segmentation_mask'] is not None:
                batch_segmentation_masks.append(batch_items['segmentation_mask'])

            if batch_items['drivable_mask'] is not None:
                batch_drivable_masks.append(batch_items['drivable_mask'])

            if batch_items.get('lane_targets') is not None:
                batch_lane_targets.append(batch_items['lane_targets'])
            if batch_items.get('lane_categories') is not None:
                batch_lane_categories.append(batch_items['lane_categories'])

        batch_images_tensor = torch.stack(batch_images, dim=0)
        batch_clean_images = torch.stack(batch_clean_images, dim=0) if batch_clean_images else None
        batch_targets_tensor = torch.cat(batch_targets, dim=0) if batch_targets else None

        # Sanitize detection targets — all paths (augment, mosaic, mixup, non-augment)
        # can produce out-of-bounds or degenerate boxes
        # Format: [batch_idx, class_id, cx, cy, w, h]
        if batch_targets_tensor is not None:
            batch_targets_tensor[:, 2:4] = batch_targets_tensor[:, 2:4].clamp(0.0, 1.0)  # cx, cy
            batch_targets_tensor[:, 4:6] = batch_targets_tensor[:, 4:6].clamp(0.001, 1.0)  # w, h
        batch_segmentation_masks_tensor = torch.stack(batch_segmentation_masks, dim=0) if batch_segmentation_masks else None
        batch_drivable_masks_tensor = torch.stack(batch_drivable_masks, dim=0) if batch_drivable_masks else None
        batch_lane_targets_tensor = torch.stack(batch_lane_targets, dim=0) if batch_lane_targets else None
        batch_lane_categories_tensor = torch.stack(batch_lane_categories, dim=0) if batch_lane_categories else None

        return {
            "images": batch_images_tensor,
            "clean_images": batch_clean_images,
            "detections": batch_targets_tensor,
            "segmentation_masks": batch_segmentation_masks_tensor,
            "drivable_area_seg": batch_drivable_masks_tensor,
            "lanes_detections": batch_lane_targets_tensor,
            "lane_categories": batch_lane_categories_tensor,
            "image_paths":batch_image_paths,
            "scene_attributes":batch_scene_attributes
        }
    
    def prepare_inference(self, image_path=None):
        
        assert image_path is not None and os.path.exists(image_path), f"Invalid Image path {image_path}"
        img = cv2.imread(image_path)
        
        orig = img.copy()
        h0, w0 = img.shape[:2]

        transformed = self.image_only_transformation(image=img)
        img = transformed['image']

        img = self.normalize_tensor(torch.from_numpy(img).permute(2, 0, 1))

        return {
            "image": img.unsqueeze(0),  # add batch dim
            "original_image": orig,
            "orig_shape": (h0, w0),
            "new_shape": img.shape[1:]
        }

class BDD100KDataset(Dataset):
    def __init__(self, dataset_kwargs:dict, dataset_type:str='train',
                perform_augmentation:bool=False, mode:DatasetMode = DatasetMode.TRAIN):
        super(BDD100KDataset, self).__init__()

        assert os.path.exists(dataset_kwargs["images_dir"]), f"Images directory {dataset_kwargs['images_dir']} does not exist."

        self.images_dir = dataset_kwargs["images_dir"]
        self.detection_annotations_dir = dataset_kwargs["detection_annotations_dir"]
        self.segmentation_annotations_dir = dataset_kwargs["segmentation_annotations_dir"]
        self.drivable_annotations_dir = dataset_kwargs["drivable_annotations_dir"]

        self.dataset_type = dataset_type
        self.perform_augmentation = perform_augmentation
        self.mode = mode

        self.preprocessor = BDDPreprocessor(dataset_kwargs.get("preprocessor_kwargs", {}))
        self.image_ids = self.get_image_ids()

        # Advanced augmentation probabilities (only for training)
        aug_config = dataset_kwargs.get("preprocessor_kwargs", {}).get("advanced_aug", {})
        self.mosaic_prob = aug_config.get("mosaic_prob", 0.5) if perform_augmentation else 0.0
        self.mixup_prob = aug_config.get("mixup_prob", 0.15) if perform_augmentation else 0.0
        self.copy_paste_prob = aug_config.get("copy_paste_prob", 0.3) if perform_augmentation else 0.0
        
    def get_image_ids(self):
        return [f.split('.')[0] for f in os.listdir(os.path.join(self.images_dir, self.dataset_type))]

    def __len__(self):
        return len(self.image_ids)

    def _load_raw(self, index):
        """Load raw image and annotations without augmentation for mosaic/mixup."""
        image_id = self.image_ids[index]

        image_path = os.path.join(self.images_dir, self.dataset_type, f"{image_id}.jpg")
        seg_path = os.path.join(self.segmentation_annotations_dir, self.dataset_type, f"{image_id}_train_id.png")
        drivable_path = os.path.join(self.drivable_annotations_dir, self.dataset_type, f"{image_id}_drivable_id.png")

        assert os.path.exists(image_path), f"Image path {image_path} does not exist."

        # Load image
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load masks
        seg = None
        if os.path.exists(seg_path):
            seg = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)

        drivable = None
        if os.path.exists(drivable_path):
            drivable = cv2.imread(drivable_path, cv2.IMREAD_GRAYSCALE)
            #merge alternative to drivable
            drivable[drivable == 2] = 1

        # Load detection and lane annotations
        bboxes = []
        class_labels = []
        lane_polys = []

        det_path = os.path.join(self.detection_annotations_dir, self.dataset_type, f"{image_id}.json")
        if self.mode != DatasetMode.INFER:
            assert os.path.exists(det_path), f"Detection path {det_path} does not exist."
            bboxes, class_labels, scene_attributes = self.preprocessor.load_detection(det_path)
            lane_polys = self.preprocessor.load_lane_annotations(det_path)

        return image, bboxes, class_labels, scene_attributes, seg, drivable, lane_polys, image_path

    def __getitem__(self, index):
        if self.mode == DatasetMode.INFER:
            image_id = self.image_ids[index]
            image_path = os.path.join(self.images_dir, self.dataset_type, f"{image_id}.jpg")
            return self.preprocessor.prepare_inference(image_path)
        else:
            return self.prepare_training_sample(index)

    def prepare_training_sample(self, index):

        # Check for advanced augmentations (mosaic, mixup, copy-paste)
        use_mosaic = self.perform_augmentation and random.random() < self.mosaic_prob
        use_mixup = self.perform_augmentation and random.random() < self.mixup_prob
        use_copy_paste = self.perform_augmentation and random.random() < self.copy_paste_prob
        image_path = None
        lane_polys = []

        if use_mosaic:
            # Mosaic: combine 4 images
            indices = [index] + [random.randint(0, len(self) - 1) for _ in range(3)]
            images_list, class_labels_list, bboxes_list = [], [], []
            segs_list, drivables_list, lane_polys_list = [], [], []

            for idx in indices:
                if idx == index:
                    image, bboxes, class_labels, scene_attributes, seg, drivable, lp, image_path = self._load_raw(idx)
                else:
                    image, bboxes, class_labels, scene_attributes, seg, drivable, lp, _ = self._load_raw(idx)

                images_list.append(image)
                class_labels_list.append(class_labels)
                bboxes_list.append(bboxes)
                segs_list.append(seg)
                drivables_list.append(drivable)
                lane_polys_list.append(lp)

            image, seg, drivable, labels_xywh, lane_polys = self.preprocessor.mosaic_augmentation(
                images_list, bboxes_list, class_labels_list,
                segs_list, drivables_list, lane_polys_list=lane_polys_list
            )

        elif use_mixup:
            # MixUp: blend 2 images
            image1, bboxes1, class_labels1, scene_attributes, seg1, drivable1, lp1, image_path = self._load_raw(index)
            idx2 = random.randint(0, len(self) - 1)
            image2, bboxes2, class_labels2, _, seg2, drivable2, lp2, _ = self._load_raw(idx2)

            image, labels_xywh, seg, drivable, lane_polys = self.preprocessor.mixup_augmentation(
                img1=image1, bboxes1=bboxes1, class_labels1=class_labels1,
                seg1=seg1, drivable1=drivable1,
                img2=image2, bboxes2=bboxes2, class_labels2=class_labels2,
                seg2=seg2, drivable2=drivable2,
                lane_polys1=lp1, lane_polys2=lp2
            )

        else:
            # Standard augmentation path
            image, bboxes, class_labels, scene_attributes, seg, drivable, lane_polys, image_path = self._load_raw(index)

            if self.perform_augmentation:
                # Apply copy-paste for long-tail classes before other augmentations
                #TODO, modify random -> sampling from known images containing long tail classes.
                if use_copy_paste:
                    source_idx = random.randint(0, len(self) - 1)
                    source_img, source_bboxes, source_class_labels, _, _, _, _, _ = self._load_raw(source_idx)

                    h, w = image.shape[:2]
                    if source_img.shape[:2] != image.shape[:2]:
                        source_img = cv2.resize(source_img, (w, h))

                    labels_xywh = self.preprocessor.to_normalize_xywh(bboxes, class_labels, h, w)
                    source_labels = self.preprocessor.to_normalize_xywh(source_bboxes, source_class_labels, h, w)

                    image, labels_xywh = copy_paste_instances(
                        image, labels_xywh, source_img, source_labels,
                        target_classes=[1, 3],  # RIDER=1, MOTOR=3
                        max_instances=2
                    )

                    bboxes, class_labels = self.preprocessor.to_pascal_voc(
                        labels_xywh, h, w
                    )

                image, seg, drivable, labels_xywh, lane_polys = self.preprocessor.standard_augmentations(
                    image=image, seg=seg, drivable=drivable,
                    bboxes=bboxes, class_labels=class_labels,
                    lane_polys=lane_polys
                )

            else:
                orig_h, orig_w = image.shape[:2]
                t = self.preprocessor.transformation(image=image,
                                                    bboxes=bboxes,
                                                    class_labels=class_labels)

                image, bboxes, class_labels = t['image'], t['bboxes'], t['class_labels']
                if seg is not None:
                    seg = self.preprocessor.mask_only_transformation(image=seg)['image']
                if drivable is not None:
                    drivable = self.preprocessor.mask_only_transformation(image=drivable)['image']
                if lane_polys:
                    lane_polys = self.preprocessor.transform_lane_points_resize(
                        lane_polys, orig_h, orig_w
                    )

                labels_xywh = self.preprocessor.prepare_targets_2d(bboxes, class_labels)

        # Build lane targets from augmented polylines
        img_h, img_w = image.shape[:2]
        lane_targets, lane_categories = build_lane_targets(
            lane_polys, img_h, img_w
        )

        # Convert to tensors
        image_tensor = self.preprocessor.normalize_tensor(torch.from_numpy(image).permute(2, 0, 1))
        targets = torch.from_numpy(labels_xywh).float()
        seg_tensor = torch.from_numpy(seg).long() if seg is not None else None
        drivable_tensor = torch.from_numpy(drivable).long() if drivable is not None else None

        return {
            "image": image_tensor,
            "segmentation_mask": seg_tensor,
            "drivable_mask": drivable_tensor,
            "detection_targets": targets,
            "lane_targets": lane_targets,
            "lane_categories": lane_categories,
            "image_path": image_path,
            "scene_attributes": scene_attributes
        }

class FoggyBDD100KDataset(BDD100KDataset):
    
    class FogLevels(Enum):
        LIGHT = 500
        MODERATE = 200
        DENSE = 100
        HEAVY = 50

        @classmethod
        def from_level(cls, fog_level: int):
            try:
                return cls(fog_level).name.lower()
            except ValueError:
                return None

        @classmethod
        def from_label(cls, label: str):
            return cls[label.upper()].value    
    
    def __init__(self, dataset_kwargs, dataset_type = 'train',
                perform_augmentation = False, mode = DatasetMode.TRAIN,
                strict_map:bool=True, apply_fog_prob:float=0.67,
                depth_estimator=None):

        super().__init__(dataset_kwargs, dataset_type, perform_augmentation, mode)

        self.adverse_params = dataset_kwargs["adverse_params"]

        self.depth_map_dir = dataset_kwargs.get("depth_map_dir", None)
        if self.depth_map_dir and os.path.exists(os.path.join(self.depth_map_dir, self.dataset_type)):
            self.depth_map_ids = self.get_depth_map_ids()
            self._cached_depth_stems = set(self.depth_map_ids)

            if strict_map:
                image_set = set(self.image_ids)
                depth_set = set(self.depth_map_ids)

                assert image_set == depth_set, (
                    f"Image-depth ID mismatch: "
                    f"{len(image_set - depth_set)} images missing depth, "
                    f"{len(depth_set - image_set)} depth maps missing images"
                )
            else:
                common_ids = set(self.image_ids) & set(self.depth_map_ids)
                self.image_ids = [id for id in self.image_ids if id in common_ids]
        else:
            self.depth_map_ids = []
            self._cached_depth_stems = set()

        self.apply_fog_prob = apply_fog_prob
        
        self.min_haze_level = self.FogLevels.LIGHT
        self.max_haze_level = self.FogLevels.DENSE
        
        #TODO, remove this attribute, replace with FogLevels : str = ["light", "Heavy"]
        self.fog_betas = self.adverse_params.get("fog_betas", [0.010, 0.020, 0.035])
        self.darkness_gammas = self.adverse_params.get("darkness_gammas", [1.5, 2.0, 3.5])
        self.atmospheric_light_quantile = self.adverse_params.get("atmospheric_light_quantile", 0.9)
        self.atmospheric_light_min_pixels = self.adverse_params.get("atmospheric_light_min_pixels", 10)
        self.atmospheric_light = self.adverse_params.get("atmospheric_light", None)
        self.max_depth_meters = self.adverse_params.get("max_depth_meters", 150.0)
        
        # All (beta, gamma) combinations
        self.variants = self._build_variants()

        self._served = defaultdict(set)
        
        if depth_estimator is not None:
            self.depth_estimator = depth_estimator
        else:
            self.depth_estimator = HeuristicDepthEstimator(
                vertical_weight=0.7,
                intensity_weight=0.2,
                edge_weight=0.1,
                edge_epsilon=1e-8,
                uint8_max=255.0,
            )

        self.fog_generator = SyntheticFogGenerator(
            depth_estimator=self.depth_estimator
        )
        self.lowlight_generator = SyntheticLowLightGenerator(
            gamma_min=min(self.darkness_gammas),
            gamma_max=max(self.darkness_gammas),
            gamma_min_threshold=1.0,
        )

    def get_depth_map_ids(self):
        return [f.split('.')[0] for f in os.listdir(os.path.join(self.depth_map_dir, self.dataset_type))]

    def _build_variants(self):
        """
        Builds variant list with three categories:
        - Fog only:      (beta, None)  — no darkness applied
        - Darkness only: (None, gamma) — no fog applied
        - Compound:      (beta, gamma) — both applied

        Controlled by adverse_params keys:
            "enable_fog_only": True (default)
            "enable_darkness_only": True (default)
            "enable_compound": True (default)
        """
        variants = []

        if self.adverse_params.get("enable_fog_only", True):
            variants += [(b, None) for b in self.fog_betas]

        if self.adverse_params.get("enable_darkness_only", True):
            variants += [(None, g) for g in self.darkness_gammas]

        if self.adverse_params.get("enable_compound", True):
            variants += [(b, g) for b in self.fog_betas for g in self.darkness_gammas]

        assert len(variants) > 0, "At least one variant type must be enabled"
        return variants

    def _select_degradation(self, scene_attributes, beta=None, gamma=None):
        """Adjust or skip degradation based on scene context."""
        """
        ┌───────────────────┬───────────────────────┬─────────────────────┐
        │       Scene       │     Fog applied?      │  Darkness applied?  │
        ├───────────────────┼───────────────────────┼─────────────────────┤
        │ clear × daytime   │ Yes (any beta)        │ Yes (any gamma)     │
        ├───────────────────┼───────────────────────┼─────────────────────┤
        │ clear × dawn/dusk │ Yes (any beta)        │ Yes (capped at 1.5) │
        ├───────────────────┼───────────────────────┼─────────────────────┤
        │ clear × night     │ Yes (any beta)        │ No                  │
        ├───────────────────┼───────────────────────┼─────────────────────┤
        │ rainy × daytime   │ Yes (capped at 0.010) │ Yes (any gamma)     │
        ├───────────────────┼───────────────────────┼─────────────────────┤
        │ foggy × daytime   │ No                    │ Yes (any gamma)     │
        ├───────────────────┼───────────────────────┼─────────────────────┤
        │ foggy × night     │ No                    │ No → serve clean    │
        ├───────────────────┼───────────────────────┼─────────────────────┤
        │ snowy × night     │ Yes (capped at 0.010) │ No                  │
        └───────────────────┴───────────────────────┴─────────────────────┘        
        """
        
        def sample_beta():
            visibility = np.random.uniform(self.min_haze_level.value, self.max_haze_level.value)
            beta = 3.912 / visibility
            return beta
        
        def sample_gamma():
            return np.random.uniform(self.lowlight_generator.gamma_min, self.lowlight_generator.gamma_max)

        weather = scene_attributes.get("weather", "undefined") if scene_attributes else "undefined"
        tod = scene_attributes.get("timeofday", "undefined") if scene_attributes else "undefined"

        # Default: sample both
        beta = sample_beta()
        gamma = sample_gamma()

        # Already foggy — don't add more fog
        if weather == "foggy":
            beta = None

        # Already night — skip darkness
        if tod == "night":
            gamma = None

        # Dawn/dusk — cap darkness to mild
        if tod == "dawn/dusk" and gamma is not None:
            gamma = min(gamma, 1.3)

        # Rainy/snowy — reduce fog intensity
        if weather in ("rainy", "snowy") and beta is not None:
            beta = min(beta, 0.010)

        return beta, gamma

    def _load_raw(self, index):
        image, bboxes, class_labels, \
            scene_attributes, seg, drivable, lane_polys, image_path = super()._load_raw(index)

        image_id = self.image_ids[index]
        depth_map_arr = None

        # # Try cached .npy first
        # if image_id in self._cached_depth_stems:
        #     depth_map_path = os.path.join(self.depth_map_dir, self.dataset_type, f"{image_id}.npy")
        #     if os.path.exists(depth_map_path):
        #         depth_map_arr = np.load(str(depth_map_path))

        # Fall back to on-the-fly depth estimation
        # if depth_map_arr is None:
        #     depth_map_arr = self.depth_estimator.estimate(image)

        return image, depth_map_arr, bboxes, class_labels, scene_attributes, seg, drivable, lane_polys, image_path
    
    def _next_variant(self, image_id: str):
        
        served = self._served[image_id]
    
        if len(served) >= len(self.variants):
            served.clear()

        remaining = [i for i in range(len(self.variants)) if i not in served]
        chosen_idx = random.choice(remaining)
        served.add(chosen_idx)

        return self.variants[chosen_idx]        
    
    def _apply_degradation(self, image:np.ndarray, depth_map_arr:np.ndarray, beta, gamma):
        """Apply fog and/or darkness to the raw image. Returns degraded uint8 RGB."""
        if beta is not None and gamma is not None:
            params = FogParameters(
                beta=beta,
                max_depth_meters=self.max_depth_meters,
                atmospheric_light_quantile=self.atmospheric_light_quantile,
                atmospheric_light_min_pixels=self.atmospheric_light_min_pixels,
                atmospheric_light=np.array(self.atmospheric_light) if self.atmospheric_light is not None else None,
            )
            image, _, _ = apply_nighttime_fog(
                image_rgb=image,
                fog_generator=self.fog_generator,
                fog_params=params,
                gamma_generator=self.lowlight_generator,
                gamma=gamma,
                apply_order="dark_then_fog",
                precomputed_depth=depth_map_arr,
            )
        elif beta is not None:
            params = FogParameters(
                beta=beta,
                max_depth_meters=self.max_depth_meters,
                atmospheric_light_quantile=self.atmospheric_light_quantile,
                atmospheric_light_min_pixels=self.atmospheric_light_min_pixels,
                atmospheric_light=np.array(self.atmospheric_light) if self.atmospheric_light is not None else None,
            )
            image, _, _ = self.fog_generator.generate(
                image_rgb=image,
                params=params,
                precomputed_depth=depth_map_arr,
            )
        elif gamma is not None:
            image = self.lowlight_generator.apply(image, gamma)

        return image

    def prepare_training_sample(self, index):
        image, depth_map_arr, bboxes, class_labels, \
            scene_attributes, seg, drivable, lane_polys, image_path = self._load_raw(index)

        # Step 1: Decide foggy or clean
        if self.mode == DatasetMode.TRAIN:
            apply_fog = random.random() < self.apply_fog_prob
        else:
            apply_fog = True

        # Step 2: Standard augmentations (with lane polys)
        if self.perform_augmentation:
            image, seg, drivable, labels_xywh, lane_polys = self.preprocessor.standard_augmentations(
                image=image, seg=seg, drivable=drivable,
                bboxes=bboxes, class_labels=class_labels,
                lane_polys=lane_polys
            )
        else:
            orig_h, orig_w = image.shape[:2]
            t = self.preprocessor.transformation(
                image=image, bboxes=bboxes, class_labels=class_labels
            )
            image, bboxes, class_labels = t['image'], t['bboxes'], t['class_labels']
            if seg is not None:
                seg = self.preprocessor.mask_only_transformation(image=seg)['image']
            if drivable is not None:
                drivable = self.preprocessor.mask_only_transformation(image=drivable)['image']
            if lane_polys:
                lane_polys = self.preprocessor.transform_lane_points_resize(
                    lane_polys, orig_h, orig_w
                )
            labels_xywh = self.preprocessor.prepare_targets_2d(bboxes, class_labels)

        clean_image = image.copy()

        if apply_fog:
            image_id = self.image_ids[index]
            depth_map_arr = self.depth_estimator.estimate(image)
            beta, gamma = self._select_degradation(scene_attributes)
            image = self._apply_degradation(image, depth_map_arr, beta, gamma)

        # Build lane targets from augmented polylines
        img_h, img_w = image.shape[:2]
        lane_targets, lane_categories = build_lane_targets(
            lane_polys, img_h, img_w
        )

        # Step 3: Convert to tensors (same output format as parent)
        image_tensor = self.preprocessor.normalize_tensor(torch.from_numpy(image).permute(2, 0, 1))
        clean_tensor = self.preprocessor.normalize_tensor(torch.from_numpy(clean_image).permute(2, 0, 1))
        targets = torch.from_numpy(labels_xywh).float()
        seg_tensor = torch.from_numpy(seg).long() if seg is not None else None
        drivable_tensor = torch.from_numpy(drivable).long() if drivable is not None else None

        return {
            "image": image_tensor,
            "clean_image":clean_tensor,
            "segmentation_mask": seg_tensor,
            "drivable_mask": drivable_tensor,
            "detection_targets": targets,
            "lane_targets": lane_targets,
            "lane_categories": lane_categories,
            "image_path": image_path,
            "scene_attributes": scene_attributes
        }