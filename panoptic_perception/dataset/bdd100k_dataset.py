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
from enum import Enum

from typing import List, Dict, Tuple

import warnings
warnings.simplefilter("once", UserWarning)

from panoptic_perception.dataset.enums import BDD100KClasses, BDD100KClassesReduced
from panoptic_perception.dataset.augmentations import (
    apply_augmentations, copy_paste_instances, random_perspective,
    mixup_augmentation, augment_hsv, flip_horizontal,
    letterbox_with_masks,
)
from panoptic_perception.dataset.mosaic_augmentation import mosaic_augmentation
from panoptic_perception.dataset.adverse_weather import (
    apply_nighttime_fog,
    FogParameters, SyntheticFogGenerator, SyntheticLowLightGenerator,
    HeuristicDepthEstimator, DepthAnythingEstimator, ONNXDepthEstimator,
    RadialDistance
)
from panoptic_perception.utils.lane_utils import (
    build_lane_targets, build_lane_seg_mask
)
from panoptic_perception.dataset.types import (
    DatasetMode, Bbox, ObjDetInstance, FrameObjDetections,
    LanePoly, FrameLaneDetections,
    FrameData
)

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

        self.image_only_transformation = A.Compose(base_resize_image)        
        self.mask_only_transformation = A.Compose(base_resize_mask)

    def load_detection(self, json_path, filter_by_area=False) -> FrameObjDetections:

        """
        Load 2D bbox annotations from a BDD100K detection JSON file.

        Returns:
            bboxes: list of [x1, y1, x2, y2] in original pixel coordinates
            class_labels - list of label_id
            attributes - dict of occluded, truncated, trafficLightColor
        """        
        with open(json_path, 'r') as f:
            data = json.load(f)

        
        MIN_AREA = 15 * 15
        frame_detections = FrameObjDetections()

        for frames in data["frames"]:
            for item in frames["objects"]:
                if "box2d" in item:
                    bbox = Bbox(item["box2d"]["x1"], item["box2d"]["x2"], 
                               item["box2d"]["y1"], item["box2d"]["y2"])
                    
                    if bbox.valid_bbox:
                        if filter_by_area:

                            # Drop only class 2 if smaller than threshold
                            if label_id == 2 and bbox.area < MIN_AREA:
                                continue

                        label = item["category"]
                        label = '_'.join(label.split())
                        label_id = BDD100KClassesReduced.from_label(label)

                        frame_detections.detections.append(ObjDetInstance(
                            bbox=bbox,
                            label_id=label_id,
                            label=label
                        ))

        attributes = data.get("attributes", None)
        if attributes is not None:
            if not isinstance(attributes, dict):
                warnings.warn(
                    f"Expected Attributes to be a dict, got: {type(attributes)}, fallback to None",
                    category=UserWarning,
                    stacklevel=2
                )                
                attributes = None

        frame_detections.attributes = attributes

        return frame_detections

    @staticmethod
    def _expand_poly2d_to_polyline(poly2d, samples_per_curve=20):
        """
        Convert BDD100K poly2d (mixed L/C/M codes) into a dense on-curve polyline.

        BDD100K uses matplotlib Path.CURVE4 (cubic Bezier) encoding:
            'M' -> move-to (start point, on-curve)
            'L' -> line-to (anchor point, on-curve)
            'C' -> cubic Bezier control point (off-curve, in groups of 3:
                   the 3rd is the next on-curve anchor; first two are tangent hints)

        Treating 'C' points as polyline vertices produces zigzags since they
        are off the actual curve. This samples each cubic from L0 -> C3 using
        C1, C2 as controls, yielding dense on-curve points.
        """
        if not poly2d:
            return np.zeros((0, 2), dtype=np.float32)

        points = []
        i = 0
        n = len(poly2d)

        if poly2d[0][2] in ('M', 'L'):
            points.append([poly2d[0][0], poly2d[0][1]])
            i = 1

        while i < n:
            code = poly2d[i][2]

            if code in ('L', 'M'):
                points.append([poly2d[i][0], poly2d[i][1]])
                i += 1
            elif code == 'C':
                if (i + 2 >= n
                        or poly2d[i + 1][2] != 'C'
                        or poly2d[i + 2][2] != 'C'
                        or not points):
                    i += 1
                    continue
                P0 = np.array(points[-1], dtype=np.float64)
                P1 = np.array(poly2d[i][:2], dtype=np.float64)
                P2 = np.array(poly2d[i + 1][:2], dtype=np.float64)
                P3 = np.array(poly2d[i + 2][:2], dtype=np.float64)
                for t in np.linspace(0.0, 1.0, samples_per_curve)[1:]:
                    pt = ((1 - t) ** 3 * P0
                          + 3 * (1 - t) ** 2 * t * P1
                          + 3 * (1 - t) * t ** 2 * P2
                          + t ** 3 * P3)
                    points.append(pt.tolist())
                i += 3
            else:
                i += 1

        return np.array(points, dtype=np.float32)

    def load_lane_annotations(self, json_path):
        """
        Load lane polyline annotations from a BDD100K detection JSON file.

        Returns:
            lane_polys: list of dicts {"points": np.array(N,2), "category": str}
                        points in original pixel coordinates
        """

        lane_polys = FrameLaneDetections()
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

                pts = self._expand_poly2d_to_polyline(item["poly2d"])

                if len(pts) >= 2:
                    lane_poly = LanePoly(
                        pts, category
                    )
                    lane_polys.lane_polys.append(lane_poly)

        return lane_polys

    def normalize_tensor(self, tensor:torch.Tensor):

        tensor = tensor.float() / 255.0 
        return tensor

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

    @staticmethod
    def collate_fn(batch):
        """Stack per-sample dicts into a batched dict suitable for the model.

        Detection targets get a leading batch_idx column appended; segmentation
        and lane tensors are stacked along dim 0. Missing optional fields are
        returned as None so downstream code can gate on their presence.
        """
        batch_images = []
        batch_clean_images = []
        batch_targets = []
        batch_segmentation_masks = []
        batch_drivable_masks = []
        batch_lane_targets = []
        batch_lane_categories = []
        batch_image_paths = []
        batch_scene_attributes = []
        batch_lane_seg_masks = []

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

            det = batch_items['detection_targets']
            if det is not None and det.shape[0] > 0:
                nt = det.shape[0]
                batch_targets.append(
                    torch.cat([
                        torch.full((nt, 1), batch_idx, dtype=det.dtype),
                        det,
                    ], dim=1)
                )

            if batch_items['segmentation_mask'] is not None:
                batch_segmentation_masks.append(batch_items['segmentation_mask'])
            if batch_items['drivable_mask'] is not None:
                batch_drivable_masks.append(batch_items['drivable_mask'])
            if batch_items.get('lane_targets') is not None:
                batch_lane_targets.append(batch_items['lane_targets'])
            if batch_items.get('lane_categories') is not None:
                batch_lane_categories.append(batch_items['lane_categories'])
            if batch_items.get('lane_seg_mask') is not None:
                batch_lane_seg_masks.append(batch_items['lane_seg_mask'])

        batch_images_tensor = torch.stack(batch_images, dim=0)
        batch_clean_images_tensor = (
            torch.stack(batch_clean_images, dim=0) if batch_clean_images else None
        )
        batch_targets_tensor = (
            torch.cat(batch_targets, dim=0) if batch_targets else None
        )

        # Sanitize detection targets — all augmentation paths can produce
        # out-of-bounds or degenerate boxes.
        # Format after collate: [batch_idx, class_id, cx, cy, w, h]
        if batch_targets_tensor is not None:
            batch_targets_tensor[:, 2:4] = batch_targets_tensor[:, 2:4].clamp(0.0, 1.0)
            batch_targets_tensor[:, 4:6] = batch_targets_tensor[:, 4:6].clamp(0.001, 1.0)

        batch_segmentation_masks_tensor = (
            torch.stack(batch_segmentation_masks, dim=0) if batch_segmentation_masks else None
        )
        batch_drivable_masks_tensor = (
            torch.stack(batch_drivable_masks, dim=0) if batch_drivable_masks else None
        )
        batch_lane_targets_tensor = (
            torch.stack(batch_lane_targets, dim=0) if batch_lane_targets else None
        )
        batch_lane_categories_tensor = (
            torch.stack(batch_lane_categories, dim=0) if batch_lane_categories else None
        )
        batch_lane_seg_masks_tensor = (
            torch.stack(batch_lane_seg_masks, dim=0) if batch_lane_seg_masks else None
        )

        return {
            "images": batch_images_tensor,
            "clean_images": batch_clean_images_tensor,
            "detections": batch_targets_tensor,
            "segmentation_masks": batch_segmentation_masks_tensor,
            "drivable_area_seg": batch_drivable_masks_tensor,
            "lanes_detections": batch_lane_targets_tensor,
            "lane_seg_masks": batch_lane_seg_masks_tensor,
            "lane_categories": batch_lane_categories_tensor,
            "image_paths": batch_image_paths,
            "scene_attributes": batch_scene_attributes,
        }


class BDD100KDataset(Dataset):
    def __init__(self, dataset_kwargs:dict, dataset_type:str='train',
                perform_augmentation:bool=False, mode:DatasetMode = DatasetMode.TRAIN,
                merge_alt2driv:bool=True):
        
        super(BDD100KDataset, self).__init__()

        assert os.path.exists(dataset_kwargs["images_dir"]), f"Images directory {dataset_kwargs['images_dir']} does not exist."

        self.images_dir = dataset_kwargs["images_dir"]
        self.detection_annotations_dir = dataset_kwargs["detection_annotations_dir"]
        self.segmentation_annotations_dir = dataset_kwargs["segmentation_annotations_dir"]
        self.drivable_annotations_dir = dataset_kwargs["drivable_annotations_dir"]

        self.dataset_type = dataset_type
        self.perform_augmentation = perform_augmentation
        self.mode = mode
        self.merge_alt2driv = merge_alt2driv

        self.preprocessor = BDDPreprocessor(dataset_kwargs.get("preprocessor_kwargs", {}))
        self.image_ids = self.get_image_ids()

        # Advanced augmentation probabilities (only for training)
        aug_config = dataset_kwargs.get("preprocessor_kwargs", {}).get("advanced_aug", {})
        self.mosaic_prob = aug_config.get("mosaic_prob", 0.5) if perform_augmentation else 0.0
        self.mixup_prob = aug_config.get("mixup_prob", 0.15) if perform_augmentation else 0.0
        self.copy_paste_prob = aug_config.get("copy_paste_prob", 0.0) if perform_augmentation else 0.0

    def get_image_ids(self):
        return [f.split('.')[0] for f in os.listdir(os.path.join(self.images_dir, self.dataset_type))]

    def __len__(self):
        return len(self.image_ids)

    def _existing_split_file(self, root_dir, filenames):
        if not root_dir:
            return None

        split_dir = os.path.join(root_dir, self.dataset_type)
        if not os.path.isdir(split_dir):
            return None

        for filename in filenames:
            path = os.path.join(split_dir, filename)
            if os.path.exists(path):
                return path

        return None

    def _load_raw(self, index) -> FrameData:

        image_id = self.image_ids[index]
        image_path = os.path.join(self.images_dir, self.dataset_type, f"{image_id}.jpg")

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image path {image_path} does not exist.")
        
        seg_path = self._existing_split_file(
            self.segmentation_annotations_dir,
            [
                f"{image_id}_train_id.png",
                f"{image_id}_{self.dataset_type}_id.png",
                f"{image_id}.png",
            ],
        )

        drivable_path = self._existing_split_file(
            self.drivable_annotations_dir,
            [f"{image_id}_drivable_id.png", f"{image_id}.png"],
        )

        # Load image
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load masks
        seg = None
        if seg_path is not None:
            seg = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)

        drivable = None
        if drivable_path is not None:
            drivable = cv2.imread(drivable_path, cv2.IMREAD_GRAYSCALE)
            #merge alternative to drivable
            if self.merge_alt2driv:
                drivable[drivable == 2] = 1
        
        det_path = self._existing_split_file(
            self.detection_annotations_dir,
            [f"{image_id}.json"],
        )

        if self.mode == DatasetMode.TRAIN:
            assert det_path is not None, (
                f"Detection path {os.path.join(self.detection_annotations_dir, self.dataset_type, f'{image_id}.json')} "
                "does not exist."
            )

        frame_detections = None
        lane_polys = None
        if det_path is not None and self.mode != DatasetMode.INFER:
            frame_detections = self.preprocessor.load_detection(det_path)
            lane_polys = self.preprocessor.load_lane_annotations(det_path)

        return FrameData(
            image=image,
            image_path=image_path,
            seg=seg,
            drivable=drivable,
            frame_detections=frame_detections,
            lane_polys=lane_polys
        )

    def __getitem__(self, index):
        if self.mode == DatasetMode.INFER:
            image_id = self.image_ids[index]
            image_path = os.path.join(self.images_dir, self.dataset_type, f"{image_id}.jpg")
            return self.preprocessor.prepare_inference(image_path)
        else:
            return self.prepare_training_sample(index)    
        
    def prepare_training_sample(self, index):
        """Load + augment a single training sample.

        All paths converge on a single FrameData carried through the function;
        only the final tensor packaging unpacks it into the dict that
        collate_fn consumes.
        """
        use_mosaic = self.perform_augmentation and random.random() < self.mosaic_prob
        use_mixup = self.perform_augmentation and random.random() < self.mixup_prob
        use_copy_paste = self.perform_augmentation and random.random() < self.copy_paste_prob

        target_size = tuple(self.preprocessor.image_resize)            # (h, w)
        aug_params  = self.preprocessor.augment_params

        # ---- 1. Build / augment the FrameData --------------------------------
        if use_mosaic:
            # Mosaic: combine 4 frames. Output is already at target_size; no
            # further augmentation on top (the mosaic itself supplies geometric
            # diversity via per-tile random scale + center placement).
            indices = [index] + [random.randint(0, len(self) - 1) for _ in range(3)]
            items = [self._load_raw(i) for i in indices]
            frame = mosaic_augmentation(items, output_size=target_size)

        elif use_mixup:
            # MixUp: blend two raw frames. Output stays at raw size — no
            # additional augmentation on top.
            frame1 = self._load_raw(index)
            idx2   = random.randint(0, len(self) - 1)
            frame2 = self._load_raw(idx2)
            frame  = mixup_augmentation(frame1, frame2)

        else:
            # Standard path
            frame = self._load_raw(index)

            if self.perform_augmentation:
                # Optional copy-paste for long-tail classes BEFORE the
                # geometric/color pipeline so pasted instances get augmented too.
                if use_copy_paste:
                    source_idx   = random.randint(0, len(self) - 1)
                    source_frame = self._load_raw(source_idx)
                    frame = copy_paste_instances(
                        frame, source_frame,
                        target_classes=[1, 3],     # RIDER, MOTOR
                        max_instances=2,
                    )

                # apply_augmentations: perspective → HSV → salt/pepper → flip → letterbox
                frame = apply_augmentations(frame, aug_params, img_size=target_size)

            else:
                # No-augment path: just letterbox to target.
                frame = letterbox_with_masks(frame, new_shape=target_size)

        # ---- 2. Build lane targets from the final frame ----------------------
        img_h, img_w = frame.image.shape[:2]
        lane_polys_list = frame.lane_polys_legacy() or []
        lane_targets, lane_categories = build_lane_targets(lane_polys_list, img_h, img_w)
        lane_seg_mask = build_lane_seg_mask(lane_polys_list, img_h, img_w)

        # ---- 3. Tensorize ----------------------------------------------------
        image_tensor = self.preprocessor.normalize_tensor(
            torch.from_numpy(frame.image).permute(2, 0, 1)
        )

        # Detection targets: (N, 5) [cls, cx, cy, w, h] normalized.
        # collate_fn prepends the batch_idx column.
        detection_targets = torch.from_numpy(frame.labels_array()).float()

        seg_tensor = torch.from_numpy(frame.seg).long() if frame.seg is not None else None
        drivable_tensor = torch.from_numpy(frame.drivable).long() if frame.drivable is not None else None

        scene_attributes = (
            frame.frame_detections.attributes
            if frame.frame_detections is not None else {}
        )

        return {
            "image":             image_tensor,
            "segmentation_mask": seg_tensor,
            "drivable_mask":     drivable_tensor,
            "detection_targets": detection_targets,
            "lane_targets":      lane_targets,
            "lane_seg_mask":     lane_seg_mask,
            "lane_categories":   lane_categories,
            "image_path":        frame.image_path,
            "scene_attributes":  scene_attributes,
        }

class FoggyBDDPreprocessor(BDDPreprocessor):

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


    def __init__(self, preprocess_kwargs, 
                apply_fog_prob:float=0.67,
                adverse_params:dict={},
                depth_estimator=None):

        super().__init__(preprocess_kwargs)

        self.adverse_params = adverse_params

        self.fog_betas = self.adverse_params.get("fog_betas", [0.010, 0.020, 0.035])
        self.darkness_gammas = self.adverse_params.get("darkness_gammas", [1.5, 2.0, 3.5])
        self.atmospheric_light_quantile = self.adverse_params.get("atmospheric_light_quantile", 0.9)
        self.atmospheric_light_min_pixels = self.adverse_params.get("atmospheric_light_min_pixels", 10)
        self.atmospheric_light = self.adverse_params.get("atmospheric_light", None)
        self.max_depth_meters = self.adverse_params.get("max_depth_meters", 150.0)

        self.apply_fog_prob = apply_fog_prob

        self.min_haze_level = self.FogLevels.LIGHT
        self.max_haze_level = self.FogLevels.DENSE

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

    def collate_fn(self, batch):
        """
            Batched depth estimation → per-image degradation → tensorize → collate.
        """
        # ----- Single gather pass over the batch -----
        images_rgb        : List[np.ndarray] = []
        image_paths       : List[str]        = []
        scene_attrs       : List[dict]       = []
        segs              : List = []
        drivables         : List = []
        det_targets       : List = []
        lane_targets_list : List = []
        lane_categories   : List = []
        lane_seg_masks    : List = []

        for s in batch:
            img = s["image"]
            assert img is not None, "image is None in batch sample"
            images_rgb.append(img)
            image_paths.append(s["image_path"])
            scene_attrs.append(s["scene_attributes"])
            segs.append(s.get("segmentation_mask"))
            drivables.append(s.get("drivable_mask"))
            det_targets.append(s.get("detection_targets"))
            lane_targets_list.append(s.get("lane_targets"))
            lane_categories.append(s.get("lane_categories"))
            lane_seg_masks.append(s.get("lane_seg_mask"))

        # ----- Batched depth estimation  -----
        depth_maps = self.depth_estimator.estimate_batch(images_rgb)

        # ----- Per-image: degrade → tensorize → record detection rows -----
        clean_tensors    : List[torch.Tensor] = []
        degraded_tensors : List[torch.Tensor] = []
        det_rows         : List[torch.Tensor] = []

        def _to_chw_tensor(img_uint8_rgb):
            # (H, W, 3) uint8 RGB → (3, H, W) float32 in [0, 1]
            return torch.from_numpy(img_uint8_rgb).permute(2, 0, 1).contiguous().float() / 255.0

        for batch_idx, (img, depth, attrs, det) in enumerate(
            zip(images_rgb, depth_maps, scene_attrs, det_targets)):

            beta, gamma = self._select_degradation(attrs)
            if beta is None and gamma is None:
                degraded = img
            else:
                degraded = self._apply_degradation(img, depth, beta, gamma)

            clean_tensors.append(_to_chw_tensor(img))
            degraded_tensors.append(_to_chw_tensor(degraded))

            if det is not None and len(det) > 0:
                det_t  = torch.as_tensor(det, dtype=torch.float32)
                prefix = torch.full((det_t.shape[0], 1), batch_idx, dtype=det_t.dtype)
                det_rows.append(torch.cat([prefix, det_t], dim=1))

        # ----- Stack to batched tensors -----
        batch_images_tensor       = torch.stack(degraded_tensors, dim=0)
        batch_clean_images_tensor = torch.stack(clean_tensors,    dim=0)

        batch_targets_tensor = torch.cat(det_rows, dim=0) if det_rows else None
        if batch_targets_tensor is not None:
            # cx, cy ∈ [0, 1]; w, h ∈ [0.001, 1] — match parent collate_fn sanitization
            batch_targets_tensor[:, 2:4] = batch_targets_tensor[:, 2:4].clamp(0.0, 1.0)
            batch_targets_tensor[:, 4:6] = batch_targets_tensor[:, 4:6].clamp(0.001, 1.0)

        def _stack_optional(items):
            present = [torch.as_tensor(x) for x in items if x is not None]
            return torch.stack(present, dim=0) if present else None

        return {
            "images":             batch_images_tensor,
            "clean_images":       batch_clean_images_tensor,
            "detections":         batch_targets_tensor,
            "segmentation_masks": _stack_optional(segs),
            "drivable_area_seg":  _stack_optional(drivables),
            "lanes_detections":   _stack_optional(lane_targets_list),
            "lane_seg_masks":     _stack_optional(lane_seg_masks),
            "lane_categories":    _stack_optional(lane_categories),
            "image_paths":        image_paths,
            "scene_attributes":   scene_attrs,
        }


class FoggyBDD100KDataset(BDD100KDataset):
    
    def __init__(self, dataset_kwargs, dataset_type = 'train', 
                perform_augmentation = False, 
                mode = DatasetMode.TRAIN, 
                merge_alt2driv = True,
                strict_map:bool=True,
                apply_fog_prob:float=0.67,
                depth_estimator=None
            ):

        super().__init__(dataset_kwargs, dataset_type, perform_augmentation, mode, merge_alt2driv)

        self.preprocessor = self._init_preprocessor(
            preprocessor_kwargs=dataset_kwargs.get("preprocessor_kwargs", {}),
            adverse_params=dataset_kwargs.get("adverse_params", {}),
            apply_fog_prob=apply_fog_prob,
            depth_estimator=depth_estimator
        )

    def _init_preprocessor(self, preprocessor_kwargs:dict, 
                        adverse_params:dict,
                        apply_fog_prob:float, 
                        depth_estimator):
        
        return FoggyBDDPreprocessor(
            preprocess_kwargs=preprocessor_kwargs,
            apply_fog_prob=apply_fog_prob,
            adverse_params=adverse_params,
            depth_estimator=depth_estimator
        )

    def prepare_training_sample(self, index):
        
        frame = self._load_raw(index)

        # Only perform standard augmentation
        if self.perform_augmentation:
            # apply_augmentations: perspective → HSV → salt/pepper → flip → letterbox
            frame = apply_augmentations(
                frame, 
                self.preprocessor.augment_params,
                img_size=tuple(self.preprocessor.image_resize)
            )

        else:
            # No-augment path: just letterbox to target.
            frame = letterbox_with_masks(frame, 
                                        new_shape=tuple(self.preprocessor.image_resize)
                                    )
            
        # ---- 2. Build lane targets from the final frame ----------------------
        img_h, img_w = frame.image.shape[:2]
        lane_polys_list = frame.lane_polys_legacy() or []
        lane_targets, lane_categories = build_lane_targets(lane_polys_list, img_h, img_w)
        lane_seg_mask = build_lane_seg_mask(lane_polys_list, img_h, img_w)

        scene_attributes = (
            frame.frame_detections.attributes
            if frame.frame_detections is not None else {}
        )
        
        # returns 
        # - un-norm image, depth-estimator requires np.array (RGB)
        # np.array: seg, drivable and frame_labels_array

        # perform tensor conversion, depth estimation inside collate_fn
        return {
            "image":             frame.image,
            "segmentation_mask": frame.seg,
            "drivable_mask":     frame.drivable,
            "detection_targets": frame.labels_array(),
            "lane_targets":      lane_targets,
            "lane_seg_mask":     lane_seg_mask,
            "lane_categories":   lane_categories,
            "image_path":        frame.image_path,
            "scene_attributes":  scene_attributes,
        }
