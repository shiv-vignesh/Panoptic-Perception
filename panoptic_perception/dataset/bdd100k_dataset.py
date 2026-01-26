import torch
from torch.utils.data import Dataset

import numpy as np
import os
import json
import cv2
import random
import albumentations as A

from panoptic_perception.dataset.enums import BDD100KClasses, BDD100KClassesReduced
from panoptic_perception.dataset.augmentations import (
    apply_augmentations, copy_paste_instances, mixup_augmentation
)
from panoptic_perception.dataset.mosaic_augmentation import mosaic_augmentation

def visualize_batch(images, seg, drivable, targets, save_dir, batch_index=0):
    """
    images: tensor (B,3,H,W) normalized to [0,1]
    targets: tensor (N,6) -> (batch_idx, class, x_center, y_center, w, h)
    batch_index: index of image in batch to visualize
    """
    img = images[batch_index].permute(1,2,0).cpu().numpy()
    img = (img * 255).astype(np.uint8)

    H, W, _ = img.shape

    # collect targets for this specific image
    t = targets[targets[:,0] == batch_index]

    # convert xywh normalized → pixel PascalVOC (x1,y1,x2,y2)
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

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    cv2.imwrite(f'{save_dir}/sample_batch_{batch_index}.png', img_draw)
    
    if seg is not None:
        if seg[batch_index].ndim == 3:
            seg_vis = seg[batch_index].permute(1, 2, 0).cpu().numpy()
        else:
            seg_vis = seg[batch_index].cpu().numpy()

        # Scale segmentation class IDs for visibility (multiply by factor)
        # BDD100K has ~19 semantic classes, scale to spread across 0-255
        seg_scaled = (seg_vis * 12).clip(0, 255).astype(np.uint8)
        cv2.imwrite(f'{save_dir}/sample_batch_seg_{batch_index}.png', seg_scaled)

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

        # self.transformation = A.Compose(
        #     [
        #         A.LongestMaxSize(max_size=max(self.image_resize)),
        #         A.PadIfNeeded(
        #             min_height=self.resized_height,
        #             min_width=self.resized_width,
        #             border_mode=cv2.BORDER_CONSTANT),
        #         A.CenterCrop(
        #             height=self.resized_height,
        #             width=self.resized_width),
        #     ],
        #     bbox_params=A.BboxParams(format='pascal_voc',
        #                             label_fields=['class_labels'])
        # )

        self.transformation = A.Compose([
                    A.LongestMaxSize(max_size=640),
                    A.PadIfNeeded(640, 640, border_mode=cv2.BORDER_CONSTANT)],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"])
            )

        self.image_only_transformation = A.Compose(
                [A.Resize(height=self.resized_height, width=self.resized_width)]
            )
        

    def load_detection(self, json_path, filter_by_area=False):
        
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

        return bboxes, class_labels
        
    def prepare_targets_2d(self, boxes, labels):
        if len(boxes) == 0:
            return torch.zeros((0, 5), dtype=torch.float32)

        targets = []

        for bbox, class_id in zip(boxes, labels):
            left, top, right, bottom = bbox

            x_center = (left + right) / 2 / self.image_resize[1]
            y_center = (top + bottom) / 2 / self.image_resize[0]
            width = (right - left) / self.image_resize[1]
            height = (bottom - top) / self.image_resize[0]

            targets.append([class_id, x_center, y_center, width, height])

        return torch.tensor(targets)

    def normalize_tensor(self, tensor:torch.Tensor):

        tensor = tensor.float() / 255.0 
        return tensor

    def __call__(self, image_path, seg_path=None, drivable_path=None, det_path=None, augment:bool=False):

        assert os.path.exists(image_path), f"Image path {image_path} does not exist."
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load segmentation mask (original size if augmenting)
        seg = None
        if seg_path is not None:
            if os.path.exists(seg_path):
                seg = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)

        # Load drivable area mask (original size if augmenting)
        drivable = None
        if drivable_path is not None:
            if os.path.exists(drivable_path):
                drivable = cv2.imread(drivable_path, cv2.IMREAD_GRAYSCALE)

        # Load detection annotations
        bboxes = []
        class_labels = []
        if det_path is not None:
            assert os.path.exists(det_path), f"Detection path {det_path} does not exist."
            bboxes, class_labels = self.load_detection(det_path)

        if augment:
            # -------------------------------------------
            # Convert pixel xyxy → normalized xywh
            # -------------------------------------------
            h0, w0 = image.shape[:2]
            labels_xywh = []
            for (x1, y1, x2, y2), cls in zip(bboxes, class_labels):
                cx = ((x1 + x2) / 2) / w0
                cy = ((y1 + y2) / 2) / h0
                bw = (x2 - x1) / w0
                bh = (y2 - y1) / h0
                labels_xywh.append([cls, cx, cy, bw, bh])
            labels_xywh = np.array(labels_xywh) if labels_xywh else np.zeros((0, 5))

            # -------------------------------------------
            # Apply YOLOP augmentations
            # (includes: random_perspective, HSV, flip, letterbox)
            # -------------------------------------------
            image, seg, drivable, labels_xywh = apply_augmentations(
                img=image,
                seg=seg,
                drivable=drivable,
                labels=labels_xywh,
                params=self.augment_params
            )

            # -------------------------------------------
            # Already resized by letterbox, convert to tensor
            # labels_xywh is already in normalized format
            # -------------------------------------------
            targets = torch.tensor(labels_xywh, dtype=torch.float32) if len(labels_xywh) else torch.zeros((0, 5))

            # Convert masks to tensors
            if seg is not None:
                seg_tensor = torch.from_numpy(seg).long()
            else:
                seg_tensor = None

            if drivable is not None:
                drivable_tensor = torch.from_numpy(drivable).long()
            else:
                drivable_tensor = None

            # Convert image to tensor and normalize
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
            image_tensor = self.normalize_tensor(image_tensor)

            return image_tensor, seg_tensor, drivable_tensor, targets

        # -------------------------------------------
        # NO AUGMENTATION: Use albumentations as before
        # -------------------------------------------
        if len(bboxes) > 0:
            transformed_dict = self.transformation(
                image=image, bboxes=bboxes, class_labels=class_labels
            )
            targets = self.prepare_targets_2d(
                transformed_dict['bboxes'],
                transformed_dict['class_labels']
            )
        else:
            transformed_dict = self.image_only_transformation(image=image)
            targets = torch.zeros((0, 5), dtype=torch.float32)

        image = transformed_dict['image']

        # Apply transformation to seg and drivable masks
        if seg is not None:
            seg = self.image_only_transformation(image=seg)['image']
            seg_tensor = torch.from_numpy(seg).long()
        else:
            seg_tensor = None

        if drivable is not None:
            drivable = self.image_only_transformation(image=drivable)['image']
            drivable_tensor = torch.from_numpy(drivable).long()
        else:
            drivable_tensor = None

        # Convert image to tensor and normalize
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        image_tensor = self.normalize_tensor(image_tensor)

        return image_tensor, seg_tensor, drivable_tensor, targets
    
    @staticmethod
    def collate_fn(batch:dict):
        
        batch_images = []
        batch_targets = []
        batch_segmentation_masks = []
        batch_drivable_masks = []
        batch_image_paths = []
        
        for batch_idx, batch_items in enumerate(batch):
            image = batch_items['image']
            image_path = batch_items["image_path"]
            assert image is not None, f"Image tensor at batch index {batch_idx} is None."
            assert image.ndim == 3, "Image tensor must have 3 dimensions (C, H, W)."
            
            batch_images.append(image)
            batch_image_paths.append(image_path)
            
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
                
        batch_images_tensor = torch.stack(batch_images, dim=0)
        batch_targets_tensor = torch.cat(batch_targets, dim=0) if batch_targets else None
        batch_segmentation_masks_tensor = torch.stack(batch_segmentation_masks, dim=0) if batch_segmentation_masks else None
        batch_drivable_masks_tensor = torch.stack(batch_drivable_masks, dim=0) if batch_drivable_masks else None
        
        return {
            "images": batch_images_tensor,
            "detections": batch_targets_tensor,
            "segmentation_masks": batch_segmentation_masks_tensor,
            "drivable_area_seg": batch_drivable_masks_tensor,
            "image_paths":batch_image_paths
        }
    
    def prepare_inference(self, image_path=None):
        
        assert os.path.exists(image_path), f"Image path {image_path} does not exist."
        img = cv2.imread(image_path)
        
        orig = img.copy()
        h0, w0 = img.shape[:2]

        # Your letterbox or resizing logic
        transformed = self.image_only_transformation(image=img)
        img = transformed['image']

        img = torch.from_numpy(img).permute(2,0,1).float() / 255.0

        return {
            "image": img.unsqueeze(0),  # add batch dim
            "original_image": orig,
            "orig_shape": (h0, w0),
            "new_shape": img.shape[1:]
        }

class BDD100KDataset(Dataset):
    def __init__(self, dataset_kwargs:dict, dataset_type:str='train', perform_augmentation:bool=False):
        super(BDD100KDataset, self).__init__()

        assert os.path.exists(dataset_kwargs["images_dir"]), f"Images directory {dataset_kwargs['images_dir']} does not exist."

        self.images_dir = dataset_kwargs["images_dir"]
        self.detection_annotations_dir = dataset_kwargs["detection_annotations_dir"]
        self.segmentation_annotations_dir = dataset_kwargs["segmentation_annotations_dir"]
        self.drivable_annotations_dir = dataset_kwargs["drivable_annotations_dir"]

        self.dataset_type = dataset_type
        self.perform_augmentation = perform_augmentation

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
        det_path = os.path.join(self.detection_annotations_dir, self.dataset_type, f"{image_id}.json")

        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h0, w0 = image.shape[:2]

        # Load masks
        seg = None
        if os.path.exists(seg_path):
            seg = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)

        drivable = None
        if os.path.exists(drivable_path):
            drivable = cv2.imread(drivable_path, cv2.IMREAD_GRAYSCALE)

        # Load detection annotations and convert to normalized xywh
        labels_xywh = np.zeros((0, 5))
        if os.path.exists(det_path):
            bboxes, class_labels = self.preprocessor.load_detection(det_path)
            if len(bboxes) > 0:
                labels_list = []
                for (x1, y1, x2, y2), cls in zip(bboxes, class_labels):
                    cx = ((x1 + x2) / 2) / w0
                    cy = ((y1 + y2) / 2) / h0
                    bw = (x2 - x1) / w0
                    bh = (y2 - y1) / h0
                    labels_list.append([cls, cx, cy, bw, bh])
                labels_xywh = np.array(labels_list)

        return image, labels_xywh, seg, drivable, image_path

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image_path = os.path.join(self.images_dir, self.dataset_type, f"{image_id}.jpg")

        # Check for advanced augmentations (mosaic, mixup, copy-paste)
        use_mosaic = self.perform_augmentation and random.random() < self.mosaic_prob
        use_mixup = self.perform_augmentation and random.random() < self.mixup_prob
        use_copy_paste = self.perform_augmentation and random.random() < self.copy_paste_prob

        if use_mosaic:
            # Mosaic: combine 4 images
            indices = [index] + [random.randint(0, len(self) - 1) for _ in range(3)]
            images_list, labels_list, segs_list, drivables_list = [], [], [], []

            for idx in indices:
                img, labels, seg, drivable, _ = self._load_raw(idx)
                images_list.append(img)
                labels_list.append(labels)
                segs_list.append(seg)
                drivables_list.append(drivable)

            # Apply mosaic
            image, labels_xywh, seg, drivable = mosaic_augmentation(
                images_list, labels_list, segs_list, drivables_list,
                output_size=self.preprocessor.augment_params.get('img_size', (640, 640))
            )

            # Apply remaining augmentations (HSV, flip, etc. but skip letterbox since mosaic outputs correct size)
            from panoptic_perception.dataset.augmentations import augment_hsv, flip_horizontal
            augment_hsv(image,
                       self.preprocessor.augment_params.get("hsv_h", 0.015),
                       self.preprocessor.augment_params.get("hsv_s", 0.7),
                       self.preprocessor.augment_params.get("hsv_v", 0.4))
            if random.random() < 0.5:
                image, seg, drivable, labels_xywh = flip_horizontal(image, seg, drivable, labels_xywh)

        elif use_mixup:
            # MixUp: blend 2 images
            img1, labels1, seg1, drivable1, _ = self._load_raw(index)
            idx2 = random.randint(0, len(self) - 1)
            img2, labels2, seg2, drivable2, _ = self._load_raw(idx2)

            # Resize both to same size first
            from panoptic_perception.dataset.augmentations import letterbox_with_masks
            img_size = self.preprocessor.augment_params.get('img_size', (640, 640))
            img1, seg1, drivable1, labels1 = letterbox_with_masks(img1, seg1, drivable1, labels1, new_shape=img_size)
            img2, seg2, drivable2, labels2 = letterbox_with_masks(img2, seg2, drivable2, labels2, new_shape=img_size)

            # Apply mixup
            image, labels_xywh, seg, drivable = mixup_augmentation(
                img1, labels1, img2, labels2, seg1, seg2, drivable1, drivable2, alpha=0.5
            )

        else:
            # Standard augmentation path
            image, labels_xywh, seg, drivable, _ = self._load_raw(index)

            if self.perform_augmentation:
                # Apply copy-paste for long-tail classes before other augmentations
                if use_copy_paste:
                    source_idx = random.randint(0, len(self) - 1)
                    source_img, source_labels, _, _, _ = self._load_raw(source_idx)
                    # Resize source to match
                    h, w = image.shape[:2]
                    source_img = cv2.resize(source_img, (w, h))
                    image, labels_xywh = copy_paste_instances(
                        image, labels_xywh, source_img, source_labels,
                        target_classes=[1, 3],  # RIDER=1, MOTOR=3
                        max_instances=2
                    )

                # Apply standard YOLOP augmentations
                from panoptic_perception.dataset.augmentations import apply_augmentations
                image, seg, drivable, labels_xywh = apply_augmentations(
                    img=image, seg=seg, drivable=drivable,
                    labels=labels_xywh, params=self.preprocessor.augment_params
                )
            else:
                # No augmentation - just resize
                seg_path = os.path.join(self.segmentation_annotations_dir, self.dataset_type, f"{image_id}_train_id.png")
                drivable_path = os.path.join(self.drivable_annotations_dir, self.dataset_type, f"{image_id}_drivable_id.png")
                det_path = os.path.join(self.detection_annotations_dir, self.dataset_type, f"{image_id}.json")

                image_tensor, seg_tensor, drivable_tensor, targets = self.preprocessor(
                    image_path=image_path, seg_path=seg_path,
                    drivable_path=drivable_path, det_path=det_path, augment=False
                )
                return {
                    "image": image_tensor,
                    "segmentation_mask": seg_tensor,
                    "drivable_mask": drivable_tensor,
                    "detection_targets": targets,
                    "image_path": image_path
                }

        # Convert to tensors
        targets = torch.tensor(labels_xywh, dtype=torch.float32) if len(labels_xywh) else torch.zeros((0, 5))
        seg_tensor = torch.from_numpy(seg).long() if seg is not None else None
        drivable_tensor = torch.from_numpy(drivable).long() if drivable is not None else None
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return {
            "image": image_tensor,
            "segmentation_mask": seg_tensor,
            "drivable_mask": drivable_tensor,
            "detection_targets": targets,
            "image_path": image_path
        }
    
