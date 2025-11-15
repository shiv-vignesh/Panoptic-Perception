import torch
from torch.utils.data import Dataset

import os
import json
import cv2
import albumentations as A

from panoptic_perception.dataset.enums import BDD100KClasses

class BDDPreprocessor:
    def __init__(self, preprocess_kwargs:dict):
        self.preprocess_kwargs = preprocess_kwargs
        
        self.image_resize = preprocess_kwargs.get("image_resize", (640, 640))
        self.original_size = preprocess_kwargs.get("original_image_size", (720, 1280))
        
        self.original_width = self.original_size[1]
        self.original_height = self.original_size[0]
        
        self.resized_width = self.image_resize[1]
        self.resized_height = self.image_resize[0]
        
        self.transformation = A.Compose(
            [
                A.LongestMaxSize(max_size=max(self.image_resize)),
                A.PadIfNeeded(
                    min_height=self.resized_height,
                    min_width=self.resized_width,
                    border_mode=cv2.BORDER_CONSTANT),
                A.CenterCrop(
                    height=self.resized_height,
                    width=self.resized_width),
            ],
            bbox_params=A.BboxParams(format='pascal_voc',
                                    label_fields=['class_labels'])
        )

        self.image_only_transformation = A.Compose(
                [A.Resize(height=self.resized_height, width=self.resized_width)]
            )
        

    def load_detection(self, json_path):
        
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        bboxes = []
        class_labels = []
        
        for frames in data["frames"]:
            for item in frames["objects"]:
                if "box2d" in item:
                    box = item["box2d"]
                    label = item["category"]
                    label = '_'.join(label.split())
                    label_id = BDD100KClasses.from_label(label)
                    
                    if box["x2"] > box["x1"] and box["y2"] > box["y1"]: 
                        bboxes.append([box["x1"], box["y1"], box["x2"], box["y2"]])
                        class_labels.append(label_id)

        return bboxes, class_labels
        
    def prepare_targets_2d(self, boxes, labels):
        
        targets = []

        for bbox, class_id in zip(boxes, labels):
            left, top, right, bottom = bbox

            x_center = (left + right) / 2 / self.image_resize[1]
            y_center = (top + bottom) / 2 / self.image_resize[0]
            width = (right - left) / self.image_resize[1]
            height = (bottom - top) / self.image_resize[0]

            targets.append([class_id, x_center, y_center, width, height])

        return torch.tensor(targets)

    def __call__(self, image_path, seg_path=None, drivable_path=None, det_path=None):
        
        assert os.path.exists(image_path), f"Image path {image_path} does not exist."
        image = cv2.imread(image_path)
        
        if seg_path is not None:
            # assert os.path.exists(seg_path), f"Segmentation path {seg_path} does not exist." #(only 10K subset)
            if not os.path.exists(seg_path):
                seg_tensor = None
            else:
                seg = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
                seg = self.image_only_transformation(image=seg)['image']
                seg_tensor = torch.from_numpy(seg).long()
        else:
            seg_tensor = None
        
        if drivable_path is not None:
            assert os.path.exists(drivable_path), f"Drivable path {drivable_path} does not exist."
            drivable = cv2.imread(drivable_path, cv2.IMREAD_GRAYSCALE)
            drivable = self.image_only_transformation(image=drivable)['image']
            drivable_tensor = torch.from_numpy(drivable).long()
        else:
            drivable_tensor = None
            
        if det_path is not None:
            assert os.path.exists(det_path), f"Detection path {det_path} does not exist."
            # Load detection annotations (implementation depends on annotation format)

            bboxes, class_labels = self.load_detection(det_path)
                        
            transformed_dict = self.transformation(
                image=image, bboxes=bboxes, class_labels=class_labels
            )

            targets = self.prepare_targets_2d(
                transformed_dict['bboxes'], 
                transformed_dict['class_labels']
            )                        

        else:
            transformed_dict = self.image_only_transformation(image=image)
            targets = None
            
        image = transformed_dict['image']
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()

        return image_tensor, seg_tensor, drivable_tensor, targets
    
    @staticmethod
    def collate_fn(batch:dict):
        
        batch_images = []
        batch_targets = []
        batch_segmentation_masks = []
        batch_drivable_masks = []
        
        for batch_idx, batch_items in enumerate(batch):
            image = batch_items['image']
            assert image is not None, f"Image tensor at batch index {batch_idx} is None."
            assert image.ndim == 3, "Image tensor must have 3 dimensions (C, H, W)."
            
            batch_images.append(image)
            
            if batch_items['detection_targets'] is not None:
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
            "drivable_area_seg": batch_drivable_masks_tensor
        }
            

class BDD100KDataset(Dataset):
    def __init__(self, dataset_kwargs:dict, dataset_type:str='train'):
        super(BDD100KDataset, self).__init__()
        
        assert os.path.exists(dataset_kwargs["images_dir"]), f"Images directory {dataset_kwargs['images_dir']} does not exist."
        
        self.images_dir = dataset_kwargs["images_dir"]
        self.detection_annotations_dir = dataset_kwargs["detection_annotations_dir"]
        self.segmentation_annotations_dir = dataset_kwargs["segmentation_annotations_dir"]
        self.drivable_annotations_dir = dataset_kwargs["drivable_annotations_dir"]
        
        self.dataset_type = dataset_type
        
        self.preprocessor = BDDPreprocessor(dataset_kwargs.get("preprocessor_kwargs", {}))
        self.image_ids = self.get_image_ids()
        
    def get_image_ids(self):
        return [f.split('.')[0] for f in os.listdir(os.path.join(self.images_dir, self.dataset_type))]

    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, index):
        image_id = self.image_ids[index]
        
        image_path = os.path.join(self.images_dir, self.dataset_type, f"{image_id}.jpg")
        seg_path = os.path.join(self.segmentation_annotations_dir, self.dataset_type, f"{image_id}_train_id.png")
        drivable_path = os.path.join(self.drivable_annotations_dir, self.dataset_type, f"{image_id}_drivable_id.png")
        det_path = os.path.join(self.detection_annotations_dir, self.dataset_type, f"{image_id}.json")
        
        image_tensor, seg_tensor, drivable_tensor, targets = self.preprocessor(
            image_path=image_path,
            seg_path=seg_path,
            drivable_path=drivable_path,
            det_path=det_path
        )
        
        sample = {
            "image": image_tensor,
            "segmentation_mask": seg_tensor,
            "drivable_mask": drivable_tensor,
            "detection_targets": targets
        }
        
        return sample