from typing import List, Optional
import torch
import torchvision
import numpy as np

class DetectionHelper:
    
    @staticmethod    
    def bbox_iou(box1:torch.Tensor, box2:torch.Tensor, x1y1x2y2:bool=True, 
                CIoU:bool=False, GIoU:bool=False, DIoU:bool=False, eps:float=1e-7):
        
        if not x1y1x2y2:
            # Convert from (x, y, w, h) to (x1, y1, x2, y2)
            # x1 = x - w/2
            # x2 = x + w/2
            
            b1_x1, b1_x2 = box1[:, 0] - box1[:, 2]/2 , box1[:, 0] + box1[:, 2]/2
            b1_y1, b1_y2 = box1[:, 1] - box1[:, 3]/2 , box1[:, 1] + box1[:, 3]/2
            b2_x1, b2_x2 = box2[:, 0] - box2[:, 2]/2 , box2[:, 0] + box2[:, 2]/2
            b2_y1, b2_y2 = box2[:, 1] - box2[:, 3]/2 , box2[:, 1] + box2[:, 3]/2        
        
        else:
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
            
        # Intersection coordinates
        inter_x1 = torch.max(b1_x1, b2_x1)
        inter_y1 = torch.max(b1_y1, b2_y1)
        inter_x2 = torch.min(b1_x2, b2_x2)
        inter_y2 = torch.min(b1_y2, b2_y2)
        
        # Intersection area
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        # Union area
        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        
        union_area = b1_area + b2_area - inter_area + eps
        iou = inter_area / union_area
        
        if CIoU or GIoU or DIoU: # Advanced IoU calculations
            cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex width
            ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
            
            # c_union = b1_area + b2_area - inter_area + eps  # union area
            w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
            w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
            
            if DIoU or CIoU:
                c2 = cw ** 2 + ch ** 2 + eps
                rho2 = (( (b2_x1 + b2_x2)/2 - (b1_x1 + b1_x2)/2 ) ** 2 + 
                        ( (b2_y1 + b2_y2)/2 - (b1_y1 + b1_y2)/2 ) ** 2) / 4 # center distance squared
                if DIoU:
                    return iou - rho2 / c2
                elif CIoU:
                    # Clamp aspect ratios to prevent gradient explosion through atan when w or h ≈ 0
                    v = (4/ (torch.pi ** 2)) * torch.pow(torch.atan(torch.clamp(w2 / h2, -1e4, 1e4)) - torch.atan(torch.clamp(w1 / h1, -1e4, 1e4)), 2)
                    with torch.no_grad():
                        alpha = v / (1 - iou + v + eps)
                    return iou - (rho2 / c2 + v * alpha)
            elif GIoU:
                c_area = cw * ch + eps
                return iou - (c_area - union_area) / c_area
            
        return iou
    
    @staticmethod
    def bbox_iou_pairwise(box1, box2, eps=1e-7):
        """
        Pairwise IoU between two sets of boxes (both in x1y1x2y2 format).
    
        box1: [M, 4]
        box2: [N, 4]
        Returns: [M, N] IoU matrix
        """
                
        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        
        inter_x1 = torch.max(box1[:, None, 0], box2[None, :, 0])
        inter_y1 = torch.max(box1[:, None, 1], box2[None, :, 1])
        inter_x2 = torch.min(box1[:, None, 2], box2[None, :, 2])
        inter_y2 = torch.min(box1[:, None, 3], box2[None, :, 3])
        
        inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
        union_area = area1[:, None] + area2[None, :] - inter_area + eps
        
        return inter_area/union_area
    
    @staticmethod
    def align_scores_pairwise(pred_logits:torch.Tensor, gt_labels:torch.Tensor,
                              iou_pairwise:torch.Tensor, is_inside_mask:torch.Tensor,
                              alpha:float = 0.5, beta:float = 6.0):
        """
        Gather the predicted score for each GT's actual class
        For each prediction-GT pair, get P(correct_class)
        
        pred_scores (raw logits) : [M, C], M=8400
        gt_labels: [N]
        iou_pairwise (result from bbox_iou_pairwise) : [M, N]
        """
        
        pred_scores = pred_logits.sigmoid()
        align_scores = pred_scores[:, gt_labels.long()] #[M, N]
        
        assert align_scores.shape == iou_pairwise.shape, \
            f"Expected Class Align Tensor and IOU Pairwise to have same shape, got: {align_scores.shape} and {iou_pairwise.shape}"
        
        aligment_metric = (align_scores ** alpha) * (iou_pairwise ** beta)
        return aligment_metric * is_inside_mask
    
    @staticmethod
    def non_max_suppression(
        predictions: torch.Tensor,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_detections: int = 300
    ) -> List[torch.Tensor]:
        """
        Perform Non-Maximum Suppression (NMS) on detection predictions.

        Args:
            predictions: Tensor of shape (batch_size, num_boxes, 5+num_classes)
                        [x, y, w, h, objectness, class_scores...]
            conf_threshold: Minimum confidence threshold
            iou_threshold: IoU threshold for NMS
            max_detections: Maximum number of detections to keep

        Returns:
            List of tensors, one per image, shape (num_detections, 6)
            [x1, y1, x2, y2, confidence, class_id]
        """
        batch_size = predictions.shape[0]
        num_classes = predictions.shape[2] - 5
        output = [None] * batch_size

        for img_idx in range(batch_size):
            pred = predictions[img_idx]  # (num_boxes, 5+num_classes)

            if pred.shape[0] == 0:
                continue

            # Multiply class scores by objectness to get final confidence
            pred[:, 5:] *= pred[:, 4:5]

            # Get max confidence across all classes
            class_conf, class_pred = pred[:, 5:].max(1, keepdim=True)

            # Filter by confidence threshold
            conf_mask = class_conf[:, 0] > conf_threshold
            pred = pred[conf_mask]
            class_conf = class_conf[conf_mask]
            class_pred = class_pred[conf_mask]

            # Convert from (x, y, w, h) to (x1, y1, x2, y2)
            boxes = DetectionHelper.xywh2xyxy(pred[:, :4])
            
            detections = torch.cat([boxes, class_conf, class_pred.float()], 1)

            # Perform NMS per class
            unique_classes = detections[:, -1].unique()
            keep_boxes = []

            for cls in unique_classes:
                cls_mask = detections[:, -1] == cls
                cls_detections = detections[cls_mask]

                # Sort by confidence (column 4 - already final confidence)
                conf_sort_idx = torch.argsort(cls_detections[:, 4], descending=True)
                cls_detections = cls_detections[conf_sort_idx]

                # NMS using final confidence (column 4 - no multiply!)
                keep_idx = DetectionHelper.nms_boxes(
                    cls_detections[:, :4],
                    cls_detections[:, 4],  # Just column 4, NOT [:, 4] * [:, 5]
                    iou_threshold
                )

                keep_boxes.append(cls_detections[keep_idx])

            if keep_boxes:
                all_dets = torch.cat(keep_boxes, 0)
                if max_detections > 0:
                    all_dets = all_dets[all_dets[:, 4].argsort(descending=True)][:max_detections]
                output[img_idx] = all_dets

        return output

    @staticmethod
    def nms_boxes(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> List[int]:
        """
        Perform NMS on boxes with given scores using optimized torchvision implementation.

        Args:
            boxes: (N, 4) tensor [x1, y1, x2, y2]
            scores: (N,) tensor of confidence scores
            iou_threshold: IoU threshold

        Returns:
            List of indices to keep
        """
        # Use torchvision's optimized NMS (GPU-accelerated, C++ backend)
        keep_indices = torchvision.ops.nms(boxes, scores, iou_threshold)
        return keep_indices.tolist()

    @staticmethod
    def non_max_suppression_v8(predictions:torch.Tensor, 
                               conf_threshold:float = 0.25, 
                               iou_threshold:float = 0.45,
                               max_detections: int = 300) -> List[torch.Tensor]:

        """        
        Perform Non-Maximum Suppression (NMS) on anchor free detection predictions.
        
        Args:
            predictions: Tensor of shape (batch_size, num_boxes = 8400, 4+num_classes)
                        [x, y, w, h, class_scores...]
                        class_scores : sigmoid activated
                        
            conf_threshold: Minimum confidence threshold
            iou_threshold: IoU threshold for NMS
            max_detections: Maximum number of detections to keep

        Returns:
            List of tensors, one per image, shape (num_detections, 6)
            [x1, y1, x2, y2, class_id]        
        
        """
        
        batch_size = predictions.shape[0]
        num_classes = predictions.shape[2] - 4
        output = [None] * batch_size
        
        for img_idx in range(batch_size):
            pred = predictions[img_idx] #(num_boxes = 8400, 4+nc)
            
            if pred.shape[0] == 0:
                continue

            boxes, scores = pred[:, :4], pred[:, 4:]
            boxes = DetectionHelper.xywh2xyxy(boxes)

            # (8400,) (8400,)
            conf, cls = scores.max(dim=1) 
            mask = conf > conf_threshold

            boxes = boxes[mask] #(num_dets, 4) where num_dets <= 8400 
            conf = conf[mask] #(num_dets,) where num_dets <= 8400
            cls = cls[mask] #(num_dets,) where num_dets <= 8400

            detections = torch.cat([boxes, conf.unsqueeze(-1), cls.unsqueeze(-1).float()], dim=1)

            unique_classes = detections[:, -1].unique()
            keep_boxes = []

            for c in unique_classes:
                cls_mask = detections[:, -1] == c
                cls_detections = detections[cls_mask]

                # Sort by confidence (column 4 - already final confidence)
                conf_sort_idx = torch.argsort(cls_detections[:, 4], descending=True)
                cls_detections = cls_detections[conf_sort_idx]

                keep_idx = DetectionHelper.nms_boxes(
                    cls_detections[:, :4],
                    cls_detections[:, 4],
                    iou_threshold
                )

                keep_boxes.append(cls_detections[keep_idx])

            if keep_boxes:
                all_dets = torch.cat(keep_boxes, 0)
                if max_detections > 0:
                    all_dets = all_dets[all_dets[:, 4].argsort(descending=True)][:max_detections]

                output[img_idx] = all_dets

        return output            

    @staticmethod
    def box_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
        """
        Compute IoU between two sets of boxes (simple version for NMS).

        Args:
            box1: (N, 4) [x1, y1, x2, y2]
            box2: (M, 4) [x1, y1, x2, y2]

        Returns:
            (N, M) IoU matrix
        """
        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

        # Intersection
        lt = torch.max(box1[:, None, :2], box2[:, :2])  # (N, M, 2)
        rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # (N, M, 2)

        wh = (rb - lt).clamp(min=0)  # (N, M, 2)
        inter = wh[:, :, 0] * wh[:, :, 1]  # (N, M)

        # Union
        union = area1[:, None] + area2 - inter

        iou = inter / (union + 1e-6)
        return iou
    
    @staticmethod
    def xywh2xyxy(boxes:torch.Tensor):
        """
        Convert bounding boxes from (x_center, y_center, width, height) to (x1, y1, x2, y2)
        boxes: Tensor of shape (..., 4)
        Returns: Tensor of same shape as input
        """
        x_c, y_c, w, h = boxes.unbind(-1)
        x1 = x_c - w / 2
        y1 = y_c - h / 2
        x2 = x_c + w / 2
        y2 = y_c + h / 2
        return torch.stack((x1, y1, x2, y2), dim=-1)
    
    @staticmethod
    def xyxy2xywh(boxes:torch.Tensor):
        """
        Convert bounding boxes from (x1, y1, x2, y2) to (x_center, y_center, width, height)
        boxes: Tensor of shape (..., 4)
        Returns: Tensor of same shape as input
        """
        x1, y1, x2, y2 = boxes.unbind(-1)
        x_c = (x1 + x2) / 2
        y_c = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        return torch.stack((x_c, y_c, w, h), dim=-1)

    @staticmethod
    def visualize_detections(
        image: torch.Tensor,
        predictions: Optional[torch.Tensor],
        targets: Optional[torch.Tensor],
        class_names: Optional[List[str]] = None,
        pred_color: tuple = (0, 255, 0),  # Green for predictions
        target_color: tuple = (255, 0, 0),  # Red for targets
        thickness: int = 1,
        font_scale: float = 0.2,
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Visualize detection predictions and ground truth on an image.

        Args:
            image: Tensor of shape (C, H, W) or (H, W, C), values in [0, 1] or [0, 255]
            predictions: Tensor of shape (N, 6) [x1, y1, x2, y2, confidence, class_id]
                        or None if no predictions
            targets: Tensor of shape (M, 5) [x1, y1, x2, y2, class_id]
                    or None if no targets
            class_names: List of class names for labeling
            pred_color: BGR color for prediction boxes (green by default)
            target_color: BGR color for target boxes (red by default)
            thickness: Line thickness for boxes
            font_scale: Font scale for labels
            save_path: If provided, save the image to this path

        Returns:
            Annotated image as numpy array (H, W, C) in BGR format
        """
        import cv2

        # Convert image to numpy array
        if torch.is_tensor(image):
            image = image.detach().cpu().numpy()

        # Handle channel dimension
        if image.shape[0] == 3:  # (C, H, W) -> (H, W, C)
            image = np.transpose(image, (1, 2, 0))

        # Convert to uint8 if needed
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

        # Convert RGB to BGR for OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Make a copy to draw on
        vis_image = image.copy()

        # Draw target boxes (red) first so predictions overlay them
        if targets is not None and len(targets) > 0:
            if torch.is_tensor(targets):
                targets = targets.detach().cpu().numpy()

            for box in targets:
                x1, y1, x2, y2 = box[:4].astype(int)
                class_id = int(box[4]) if len(box) > 4 else -1

                # Draw box
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), target_color, thickness)

                # Draw label
                if class_names is not None and 0 <= class_id < len(class_names):
                    label = f"GT: {class_names[class_id]}"
                else:
                    label = f"GT: {class_id}"

                label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
                y1_label = max(y1, label_size[1] + 5)
                cv2.rectangle(vis_image, (x1, y1_label - label_size[1] - 5),
                            (x1 + label_size[0], y1_label), target_color, -1)
                cv2.putText(vis_image, label, (x1, y1_label - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)

        # Draw prediction boxes (green)
        if predictions is not None and len(predictions) > 0:
            if torch.is_tensor(predictions):
                predictions = predictions.detach().cpu().numpy()

            for box in predictions:
                x1, y1, x2, y2 = box[:4].astype(int)
                confidence = box[4] if len(box) > 4 else 0.0
                class_id = int(box[5]) if len(box) > 5 else -1

                # Draw box
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), pred_color, thickness)

                # Draw label with confidence
                if class_names is not None and 0 <= class_id < len(class_names):
                    label = f"{class_names[class_id]}: {confidence:.2f}"
                else:
                    label = f"{class_id}: {confidence:.2f}"

                label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
                y2_label = min(y2 + label_size[1] + 10, vis_image.shape[0])
                cv2.rectangle(vis_image, (x1, y2),
                            (x1 + label_size[0], y2_label), pred_color, -1)
                cv2.putText(vis_image, label, (x1, y2 + label_size[1] + 2),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)

        # Save if path provided
        if save_path is not None:
            cv2.imwrite(save_path, vis_image)

        return vis_image

class DetectionLossCalculator:
    
    
    bbox_weight: float = 0.05
    obj_weight: float = 1.0
    cls_weight: float = 0.5
    
    balance:list = [4.0, 1.0, 0.4]
    gamma:float=2.0
    class_weights = None  # Not needed: detection loss is computed on matched targets which are already balanced
    
    iou_aware_cls:bool=False
    label_smoothing:float=0.0 
    autobalance:bool=False
    
    ssi = [1.0, 1.0, 1.0]

    @staticmethod
    def focal_loss(inputs, targets, alpha=0.25, gamma=2.0, reduction="mean"):
        """
        Compute focal loss for binary classification.
        Inputs:
            inputs: raw logits, tensor of shape (N, *).
            targets: binary targets (0 or 1), same shape as inputs.
            alpha: balancing factor.
            gamma: focusing parameter.
            reduction: 'mean', 'sum', or 'none'.
        Returns:
            scalar loss or per-element loss if reduction='none'.
        """
        # Compute binary cross-entropy loss with logits (no reduction)
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(inputs, targets,
                                                                    reduction="none")
        # Get the probability of the true class
        pt = torch.exp(-bce_loss)
        loss = alpha * (1 - pt) ** gamma * bce_loss
        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        elif reduction == "none":
            return loss
        else:
            return loss   

    @staticmethod
    def build_targets_2(preds:List[torch.Tensor], targets:torch.Tensor, 
                        num_layers:int, anchors:torch.Tensor, stride:torch.Tensor):
        device = targets.device
        # num_layers = detect_module.num_layers
        tcls, tbox, indices, anch = [], [], [], []

        nt = targets.shape[0]  # total number of targets
        if nt == 0:
            # return empty lists for all layers
            for i in range(num_layers):
                tcls.append(torch.zeros((0,), dtype=torch.long, device=device))
                tbox.append(torch.zeros((0,4), device=device))
                indices.append((torch.tensor([], device=device),
                                torch.tensor([], device=device),
                                torch.tensor([], device=device),
                                torch.tensor([], device=device)))
                anch.append(torch.zeros((0,2), device=device))
            return tcls, tbox, indices, anch

        na = anchors.shape[1]  # number of anchors per layer
        # repeat targets for each anchor
        ai = torch.arange(na, device=device).view(1, na).repeat(nt,1).T  # shape (na, nt)
        targets = torch.cat((targets.repeat(na,1,1), ai[:,:,None]), dim=2)  # (na, nt, 7)

        for i in range(num_layers):
            # preds[i].shape = (B, A, ny, nx, 5+nc)
            bs, A, ny, nx, nc = preds[i].shape
            anchors_grid = anchors[i].to(device) / stride[i]  # scale anchors to grid

            # scale normalized targets to grid coordinates
            gain = torch.tensor([1,1,nx,ny,nx,ny,1], device=device, dtype=targets.dtype)
            t = targets * gain

            # filter targets by anchor ratio
            r = t[:,:,4:6] / anchors_grid[:, None]  # shape (na, nt, 2)
            j = torch.max(r, 1./r).max(2)[0] < 4.0
            t = t[j]

            if t.shape[0]:
                b, c = t[:, :2].long().T
                gxy = t[:, 2:4]  # x,y in grid units
                gwh = t[:, 4:6]  # w,h in grid units
                gij = gxy.long()
                gi, gj = gij.T
                a = t[:, 6].long()

                indices.append((b, a, gj.clamp(0, ny-1), gi.clamp(0, nx-1)))
                tbox.append(torch.cat((gxy - gij, gwh), dim=1))
                anch.append(anchors_grid[a])
                tcls.append(c.clamp(0, nc-5-1))  # class index safe
            else:
                tcls.append(torch.zeros((0,), dtype=torch.long, device=device))
                tbox.append(torch.zeros((0,4), device=device))
                indices.append((torch.tensor([], device=device),
                                torch.tensor([], device=device),
                                torch.tensor([], device=device),
                                torch.tensor([], device=device)))
                anch.append(torch.zeros((0,2), device=device))

        return tcls, tbox, indices, anch

    @staticmethod
    def compute_detection_loss_2(outputs:List[torch.Tensor], targets:torch.Tensor, 
                                num_layers:int, anchors:torch.Tensor, stride:torch.Tensor,
                                cls_loss_type:str='BCE'):
        device = targets.device
        lcls = torch.zeros(1, device=device)
        lbox = torch.zeros(1, device=device)
        lobj = torch.zeros(1, device=device)

        # Build targets
        tcls, tbox, indices, anch_per_target = DetectionLossCalculator.build_targets_2(outputs, targets, 
                                                                                    num_layers, anchors, stride)

        ious = []

        for i, pred in enumerate(outputs):
            b, a, gj, gi = indices[i]
            tobj = torch.zeros_like(pred[..., 4], device=device)  # objectness target

            nt = b.shape[0]  # number of targets in this layer
            if nt:
                ps = pred[b, a, gj, gi]  # (nt, 5+nc)

                # boxes in grid space - YOLOP style decoding
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2.) ** 2 * anch_per_target[i]
                pbox = torch.cat((pxy, pwh), dim=1)

                # Compute IoU with targets
                iou = DetectionHelper.bbox_iou(pbox, tbox[i], x1y1x2y2=False, CIoU=True)
                ious.append(iou.mean().item())
                lbox += (1.0 - iou).mean()

                # Objectness targets = IoU
                tobj[b, a, gj, gi] = iou.detach()

                # Classification loss (IoU-aware weighting)
                nc = pred.shape[-1] - 5
                if nc > 1:
                    
                    cp = 1.0 - 0.5 * DetectionLossCalculator.label_smoothing
                    cn = 0.5 * DetectionLossCalculator.label_smoothing                    
                    
                    t = torch.full_like(ps[:, 5:], device=device, fill_value=cn)
                    t[range(nt), tcls[i]] = cp
                    t = t.detach()

                    # IoU-aware weighting: weight classification loss by prediction quality
                    iou_weight = iou.detach().clamp(0.1, 1.0)  # Clamp to avoid zero weights

                    if cls_loss_type.lower() == 'focal':
                        cls_loss_per_sample = DetectionLossCalculator.focal_loss(ps[:, 5:], t, alpha=0.25, 
                                                                                gamma=DetectionLossCalculator.gamma, 
                                                                                reduction='none')
                        # Weight by IoU and take mean
                        if DetectionLossCalculator.iou_aware_cls:
                            lcls += (cls_loss_per_sample.mean(dim=1) * iou_weight).mean()
                        else:
                            lcls += cls_loss_per_sample.mean()
                    else:  # BCE
                        cls_loss_per_sample = torch.nn.functional.binary_cross_entropy_with_logits(ps[:, 5:],
                                                                                                   t,
                                                                                                   reduction='none')
                        # Weight by IoU and take mean
                        if DetectionLossCalculator.iou_aware_cls:
                            lcls += (cls_loss_per_sample.mean(dim=1) * iou_weight).mean()
                        else:
                            lcls += cls_loss_per_sample.mean()

            # Objectness loss
            obji = torch.nn.functional.binary_cross_entropy_with_logits(pred[..., 4], 
                                                                        tobj, 
                                                                        reduction='mean')
            
            if DetectionLossCalculator.autobalance:
                obji_val = obji.detach().item()
                DetectionLossCalculator.ssi[i] = \
                    DetectionLossCalculator.ssi[i] * 0.9999 + obji_val * 0.0001
                
                ssi_max = max(DetectionLossCalculator.ssi)
                balance_i = (DetectionLossCalculator.ssi[i] / (ssi_max + 1e-9))
                lobj += obji * balance_i
                    
            else:
                lobj += obji * DetectionLossCalculator.balance[i]

        # YOLOv3 loss weights
        lbox *= DetectionLossCalculator.bbox_weight
        lobj *= DetectionLossCalculator.obj_weight
        lcls *= DetectionLossCalculator.cls_weight

        loss = lbox + lobj + lcls
        return loss, {"lbox": lbox, "lobj": lobj, "lcls": lcls, "iou": sum(ious)/len(ious) if ious else 0.0}

    @staticmethod
    def compute_detection_loss_v8(pred_scores_logits:torch.Tensor, pred_distri_logits:torch.Tensor,
                                  anchor_points:torch.Tensor, strides:torch.Tensor, 
                                  targets:torch.Tensor, image_size:tuple, 
                                  xywh2xyxy:bool=True, topk:int=10):
        """
        pred_scores - (bs, 8400, C)
        pred_distri - (bs, 8400, 64)
        anchor_points - (8400, 2)
        strides - (8400, 1)
        """
        
        img_h, img_w = image_size
        device = pred_distri_logits.device

        bs, num_dets, num_classes = pred_scores_logits.shape
        reg_max = pred_distri_logits.shape[-1] // 4 # 64 // 4 = 16
        pred_distri_logits = pred_distri_logits.view(bs, num_dets, 4, reg_max)
        pred_distri = torch.softmax(pred_distri_logits, dim=-1)

        project = torch.arange(pred_distri.shape[-1], dtype=pred_distri.dtype, device=device)
        pred_ltrb = (pred_distri * project).sum(dim=-1)

        anchor_x = anchor_points[:, 0].unsqueeze(0)
        anchor_y = anchor_points[:, 1].unsqueeze(0)
        
        strides_orig = strides
        strides = strides.unsqueeze(0)

        # Convert to xyxy in pixel coords:
        x1 = anchor_x - pred_ltrb[:, :, 0] * strides.squeeze(-1)
        y1 = anchor_y - pred_ltrb[:, :, 1] * strides.squeeze(-1)
        x2 = anchor_x + pred_ltrb[:, :, 2] * strides.squeeze(-1)
        y2 = anchor_y + pred_ltrb[:, :, 3] * strides.squeeze(-1)

        pred_bboxes = torch.stack([x1, y1, x2, y2], dim=-1)
        
        lcls = torch.zeros(1, device=device)
        lbox = torch.zeros(1, device=device)
        ldfl = torch.zeros(1, device=device)

        for b in range(bs):
            gt_bboxes = targets[targets[:, 0] == b][:, 2:]
            gt_labels = targets[targets[:, 0] == b][:, 1]
            pbox = pred_bboxes[b]
            plabels = pred_scores_logits[b]

            if not gt_bboxes.shape[0]:
                continue

            if xywh2xyxy:
                gt_x1 = (gt_bboxes[:, 0] - gt_bboxes[:, 2] / 2) * img_w
                gt_y1 = (gt_bboxes[:, 1] - gt_bboxes[:, 3] / 2) * img_h
                gt_x2 = (gt_bboxes[:, 0] + gt_bboxes[:, 2] / 2) * img_w
                gt_y2 = (gt_bboxes[:, 1] + gt_bboxes[:, 3] / 2) * img_h
                gt_bboxes = torch.stack([gt_x1, gt_y1, gt_x2, gt_y2], dim=-1)

            is_inside = (
                (anchor_points[:, 0:1] > gt_bboxes[None, :, 0]) &
                (anchor_points[:, 1:2] > gt_bboxes[None, :, 1]) &
                (anchor_points[:, 0:1] < gt_bboxes[None, :, 2]) &
                (anchor_points[:, 1:2] < gt_bboxes[None, :, 3]) 
            )

            iou = DetectionHelper.bbox_iou_pairwise(pbox, gt_bboxes)
            align_scores = DetectionHelper.align_scores_pairwise(
                plabels, gt_labels, iou, is_inside
            )

            # For each GT (column), get top-k cells by alignment metric 
            topk_metrics, topk_indices = torch.topk(align_scores, dim=0, k=topk)

            topk_mask = torch.zeros_like(align_scores, dtype=torch.bool, device=device)
            topk_mask.scatter_(0, topk_indices, True) # Build a mask: [8400, N]            
            align_scores *= topk_mask

            # A cell might be in top-k for multiple GTs, Assign each cell to the GT with highest alignment
            # fg_mask: which cells are positive (assigned to any GT)            
            fg_mask = align_scores.amax(dim=1) > 0 #[8400] bool
            assigned_gt_idx = align_scores.argmax(dim=1) # [8400] — index of best GT per cell

            assigned_gt_bboxes = gt_bboxes[assigned_gt_idx] # [8400, 4]
            assigned_gt_cls = gt_labels[assigned_gt_idx].long() # [8400]
            
            align_values = align_scores[torch.arange(num_dets), assigned_gt_idx]
            
            # Normalize per GT: divide by max alignment for that GT
            max_per_gt = align_scores.amax(dim=0)
            norm_align = align_values / (max_per_gt[assigned_gt_idx] + 1e-5)
            
            # Scale by IoU between prediction and assigned GT
            iou_values = iou[torch.arange(num_dets), assigned_gt_idx] #[8400]
            soft_targets = norm_align * iou_values #[8400]
            
            # Build the full target tensor: [8400, C]
            target_scores = torch.zeros(num_dets, num_classes, device=device)
            target_scores[fg_mask, assigned_gt_cls[fg_mask]] = soft_targets[fg_mask]

            lcls += torch.nn.functional.binary_cross_entropy_with_logits(
                pred_scores_logits[b],
                target_scores,
                reduction="mean"
            ) 
            
            if fg_mask.sum() > 0:
                ciou = DetectionHelper.bbox_iou(
                    pbox[fg_mask],
                    assigned_gt_bboxes[fg_mask],
                    x1y1x2y2=True, CIoU=True
                )
                
                weight = target_scores[fg_mask, assigned_gt_cls[fg_mask]]
                lbox += ((1 - ciou) * weight).sum() / max(weight.sum(), 1e-8)            
            else:
                lbox += torch.tensor(0.0, device=device)
                
            if fg_mask.sum() > 0:
                gt_xyxy = assigned_gt_bboxes[fg_mask]
                anchor_fg = anchor_points[fg_mask]
                stride_fg = strides_orig.squeeze(-1)[fg_mask]

                target_l = (anchor_fg[:, 0] - gt_xyxy[:, 0]) / stride_fg
                target_t = (anchor_fg[:, 1] - gt_xyxy[:, 1]) / stride_fg
                target_r = (gt_xyxy[:, 2] - anchor_fg[:, 0]) / stride_fg
                target_b = (gt_xyxy[:, 3] - anchor_fg[:, 1]) / stride_fg

                target_ltrb = torch.stack([target_l, target_t, target_r, target_b], dim=-1)  # [M, 4]
                target_ltrb = target_ltrb.clamp(0, reg_max - 1 - 0.01)  # clamp to [0, 14.99]

                # DFL: cross-entropy targeting two adjacent bins
                raw_logits = pred_distri_logits[b][fg_mask]  # [M, 4, 16]
                raw_logits = raw_logits.view(-1, 16)      # [M*4, 16]
                target_flat = target_ltrb.view(-1)        # [M*4]

                target_left = target_flat.long()          # floor
                target_right = target_left + 1            # ceil
                weight_right = target_flat - target_left.float()
                weight_left = 1.0 - weight_right

                log_probs = torch.nn.functional.log_softmax(raw_logits, dim=-1)  # [M*4, 16]

                loss_dfl = -(
                    weight_left  * log_probs[torch.arange(len(target_flat)), target_left] +
                    weight_right * log_probs[torch.arange(len(target_flat)), target_right]
                )  # [M*4]

                ldfl += loss_dfl.mean()
            else:
                ldfl += torch.tensor(0.0, device=device)

        loss = 0.5 * lcls + 7.5 * lbox + 1.5 * ldfl
        return loss, {"lcls": lcls.item(), "lbox": lbox.item(), "ldfl": ldfl.item()}