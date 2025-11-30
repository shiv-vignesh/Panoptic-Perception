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
                    v = (4/ (torch.pi ** 2)) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                    with torch.no_grad():
                        alpha = v / (1 - iou + v + eps)
                    return iou - (rho2 / c2 + v * alpha)
            elif GIoU:
                c_area = cw * ch + eps
                return iou - (c_area - union_area) / c_area
            
        return iou
    
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
                output[img_idx] = torch.cat(keep_boxes, 0)[:max_detections]

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
        thickness: int = 2,
        font_scale: float = 0.5,
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

    @staticmethod
    def build_targets(outputs:List[torch.Tensor], targets:torch.Tensor, num_anchors:int,
                    anchors_tensor:torch.Tensor, strides:List[int]):

        """
        gain is a length-7 tensor used to scale targets from normalized image coordinates to
            grid coordinates for the current detection layer.
            The seven entries correspond to columns of targets plus the appended anchor id later:
                [img, cls, x, y, w, h, anchor_id]. Initially all ones; 2:6 will be set to grid sizes.

        ai : initial tensor of shape (num_anchors, num_targets) where each row is the anchor index (0 to num_anchors-1)
            ex - [[0,0,0,0,0,0,0,0],
            [1,1,1,1,1,1,1,1],
            [2,2,2,2,2,2,2,2]]

        """

        num_targets = targets.shape[0]
        # Infer num_classes from output shape: (batch, anchors, h, w, 5+num_classes)
        num_classes = outputs[0].shape[-1] - 5

        tcls = []
        tbox = []
        indices = []
        anchors_list = []

        # print(f"\n{'='*60}")
        # print(f"[build_targets] Starting with {num_targets} targets")
        # print(f"  Input targets shape: {targets.shape}")
        # print(f"  Anchors tensor shape: {anchors_tensor.shape}")
        # print(f"  Anchors: {anchors_tensor}")
        # print(f"  Strides: {strides}")
        # print(f"  Num detection layers: {len(outputs)}")
        # for i, output in enumerate(outputs):
        #     print(f"    Layer {i}: output shape = {output.shape}")
        # print(f"{'='*60}")

        anchors_tensor = anchors_tensor.to(outputs[0].device)

        gain = torch.ones(7, device=outputs[0].device)  # normalized to gridspace gain
        
        # Make a tensor that iterates 0-2 for 3 anchors and repeat that as many times as we have target boxes
        # tensor shape: (num_anchors, num_targets)
        ai = torch.arange(num_anchors, device=outputs[0].device).float().view(num_anchors, 1).repeat(1, num_targets)
        
        # Copy target boxes anchor size times and append an anchor index to each copy the anchor index is also expressed by the new first dimension
        # targets original shape (ntargets, 6) -> new shape (num_anchors, num_targets, 7)
        targets = torch.cat((targets.repeat(num_anchors, 1, 1), ai[:, :, None]), 2)

        for i, anchors in enumerate(anchors_tensor):
            #scale anchors to grid size
            anchors = anchors / strides[i]
            
            # [1, 1, grid_w, grid_h, grid_w, grid_h, 1]
            # gain[2:6] = torch.tensor(outputs[i].shape)[[3, 4, 3, 4]] # FIXME, indexing by height and width of feature map
            gain[2:6] = torch.tensor(outputs[i].shape)[[3, 2, 3, 2]] # indexing by height and width of feature map

            # multiplying targets by gain scales normalized x,y,w,h to grid-cell coordinates.
            t = targets * gain # shape (num_anchors, num_targets, 7)

            if num_targets:
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio

                # DEBUG: Print anchor matching info
                # print(f"\n[build_targets] Layer {i}:")
                # print(f"  Anchors (grid-scaled): {anchors}")
                # print(f"  Targets WH (grid-scaled): {t[:, :, 4:6].shape} -> {t[0, :5, 4:6] if t.shape[1] > 0 else 'empty'}")
                # print(f"  WH ratios shape: {r.shape}")
                # print(f"  Max ratio per target: {torch.max(r, 1 / r).max(2)[0][:5]}")

                # Select the ratios that have the highest divergence in any axis and check if the ratio is less than 4
                j = torch.max(r, 1 / r).max(2)[0] < 4.0 # compare to threshold
                # print(f"  Targets passing filter: {j.sum()} / {j.numel()}")
                t = t[j] #filter
                # print(f"  Final targets for this layer: {t.shape[0]}")
            else:
                t = torch.zeros((0, 7), device=targets.device)
            
            b, c = t[:, :2].long().T # batch index, class

            # We isolate the target cell associations.
            # x, y, w, h are allready in the cell coordinate system meaning an x = 1.2 would be 1.2 times cellwidth
            gxy = t[:, 2:4] # grid x,y
            gwh = t[:, 4:6] # grid w,h

            gij = gxy.long() #grid cell indices
            gi, gj = gij.T #grid cell x,y indices

            a = t[:, 6].long() #anchor indices

            # Add target tensors for this yolo layer to the output lists
            # Add to index list and limit index range to prevent out of bounds
            indices.append((b, a, gj.clamp_(0, gain[3].long() - 1), gi.clamp_(0, gain[2].long() - 1)))
            # Add to target box list and convert box coordinates from global grid coordinates to local offsets in the grid cell
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            # Add correct anchor for each target to the list
            anchors_list.append(anchors[a])
            # Add class for each target to the list (clamp to valid range [0, num_classes-1])
            # This prevents IndexError if dataset has invalid class indices
            tcls.append(c.clamp(0, num_classes - 1))
            
        return tcls, tbox, indices, anchors_list

    @staticmethod
    def focal_loss(inputs, targets, alpha=0.25, gamma=2.0, reduction="mean"):
        """
        Compute focal loss for binary classification.
        Inputs:
            inputs: raw logits, tensor of shape (N, *).
            targets: binary targets (0 or 1), same shape as inputs.
            alpha: balancing factor.
            gamma: focusing parameter.
            reduction: 'mean' or 'sum'.
        Returns:
            scalar loss.
        """
        # Compute binary cross-entropy loss with logits (no reduction)
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        # Get the probability of the true class
        pt = torch.exp(-bce_loss)
        loss = alpha * (1 - pt) ** gamma * bce_loss
        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        else:
            return loss   

    @staticmethod
    def compute_detection_loss(outputs:List[torch.Tensor], targets:torch.Tensor, 
                            num_anchors:int, anchors_tensor:torch.Tensor, strides:List[int],
                            cls_loss_type:str='BCE'):
        """
        outputs[i] : Tensor of shape (batch_size, num_anchors, grid_h, grid_w, (5 + num_classes))
        targets : Tensor of shape (num_targets, 6) where each target is (batch_idx, class, x, y, w, h)
        """

        lcls = torch.tensor(0.0, device=outputs[0].device)  # Classification loss
        lbox = torch.tensor(0.0, device=outputs[0].device)  # Bounding box regression loss
        lobj = torch.tensor(0.0, device=outputs[0].device)  # Objectness loss
        
        tcls, tbox, indices, anchors = DetectionLossCalculator.build_targets(outputs, targets, num_anchors, anchors_tensor, strides)
        
        ious = []

        for i, output in enumerate(outputs):
            b, a, gj, gi = indices[i]
            tobj = torch.zeros_like(output[..., 0], device=output.device)
            num_targets = b.shape[0]

            # print(f"\n[compute_detection_loss] Layer {i}: num_targets = {num_targets}")

            if num_targets:
                                
                ps = output[b, a, gj, gi] # shape (num_targets, 5 + num_classes)

                # # Predictions in grid space
                # grid_xy = torch.stack((gi, gj), dim=1).float()
                # pxy = ps[:, 0:2].sigmoid() + grid_xy  # Absolute grid coordinates
                # pwh = torch.exp(ps[:, 2:4]) * anchors[i]
                # pbox = torch.cat((pxy, pwh), 1)

                # # Targets in grid space (convert offset to absolute)
                # tbox_absolute = tbox[i].clone()
                # tbox_absolute[:, :2] += grid_xy  # offset → absolute grid coordinate

                # YOLOP style decoding in grid space
                pxy = ps[:, 0:2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2.) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)

                # Target boxes also need adjustment for YOLOP style
                # tbox xy is (gxy - gij) which is in [0,1), YOLOP pxy is in [-0.5, 1.5]
                tbox_adj = tbox[i].clone()
                # No adjustment needed for tbox since it stores actual offsets

                iou = DetectionHelper.bbox_iou(pbox, tbox_adj, x1y1x2y2=False, CIoU=True)
                ious.append(iou.mean().item())
                lbox += (1.0 - iou).mean()

                # Objectness
                tobj[b, a, gj, gi] = iou.detach().clamp(0).type(tobj.dtype)

                if ps.shape[1] - 5 >= 1:
                    # Classification loss (only if multiple classes)
                    t = torch.zeros_like(ps[:, 5:], device=ps.device)
                    t[range(num_targets), tcls[i]] = 1.0

                    if cls_loss_type == 'BCE':
                        lcls += torch.nn.functional.binary_cross_entropy_with_logits(ps[:, 5:], t, reduction='mean')
                    elif cls_loss_type == 'focal':
                        lcls += DetectionLossCalculator.focal_loss(ps[:, 5:], t, reduction='mean')

            lobj += torch.nn.functional.binary_cross_entropy_with_logits(
                output[..., 4], tobj, reduction='mean'
            )

        lbox *= DetectionLossCalculator.bbox_weight
        lobj *= DetectionLossCalculator.obj_weight
        lcls *= DetectionLossCalculator.cls_weight

        loss = lbox + lobj + lcls

        return loss, {
            "lbox": lbox, 
            "lobj": lobj,
            "lcls": lcls,
            "iou": sum(ious) / len(ious) if len(ious) > 0 else 0.0
        }
        
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

                # Classification loss
                nc = pred.shape[-1] - 5
                if nc > 1:
                    t = torch.zeros_like(ps[:, 5:], device=device)
                    t[range(nt), tcls[i]] = 1.0
                    t = t.detach()

                    if cls_loss_type.lower() == 'focal':
                        lcls += DetectionLossCalculator.focal_loss(ps[:, 5:], t, alpha=0.25, gamma=2.0)
                    else:  # BCE
                        lcls += torch.nn.functional.binary_cross_entropy_with_logits(ps[:, 5:], t, reduction='mean')

            # Objectness loss
            lobj += torch.nn.functional.binary_cross_entropy_with_logits(pred[..., 4], tobj, reduction='mean')

        # YOLOv3 loss weights
        lbox *= 0.05
        lobj *= 1.0
        lcls *= 0.5

        loss = lbox + lobj + lcls
        return loss, {"lbox": lbox, "lobj": lobj, "lcls": lcls, "iou": sum(ious)/len(ious) if ious else 0.0}
