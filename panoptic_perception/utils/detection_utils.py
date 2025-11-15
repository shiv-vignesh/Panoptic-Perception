from typing import List
import torch

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

            # Filter by objectness threshold
            obj_conf = pred[:, 4]
            pred = pred[obj_conf > conf_threshold]

            if pred.shape[0] == 0:
                continue

            # Convert from (x, y, w, h) to (x1, y1, x2, y2)
            boxes = DetectionHelper.xywh2xyxy(pred[:, :4])

            # Get objectness and class predictions from filtered pred
            obj_conf_filtered = pred[:, 4]
            class_conf, class_pred = pred[:, 5:].max(1, keepdim=True)

            # Combine: [x1, y1, x2, y2, obj_conf, class_conf, class_pred]
            detections = torch.cat([boxes, obj_conf_filtered.unsqueeze(1), class_conf, class_pred.float()], 1)

            # Perform NMS per class
            unique_classes = detections[:, -1].unique()
            keep_boxes = []

            for cls in unique_classes:
                cls_mask = detections[:, -1] == cls
                cls_detections = detections[cls_mask]

                # Sort by confidence
                conf_sort_idx = torch.argsort(cls_detections[:, 4] * cls_detections[:, 5], descending=True)
                cls_detections = cls_detections[conf_sort_idx]

                # NMS
                keep_idx = DetectionHelper.nms_boxes(
                    cls_detections[:, :4],
                    cls_detections[:, 4] * cls_detections[:, 5],
                    iou_threshold
                )

                keep_boxes.append(cls_detections[keep_idx])

            if keep_boxes:
                output[img_idx] = torch.cat(keep_boxes, 0)[:max_detections]

        return output

    @staticmethod
    def nms_boxes(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> List[int]:
        """
        Perform NMS on boxes with given scores.

        Args:
            boxes: (N, 4) tensor [x1, y1, x2, y2]
            scores: (N,) tensor of confidence scores
            iou_threshold: IoU threshold

        Returns:
            List of indices to keep
        """
        keep = []
        order = scores.argsort(descending=True)

        while order.numel() > 0:
            if order.numel() == 1:
                keep.append(order.item())
                break

            i = order[0].item()
            keep.append(i)

            # Compute IoU of the kept box with the rest
            ious = DetectionHelper.box_iou(boxes[i:i+1], boxes[order[1:]])

            # Keep boxes with IoU less than threshold
            mask = ious[0] <= iou_threshold
            order = order[1:][mask]

        return keep

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
            gain[2:6] = torch.tensor(outputs[i].shape)[[3, 4, 3, 4]] #indexing by height and width of feature map

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

                # Regression
                pxy = ps[:, 0:2].sigmoid()
                pwh = torch.exp(ps[:, 2:4]) * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)            

                iou = DetectionHelper.bbox_iou(pbox, tbox[i], x1y1x2y2=False, CIoU=True)
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
                    