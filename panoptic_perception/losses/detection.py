from abc import ABC, abstractmethod

import torch
import torchvision
import numpy as np

from scipy.interpolate import interp1d

from typing import Tuple, Dict, List

from panoptic_perception.utils.detection_utils import DetectionHelper, DetectionLossCalculator, ATSS
from panoptic_perception.models.types import LaneDetectionLossItems

from panoptic_perception.losses.loss_factory import LossFactory
from panoptic_perception.models.types import DetectionLossItems

class DetectionLoss(ABC):

    def __init__(self, bbox_weight: float = 0.05,
            obj_weight: float = 1.0,
            cls_weight: float = 0.5,
            balance:list = [4.0, 1.0, 0.4],
            gamma:float = 2.0,
            iou_aware_cls:bool=False,
            label_smoothing:float=0.0, 
            autobalance:bool=False,
            ssi = [1.0, 1.0, 1.0]):
        
        self.bbox_weight = bbox_weight
        self.obj_weight = obj_weight
        self.cls_weight = cls_weight

        self.balance = balance
        self.gamma = gamma

        self.iou_aware_cls = iou_aware_cls
        self.label_smoothing = label_smoothing
        self.autobalance = autobalance
        self.ssi = ssi

    def focal_loss(self, inputs, targets, alpha=0.25, gamma=2.0, reduction="mean"):
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
        loss = alpha * (1. - pt) ** gamma * bce_loss
        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        elif reduction == "none":
            return loss
        else:
            return loss

    @abstractmethod
    def __call__(self, model:DetectionLossItems) -> Tuple[torch.Tensor, Dict]:
        raise NotImplementedError

@LossFactory.register_loss_function("detection-loss-anchor")
class YOLODetectionLoss(DetectionLoss):
    def __init__(self, bbox_weight = 0.05, obj_weight = 1, cls_weight = 0.5, balance = [4, 1, 0.4], gamma = 2, iou_aware_cls = False, label_smoothing = 0, autobalance = False, ssi=[1, 1, 1]):
        super().__init__(bbox_weight, obj_weight, cls_weight, balance, gamma, iou_aware_cls, label_smoothing, autobalance, ssi)

        self.detection_loss_weight = 1.0

    def build_targets_2(self, preds:List[torch.Tensor], targets:torch.Tensor, 
                        num_layers:int, anchors:torch.Tensor, stride:torch.Tensor):
                
        assert anchors.shape[0] == num_layers, \
            f'Expected Number of Anchors to be equal to num layers, got {anchors.shape[0]} and {num_layers}'
        
        if stride is None or stride.numel() == 0:
            raise ValueError(f'Expected non-empty stride tensor, \
                             got {stride.shape if isinstance(stride, torch.Tensor) else stride}')
        
        assert stride.shape[0] == num_layers, \
            f'Expected Number of stride to be equal to num layers, got {stride.shape[0]} and {num_layers}'

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

    def __call__(self, loss_items:DetectionLossItems, cls_loss_type:str='focal'):

        targets = loss_items.targets

        device = targets.device
        lcls = torch.zeros(1, device=device)
        lbox = torch.zeros(1, device=device)
        lobj = torch.zeros(1, device=device)

        if loss_items.detection_logits is None:
            raise ValueError(f'Expected detection_logits, got {loss_items.detection_logits}')

        if loss_items.targets is None or loss_items.targets.numel() == 0:
            raise ValueError(f'Expected targets to be non-empty, got {loss_items.targets}')

        # Build targets
        tcls, tbox, indices, anch_per_target = self.build_targets_2(loss_items.detection_logits, 
                                                                                    targets, 
                                                                                    loss_items.num_layers, 
                                                                                    loss_items.anchors, 
                                                                                    loss_items.stride)

        ious = []
        for i, pred in enumerate(loss_items.detection_logits):
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
        loss = self.detection_loss_weight * loss

        return loss, {"lbox": lbox, "lobj": lobj, "lcls": lcls, "iou": sum(ious)/len(ious) if ious else 0.0}

@LossFactory.register_loss_function("detection-loss-anchor-free")
class YOLOv8DetectionLoss(DetectionLoss):

    def __init__(self, 
                bbox_weight = 7.5, obj_weight = 1, 
                dfl_weight = 1.5, cls_weight = 0.5, balance = [4, 1, 0.4], 
                gamma = 2, iou_aware_cls = False, 
                label_smoothing = 0, autobalance = False, 
                ssi=[1, 1, 1],
                topk:int=10, xywh2xyxy:bool=True):
        
        super().__init__(bbox_weight, obj_weight, cls_weight, balance, 
                        gamma, iou_aware_cls, label_smoothing, autobalance, ssi)

        self.topk = topk
        self.xywh2xyxy = xywh2xyxy
        self.dfl_weight = dfl_weight

        self.detection_loss_weight = 1.0

    def __call__(self, loss_items:DetectionLossItems, cls_loss_type:str='focal'):

        """
        pred_scores - (bs, 8400, C)
        pred_distri - (bs, 8400, 64)
        anchor_points - (8400, 2)
        strides - (8400, 1)
        """

        img_h, img_w = loss_items.image_size
        device = loss_items.pred_distri_logits.device
        targets:torch.Tensor = loss_items.targets

        bs, num_dets, num_classes = loss_items.pred_scores_logits.shape
        reg_max = loss_items.pred_distri_logits.shape[-1] // 4 # 64 // 4 = 16
        loss_items.pred_distri_logits = loss_items.pred_distri_logits.view(bs, num_dets, 4, reg_max)
        pred_distri = torch.softmax(loss_items.pred_distri_logits, dim=-1)

        project = torch.arange(pred_distri.shape[-1], dtype=pred_distri.dtype, device=device)
        pred_ltrb = (pred_distri * project).sum(dim=-1)

        anchor_x = loss_items.anchor_points[:, 0].unsqueeze(0)
        anchor_y = loss_items.anchor_points[:, 1].unsqueeze(0)
        
        strides_orig = loss_items.strides_v8
        loss_items.strides_v8 = loss_items.strides_v8.unsqueeze(0)

        # Convert to xyxy in pixel coords:
        x1 = anchor_x - pred_ltrb[:, :, 0] * loss_items.strides_v8.squeeze(-1)
        y1 = anchor_y - pred_ltrb[:, :, 1] * loss_items.strides_v8.squeeze(-1)
        x2 = anchor_x + pred_ltrb[:, :, 2] * loss_items.strides_v8.squeeze(-1)
        y2 = anchor_y + pred_ltrb[:, :, 3] * loss_items.strides_v8.squeeze(-1)

        pred_bboxes = torch.stack([x1, y1, x2, y2], dim=-1)

        lcls = torch.zeros(1, device=device)
        lbox = torch.zeros(1, device=device)
        ldfl = torch.zeros(1, device=device)

        for b in range(bs):
            gt_bboxes = targets[targets[:, 0] == b][:, 2:]
            gt_labels = targets[targets[:, 0] == b][:, 1]
            pbox = pred_bboxes[b]
            plabels = loss_items.pred_scores_logits[b]

            if not gt_bboxes.shape[0]:
                continue

            if self.xywh2xyxy:
                gt_x1 = (gt_bboxes[:, 0] - gt_bboxes[:, 2] / 2) * img_w
                gt_y1 = (gt_bboxes[:, 1] - gt_bboxes[:, 3] / 2) * img_h
                gt_x2 = (gt_bboxes[:, 0] + gt_bboxes[:, 2] / 2) * img_w
                gt_y2 = (gt_bboxes[:, 1] + gt_bboxes[:, 3] / 2) * img_h
                gt_bboxes = torch.stack([gt_x1, gt_y1, gt_x2, gt_y2], dim=-1)

            is_inside = (
                (loss_items.anchor_points[:, 0:1] > gt_bboxes[None, :, 0]) &
                (loss_items.anchor_points[:, 1:2] > gt_bboxes[None, :, 1]) &
                (loss_items.anchor_points[:, 0:1] < gt_bboxes[None, :, 2]) &
                (loss_items.anchor_points[:, 1:2] < gt_bboxes[None, :, 3]) 
            )

            iou = DetectionHelper.bbox_iou_pairwise(pbox, gt_bboxes)
            align_scores = DetectionHelper.align_scores_pairwise(
                plabels, gt_labels, iou, is_inside
            )

            # For each GT (column), get top-k cells by alignment metric 
            topk_metrics, topk_indices = torch.topk(align_scores, dim=0, k=self.topk)

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
                loss_items.pred_scores_logits[b],
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
                anchor_fg = loss_items.anchor_points[fg_mask]
                stride_fg = strides_orig.squeeze(-1)[fg_mask]

                target_l = (anchor_fg[:, 0] - gt_xyxy[:, 0]) / stride_fg
                target_t = (anchor_fg[:, 1] - gt_xyxy[:, 1]) / stride_fg
                target_r = (gt_xyxy[:, 2] - anchor_fg[:, 0]) / stride_fg
                target_b = (gt_xyxy[:, 3] - anchor_fg[:, 1]) / stride_fg

                target_ltrb = torch.stack([target_l, target_t, target_r, target_b], dim=-1)  # [M, 4]
                target_ltrb = target_ltrb.clamp(0, reg_max - 1 - 0.01)  # clamp to [0, 14.99]

                # DFL: cross-entropy targeting two adjacent bins
                raw_logits = loss_items.pred_distri_logits[b][fg_mask]  # [M, 4, 16]
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

        loss = self.cls_weight * lcls + self.bbox_weight * lbox + self.dfl_weight * ldfl
        loss = self.detection_loss_weight * loss

        return loss, {"lcls": lcls.item(), "lbox": lbox.item(), "ldfl": ldfl.item()}

@LossFactory.register_loss_function("detection-loss-ATSS")
class ATSSDetectionLoss(DetectionLoss):

    def __init__(self, bbox_weight = 0.05, obj_weight = 1, 
                cls_weight = 0.5, balance = [4, 1, 0.4], 
                gamma = 2, iou_aware_cls = False, 
                label_smoothing = 0, autobalance = False, ssi=[1, 1, 1]):
        
        super().__init__(bbox_weight, obj_weight, cls_weight, balance, gamma, iou_aware_cls, label_smoothing, autobalance, ssi)

        self.detection_loss_weight = 1.0

    def _check_center_constraint(self, anchor_centers:torch.Tensor,
                                gt_bboxes:torch.Tensor,
                                candidate_indices:torch.Tensor):
        
        """
            anchor_centers - (total_anchors, 2)
            gt_bboxes - (N_gt, 4)
            candidate_indices - (N_gt, L * K) , L = len(level_sizes)
        """

        candidate_centers = anchor_centers[candidate_indices] #(N_gt, L * K, 2)

        x1 = gt_bboxes[:, 0:1]
        y1 = gt_bboxes[:, 1:2]
        x2 = gt_bboxes[:, 2:3]
        y2 = gt_bboxes[:, 3:4]

        in_box = (
            (candidate_centers[..., 0] >= x1) & (candidate_centers[..., 0] <= x2) &
            (candidate_centers[..., 1] >= y1) & (candidate_centers[..., 1] <= y2) 
        )

        return in_box
    
    def _match_targets_to_proposals(self, anchor_proposals:torch.Tensor, anchor_centers:torch.Tensor,
                                    gt_centers:torch.Tensor, gt_bboxes:torch.Tensor, 
                                    gt_labels:torch.Tensor,
                                    level_sizes:List[int],
                                    k:int=9):

        # iou_matrix shape: (N_gt, A_anchors)
        iou_pairwise = DetectionHelper.bbox_iou_pairwise(gt_bboxes, anchor_proposals)
        if iou_pairwise.numel() == 0 or iou_pairwise.shape[0] == 0:
            raise ValueError(
                f"IOU Pairwise contains no elements or has no ground truth"
            )

        # dist_matrix shape: (N_gt, A_anchors)
        dist_matrix = torch.cdist(gt_centers, anchor_centers, p=2)

        candidate_indices = []
        start_idx = 0
        
        for level_sz in level_sizes:
            end_idx = start_idx + level_sz

            level_dist = dist_matrix[:, start_idx:end_idx]
            _, topk_idxs = torch.topk(level_dist, 
                                    k=k, dim=1, largest=False)

            # Convert local level index back to global anchor proposal index
            candidate_indices.append(start_idx + topk_idxs)
            start_idx = end_idx

        # (N_gt, L * K) , L = len(level_sizes)
        candidate_indices = torch.concat(candidate_indices, dim=1)
        # (N_gt, L * K) 

        assert iou_pairwise.shape[0] == candidate_indices.shape[0], \
            f"Expected IOU Pairwise Matrix and indices to be same at dim=0, got: {iou_pairwise.shape[0]} and {candidate_indices.shape[0]}"
        
        candidate_ious = torch.gather(iou_pairwise, dim=1, index=candidate_indices)

        # (N_gt, 1)
        iou_mean = candidate_ious.mean(dim=1, keepdim=True)
        iou_std = candidate_ious.std(dim=1, keepdim=True)
        #unique dynamic threshold formula per individual object
        adaptive_thresholds = iou_mean + iou_std 

        # (N_gt, L * K) , L = len(level_sizes)
        iou_filter = candidate_ious >= adaptive_thresholds 
        in_box = self._check_center_constraint(anchor_centers, gt_bboxes, candidate_indices)

        accept = iou_filter & in_box #(N_gt, L * K)

        matches = torch.full((anchor_proposals.shape[0],), -1, dtype=torch.long, device=gt_bboxes.device)
        best_iou = torch.full((anchor_proposals.shape[0],), -1.0, device=gt_bboxes.device)
        for gt_idx in range(gt_bboxes.shape[0]):
            valid_candidate_mask = accept[gt_idx]
            global_anchors_for_gt = candidate_indices[gt_idx][valid_candidate_mask]
            global_anchor_ious_for_gt = iou_pairwise[gt_idx][global_anchors_for_gt]

            is_better_match = global_anchor_ious_for_gt > best_iou[global_anchors_for_gt]
            selected_anchor_ids = global_anchors_for_gt[is_better_match]

            matches[selected_anchor_ids] = gt_idx
            best_iou[selected_anchor_ids] = global_anchor_ious_for_gt[is_better_match]

        clamped = matches.clamp(min=0)
        matched_gt_bboxes = gt_bboxes[clamped]
        matched_gt_labels = gt_labels[clamped]
        pos_mask = matches >= 0

        return matches, matched_gt_bboxes, matched_gt_labels, pos_mask

    def _prepare_targets(self, 
                        anchor_proposals:torch.Tensor, 
                        targets:torch.Tensor, 
                        img_w, img_h, 
                        batch_size:int, 
                        level_sizes:List[int],
                        device:torch.device):

        """
            anchor_proposals - [A, 4]
                A - sum(anchors_per_level)
                4 - xyxy
            targets - [N, 6]
                [batch_idx, class_id, x_center, y_center, width, height]
        """

        anchor_centers = (anchor_proposals[:, :2] + anchor_proposals[:, 2:]) / 2
        num_anchors = anchor_centers.shape[0]
        num_coords = anchor_proposals.shape[1]

        # per batch_idx -> matched_gt_labels: (A,) int
        #   gathered gt class id at each anchor (garbage where pos_mask is False)

        batch_labels = torch.zeros(
            (batch_size, num_anchors),
            dtype=torch.long,
            device=device
        )  

        # per batch_idx -> matched_gt_bboxes: (A, 4) float
        #   gathered gt xyxy at each anchor (garbage where pos_mask is False)
        batch_bboxes = torch.zeros(
            (batch_size, num_anchors, num_coords),
            dtype=torch.float32,
            device=device
        )  

        # per batch_idx -> pos_mask: (A,) bool
        # True if anchor was assigned a positive GT, False otherwise
        # NO positives → loss skips this image's reg+cls
        batch_pos_masks = torch.full(
            (batch_size, num_anchors),
            fill_value=False,
            dtype=torch.bool,
            device=device
        )

        # per batch_idx -> matches: (A,) int
        # gt_idx in [0, M-1] for matched anchors, -1 for unmatched
        batch_matches = torch.full(
            (batch_size, num_anchors),
            fill_value=-1,
            dtype=torch.long,
            device=device,
        )

        if targets.ndim != 2 or targets.shape[0] == 0:
            return batch_labels, batch_bboxes, batch_pos_masks, batch_matches

        for batch_idx in range(batch_size):
            gt_bboxes = targets[targets[:, 0] == batch_idx][:, 2:] #(M, 4)
            gt_labels = targets[targets[:, 0] == batch_idx][:, 1] # (M)

            if gt_bboxes.numel() == 0:
                continue

            gt_x1 = (gt_bboxes[:, 0] - gt_bboxes[:, 2] / 2) * img_w
            gt_y1 = (gt_bboxes[:, 1] - gt_bboxes[:, 3] / 2) * img_h
            gt_x2 = (gt_bboxes[:, 0] + gt_bboxes[:, 2] / 2) * img_w
            gt_y2 = (gt_bboxes[:, 1] + gt_bboxes[:, 3] / 2) * img_h

            gt_bboxes = torch.stack([gt_x1, gt_y1, gt_x2, gt_y2], dim=-1) # (M, 4)
            gt_centers = (gt_bboxes[:, :2] + gt_bboxes[:, 2:]) / 2

            # Shapes same order as return 
            # [A], [A, 4], [A], [A]
            matches, matched_gt_bboxes, matched_gt_labels, pos_mask = \
                self._match_targets_to_proposals(
                anchor_proposals, anchor_centers,
                gt_centers, gt_bboxes,
                gt_labels, level_sizes
            )
            
            batch_labels[batch_idx] = matched_gt_labels
            batch_bboxes[batch_idx] = matched_gt_bboxes
            batch_pos_masks[batch_idx] = pos_mask
            batch_matches[batch_idx] = matches
        
        return batch_labels, batch_bboxes, batch_pos_masks, batch_matches
    
    def subsample(self, 
                anchor_proposals:List[torch.Tensor], 
                targets:torch.Tensor, 
                img_w, img_h, 
                batch_size:int,
                device:torch.device):
        """
        Args:
            anchor_proposals List[torch.Tensor]: pre-computed (x, y) center locations per grid 
                (x_grid + 0.5, y_grid + 0.5) * stride for every level of FPN.

                Ensure x1y1x2y2 in pixel coordinates
                e.g. [(20, 20, num_anch), 
                      (40, 40, num_anch), 
                      (80, 80, num_anch)] -> [(1200, 4), (4800, 4), (19200, 4)]

            targets: ground truth from dataset 
                (batch_idx, class_id, x_center, y_center, width, height)

        Returns:
                batch_labels - per batch_idx -> matched_gt_labels: (A,) int
                            gathered gt class id at each anchor (garbage where pos_mask is False)      
                batch_bboxes - per batch_idx -> matched_gt_bboxes: (A, 4) float
                            gathered gt xyxy at each anchor (garbage where pos_mask is False)
                batch_pos_mask - per batch_idx -> pos_mask: (A,) bool
                            True if anchor was assigned a positive GT, False otherwise        

                batch_matches - per batch_idx -> matches: (A,) int
                            gt_idx in [0, M-1] for matched anchors, -1 for unmatched
        """

        # assert len(targets.shape) == 2 and targets.shape[-1] == 6, \
        #     f'Supported targets shape (batch_size, 6), [(batch_idx, class_id, x_center, y_center, width, height)]'

        level_sizes = []        
        for anchor_proposal in anchor_proposals:
            assert len(anchor_proposal.shape) == 2 and anchor_proposal.shape[-1] == 4, \
                f'Expected Anchor Proposals must be size (num_anchors, 4), got: {anchor_proposal.shape}'
            
            num_anchors = anchor_proposal.shape[0]
            level_sizes.append(num_anchors)

        anchor_proposals = torch.cat(anchor_proposals, dim=0).to(device)
        return self._prepare_targets(
            anchor_proposals, 
            targets, 
            img_w, img_h, 
            batch_size, level_sizes, device
        )

    def _validate_anchor_metadata(
        self,
        anchor_cxcy: List[torch.Tensor],
        anchor_wh: List[torch.Tensor],
        anchor_strides: List[torch.Tensor],
        expected_levels: int,
        pred_anchor_counts: List[int],
        device: torch.device | None = None
    ):
        """
        Validates anchor metadata consistency across feature pyramid levels.
        """

        # 1. Level count check
        if len(anchor_cxcy) != expected_levels:
            raise ValueError(
                f"anchor_cxcy has {len(anchor_cxcy)} levels, "
                f"expected {expected_levels}"
            )

        if len(anchor_wh) != expected_levels:
            raise ValueError(
                f"anchor_wh has {len(anchor_wh)} levels, "
                f"expected {expected_levels}"
            )

        if len(anchor_strides) != expected_levels:
            raise ValueError(
                f"anchor_strides has {len(anchor_strides)} levels, "
                f"expected {expected_levels}"
            )


        for i, (cxcy, wh, stride) in enumerate(
            zip(anchor_cxcy, anchor_wh, anchor_strides)
        ):

            # ---- shape checks ----
            if cxcy.ndim != 2 or cxcy.shape[-1] != 2:
                raise ValueError(
                    f"Level {i}: anchor_cxcy expected (A, 2), got {tuple(cxcy.shape)}"
                )

            if wh.ndim != 2 or wh.shape[-1] != 2:
                raise ValueError(
                    f"Level {i}: anchor_wh expected (A, 2), got {tuple(wh.shape)}"
                )

            if stride.ndim != 1:
                raise ValueError(
                    f"Level {i}: anchor_strides expected (A,), got {tuple(stride.shape)}"
                )

            # ---- anchor count consistency ----
            
            A = pred_anchor_counts[i]

            if cxcy.shape[0] != A:
                raise ValueError(
                    f"Level {i}: mismatch in anchor count between prediction anchors and cxcy"
                    f"({A} vs {cxcy.shape[0]})"
                )                

            if wh.shape[0] != A:
                raise ValueError(
                    f"Level {i}: mismatch in anchor count between prediction anchors and  wh "
                    f"({A} vs {wh.shape[0]})"
                )

            if stride.shape[0] != A:
                raise ValueError(
                    f"Level {i}: mismatch in anchor count between prediction anchors and stride "
                    f"({A} vs {stride.shape[0]})"
                )

            # ---- device consistency ----
            if device is not None:
                if cxcy.device != device:
                    raise ValueError(
                        f"Level {i}: cxcy on {cxcy.device}, expected {device}"
                    )
                if wh.device != device:
                    raise ValueError(
                        f"Level {i}: wh on {wh.device}, expected {device}"
                    )
                if stride.device != device:
                    raise ValueError(
                        f"Level {i}: stride on {stride.device}, expected {device}"
                    )

        return True

    def __call__(self, loss_items: DetectionLossItems, cls_loss_type: str = 'focal'):

        # outputs = loss_items.detection_logits
        targets = loss_items.targets

        if loss_items.detection_logits is None:
            raise ValueError(f'Expected detection_logits, got {loss_items.detection_logits}')

        if len(loss_items.detection_logits) == 0 or len(loss_items.anchor_proposals) == 0:
            raise ValueError(
                f"Expected non-empty detection_logits and anchor proposals, "
                f"got {len(loss_items.detection_logits)} detection_logits and "
                f"{len(loss_items.anchor_proposals)} anchor sets."
            )

        if len(loss_items.detection_logits) != len(loss_items.anchor_proposals):
            raise ValueError(
                f"Expected equal number of outpdetection_logitsuts and anchor proposal tensors, "
                f"got {len(loss_items.detection_logits)} detection_logits and "
                f"{len(loss_items.anchor_proposals)} anchor sets."
            )

        device = targets.device
        img_h, img_w = loss_items.image_size
        batch_size = loss_items.batch_size

        batch_labels, batch_bboxes, batch_pos_masks, batch_matches = \
            self.subsample(
                anchor_proposals=loss_items.anchor_proposals,
                targets=targets,
                img_h=img_h,
                img_w=img_w,
                batch_size=batch_size,
                device=device,
            )

        feature_dim = loss_items.detection_logits[0].shape[-1]

        def reshape_output(i: int, output: torch.Tensor):
            if output.shape[0] != batch_size:
                raise ValueError(
                    f"Output tensor {i} has batch size "
                    f"{output.shape[0]} != {batch_size}"
                )
            if output.shape[-1] != feature_dim:
                raise ValueError(
                    f"Output tensor {i} has feature dim "
                    f"{output.shape[-1]} != {feature_dim}"
                )
            return output.permute(0, 2, 3, 1, 4) \
                         .contiguous() \
                         .view(batch_size, -1, output.shape[-1])

        reshaped_outputs = []
        pred_anchor_counts = []
        for i, output in enumerate(loss_items.detection_logits):
            output = reshape_output(i, output)
            reshaped_outputs.append(output)
            pred_anchor_counts.append(output.shape[1])

        outputs_flat = torch.cat(reshaped_outputs, dim=1)   # renamed to avoid shadowing

        self._validate_anchor_metadata(
            loss_items.anchor_cxcy,
            loss_items.anchor_wh,
            loss_items.anchor_strides,
            len(pred_anchor_counts),
            pred_anchor_counts,
            device=device,
        )

        anchor_cxcy = torch.cat(loss_items.anchor_cxcy, dim=0)     # (A, 2)
        anchor_wh = torch.cat(loss_items.anchor_wh, dim=0)       # (A, 2)
        anchor_strides = torch.cat(loss_items.anchor_strides, dim=0)  # (A,)

        lcls = torch.zeros(1, device=device)
        lbox = torch.zeros(1, device=device)
        lobj = torch.zeros(1, device=device)

        raw_xy = outputs_flat[..., :2]
        raw_wh = outputs_flat[..., 2:4]
        obj_logits = outputs_flat[..., 4]
        cls_logits = outputs_flat[..., 5:]

        # offset in (-1, 1) * scale to ±stride pixels + anchor cell center
        pxy = (raw_xy.sigmoid() * 2 - 1) * anchor_strides[None, :, None] + anchor_cxcy[None, :, :]
        # pwh: predicted size in pixels. (sigmoid*2)^2 ∈ [0, 4]
        pwh = (raw_wh.sigmoid() * 2) ** 2 * anchor_wh[None, :, :]
        pred_xyxy = torch.cat([pxy - pwh / 2, pxy + pwh / 2], dim=-1)   # (B, A, 4)

        iou = DetectionHelper.bbox_iou(
            pred_xyxy[batch_pos_masks],
            batch_bboxes[batch_pos_masks],
            x1y1x2y2=True, CIoU=True,
        )
        plain_iou = DetectionHelper.bbox_iou(
            pred_xyxy[batch_pos_masks],
            batch_bboxes[batch_pos_masks],
            x1y1x2y2=True,
        )

        if (plain_iou < 0).any():
            raise ValueError(
                f"IOU Tensor contains negative values: {plain_iou[plain_iou < 0]}"
            )

        num_pos = batch_pos_masks.sum().item()
        if num_pos > 0:
            lbox += (1.0 - iou).mean()

            pos_cls_logits = cls_logits[batch_pos_masks]              # (n_pos, K)
            pos_labels = batch_labels[batch_pos_masks].long()     # (n_pos,)

            num_classes = cls_logits.shape[-1]
            if num_classes > 1:
                cp = 1.0 - 0.5 * self.label_smoothing
                cn = 0.5 * self.label_smoothing

                cls_target = torch.full_like(pos_cls_logits, fill_value=cn)
                cls_target[torch.arange(pos_cls_logits.shape[0]), pos_labels] = cp
                cls_target = cls_target.detach()

                iou_weight = iou.detach().clamp(0.1, 1.0)
                if cls_loss_type == "focal":
                    per_sample = self.focal_loss(
                        pos_cls_logits, cls_target,
                        alpha=0.25, gamma=self.gamma,
                        reduction="none",
                    )
                else:
                    per_sample = torch.nn.functional.binary_cross_entropy_with_logits(
                        pos_cls_logits, cls_target, reduction="none",
                    )

                if self.iou_aware_cls:
                    lcls += (per_sample.mean(dim=1) * iou_weight).mean()
                else:
                    lcls += per_sample.mean()

        obj_target = torch.zeros_like(obj_logits, device=device)
        obj_target[batch_pos_masks] = plain_iou.detach().clamp(0.1, 1.0)

        obj_per_anchor = torch.nn.functional.binary_cross_entropy_with_logits(
            obj_logits, obj_target, reduction="none",
        )  # (B, A)

        start = 0
        for i, count in enumerate(pred_anchor_counts):
            end = start + count
            obji = obj_per_anchor[:, start:end].mean()

            if self.autobalance:
                obji_val = obji.detach().item()
                self.ssi[i] = self.ssi[i] * 0.9999 + obji_val * 0.0001
                ssi_max = max(self.ssi)
                balance_i = self.ssi[i] / (ssi_max + 1e-9)
                lobj += obji * balance_i
            else:
                lobj += obji * self.balance[i]

            start = end

        lbox *= self.bbox_weight
        lobj *= self.obj_weight
        lcls *= self.cls_weight

        loss = (lbox + lobj + lcls) * self.detection_loss_weight
        return loss, {"lbox": lbox, "lobj": lobj, "lcls": lcls}
    
# ----- Lane Detection -----
# Adapted from: https://github.com/Turoad/CLRNet

@LossFactory.register_loss_function("lane-detection-loss")
class LaneDetectionLossCalculator:

    cls_loss_weight = 2.0
    xyt_loss_weight = 0.5
    iou_loss_weight = 2.0

    NUM_LANE_POINTS = 72
    MAX_LANES = 8    

    loss_weights = {
        "lane_detection":1.0,
        "lane_segmentation": 1.0
    }

    
    def focal_loss(self, inputs, targets, alpha=0.25, gamma=2.0, reduction='none'):
        """
        Softmax-based focal loss for multi-class (2-class bg/fg) classification.
        Note: DetectionLossCalculator.focal_loss is BCE-based (binary), not reusable here.
        """
        # inputs: (N, C) raw logits, targets: (N,) long labels
        p = torch.softmax(inputs, dim=1) + 1e-8        # (N, C)
        ce = torch.nn.functional.cross_entropy(inputs, targets, reduction='none')   # (N,)
        p_t = p[torch.arange(len(targets), device=targets.device), targets]         # (N,)
        # Constant alpha for all classes (matches official CLRNet focal_loss.py)
        loss = alpha * (1 - p_t) ** gamma * ce                                     # (N,)
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        return loss

    
    def line_iou(self, pred, target, img_w, length=15, aligned=True):
        """
        1D lane IoU with rectangular buffer strips.
        Only target validity is checked (predictions can be out-of-bounds and
        naturally receive 0 overlap via geometry).
        """
        # pred: (N, 72) pixels, target: (N, 72) or (M, 72) pixels
        px1, px2 = pred - length, pred + length         # (N, 72)
        tx1, tx2 = target - length, target + length

        if aligned:
            invalid = (target < 0) | (target >= img_w)  # (N, 72)
            ovr = torch.min(px2, tx2) - torch.max(px1, tx1)                         # (N, 72)
            union = torch.max(px2, tx2) - torch.min(px1, tx1)                       # (N, 72)
        else:
            num_pred = pred.shape[0]
            tgt_expanded = target.unsqueeze(0).expand(num_pred, -1, -1)             # (N, M, 72)
            invalid = (tgt_expanded < 0) | (tgt_expanded >= img_w)                  # (N, M, 72)
            ovr = (torch.min(px2[:, None, :], tx2[None, ...]) -
                   torch.max(px1[:, None, :], tx1[None, ...]))                      # (N, M, 72)
            union = (torch.max(px2[:, None, :], tx2[None, ...]) -
                     torch.min(px1[:, None, :], tx1[None, ...]))                    # (N, M, 72)

        # Match official CLRNet: do NOT clamp ovr to >=0.
        # Negative overlap at valid positions penalizes misaligned lanes,
        # giving stronger gradients for geometry refinement.
        ovr[invalid] = 0.
        union[invalid] = 0.
        return ovr.sum(-1) / (union.sum(-1) + 1e-9)    # (N,) or (N, M)

    
    def _distance_cost(self, predictions, targets, img_w):
        """Mean absolute x-distance between pred and GT at valid y-levels (pixel space)."""
        # predictions[:, 6:] and targets[:, 6:] already in pixel space (scaled by assign)
        pred_xs = predictions[:, 6:].unsqueeze(1)       # (P, 1, 72)
        tgt_xs = targets[:, 6:].unsqueeze(0)            # (1, T, 72)

        valid = (tgt_xs >= 0) & (tgt_xs < img_w)       # (P, T, 72)
        dist = torch.abs(pred_xs - tgt_xs) * valid.float()                          # (P, T, 72)
        return dist.sum(-1) / valid.sum(-1).float().clamp(min=1)                    # (P, T)

    
    def _focal_cost(self, cls_pred, gt_labels, alpha=0.25, gamma=2.0, eps=1e-12):
        """Classification cost for assignment (takes softmaxed probabilities)."""
        # cls_pred: (P, 2) softmaxed, gt_labels: (T,) all 1s for valid lanes
        neg_cost = -(1 - cls_pred + eps).log() * (1 - alpha) * cls_pred.pow(gamma)  # (P, 2)
        pos_cost = -(cls_pred + eps).log() * alpha * (1 - cls_pred).pow(gamma)      # (P, 2)
        return pos_cost[:, gt_labels] - neg_cost[:, gt_labels]                      # (P, T)

    
    def _dynamic_k_assign(self, cost, pair_wise_ious):
        """Dynamic k matching: top-4 IoU per GT determines number of assigned priors."""
        # cost: (P, T), pair_wise_ious: (P, T)
        matching = torch.zeros_like(cost)               # (P, T)
        n_candidate_k = min(4, cost.shape[0])

        topk_ious, _ = torch.topk(pair_wise_ious, n_candidate_k, dim=0)            # (4, T)
        dynamic_ks = topk_ious.sum(0).int().clamp(min=1)# (T,)

        for gt_idx in range(cost.shape[1]):
            _, pos_idx = torch.topk(
                cost[:, gt_idx], k=dynamic_ks[gt_idx].item(), largest=False)
            matching[pos_idx, gt_idx] = 1.0

        # Resolve conflicts: prior matched to multiple GTs -> keep lowest cost
        conflict_rows = torch.where(matching.sum(1) > 1)[0]                         # (C,)
        if len(conflict_rows) > 0:
            _, best_gt = cost[conflict_rows].min(dim=1) # (C,)
            matching[conflict_rows] = 0
            matching[conflict_rows, best_gt] = 1.0

        matched_row = matching.sum(1).nonzero().flatten()                           # (K,)
        matched_col = matching[matched_row].argmax(-1).flatten()                    # (K,)
        return matched_row, matched_col

    
    def assign(self, predictions, targets, img_w, img_h,
               distance_cost_weight=3.0, cls_cost_weight=1.0):
        """
        Full lane-to-prior assignment with combined spatial similarity + classification cost.
        All spatial metrics are normalized to similarity scores in (0, 1] before combining.
        """
        # predictions: (P, 78) normalized, targets: (T, 78) normalized (valid GT only)
        with torch.no_grad():
            predictions = predictions.detach().clone()
            targets = targets.detach().clone()
            num_priors = predictions.shape[0]
            num_targets = targets.shape[0]

            # Scale to pixel space for spatial costs
            predictions[:, 3] *= (img_w - 1)           # start_x -> px
            predictions[:, 6:] *= (img_w - 1)          # x-coords -> px
            targets[:, 3] *= (img_w - 1)               # start_x -> px
            targets[:, 6:] *= (img_w - 1)              # x-coords -> px

            # 1. Distance similarity
            distances = self._distance_cost(
                predictions, targets, img_w)            # (P, T)
            distances = 1 - (distances / distances.max().clamp(min=1e-4)) + 1e-2   # -> similarity

            # 2. Classification cost
            cls_pred = torch.softmax(predictions[:, :2], dim=1)                     # (P, 2)
            gt_labels = torch.ones(num_targets, dtype=torch.long,
       device=targets.device)                           # (T,)
            cls_cost = self._focal_cost(
                cls_pred, gt_labels)                    # (P, T)

            # 3. Start point similarity (scale start_y to pixels for comparable L2)
            pred_start = predictions[:, 2:4].clone()    # (P, 2)
            pred_start[:, 0] *= (img_h - 1)            # start_y -> px
            tgt_start = targets[:, 2:4].clone()         # (T, 2)
            tgt_start[:, 0] *= (img_h - 1)             # start_y -> px
            start_xys = torch.cdist(pred_start, tgt_start, p=2).reshape(
                num_priors, num_targets)                # (P, T)
            start_xys = (1 - start_xys / start_xys.max().clamp(min=1e-4)) + 1e-2  # -> similarity

            # 4. Theta similarity (scale to degrees for magnitude)
            thetas = torch.cdist(
                predictions[:, 4:5], targets[:, 4:5], p=1).reshape(
                num_priors, num_targets) * 180          # (P, T)
            thetas = (1 - thetas / thetas.max().clamp(min=1e-4)) + 1e-2            # -> similarity

            # Combined cost: high similarity product -> large negative -> selected by topk(largest=False)
            cost = -(distances * start_xys * thetas) ** 2 * distance_cost_weight \
                   + cls_cost * cls_cost_weight         # (P, T)

            # Line IoU for dynamic-k
            pair_wise_ious = self.line_iou(
                predictions[:, 6:], targets[:, 6:],
                img_w, aligned=False)                   # (P, T)

        return self._dynamic_k_assign(cost, pair_wise_ious)

    
    def compute_lane_det_loss(self, predictions_lists, targets, 
                            img_w, img_h, n_strips=71):
        """
        predictions_lists: List[(bs, 192, 78)] per refinement stage, or single (bs, 192, 78)
        targets: (bs, max_lanes, 78)
            [0]=valid, [1]=category, [2]=start_y, [3]=start_x, [4]=theta, [5]=length, [6:78]=x_coords
            x_coords normalized [0,1], invalid positions = -1e5
        """
        if isinstance(predictions_lists, torch.Tensor):
            predictions_lists = [predictions_lists]

        device = predictions_lists[0].device
        bs = predictions_lists[0].shape[0]
        num_stages = len(predictions_lists)

        cls_loss = torch.tensor(0.0, device=device)
        reg_xytl_loss = torch.tensor(0.0, device=device)
        iou_loss = torch.tensor(0.0, device=device)

        for stage in range(num_stages):
            preds_stage = predictions_lists[stage]      # (bs, 192, 78)

            for b in range(bs):
                pred = preds_stage[b]                   # (192, 78)
                target = targets[b]                     # (max_lanes, 78)
                target = target[target[:, 0] == 1] #TODO, target[:, 1] == 1         # (T, 78) valid

                if len(target) == 0:
                    cls_target = pred.new_zeros(pred.shape[0]).long()                # (192,)
                    cls_loss = cls_loss + self.focal_loss(
                        pred[:, :2], cls_target).sum()
                    continue

                with torch.no_grad():
                    matched_row, matched_col = self.assign(
                        pred, target, img_w, img_h)

                # --- Classification ---
                cls_target = pred.new_zeros(pred.shape[0]).long()                   # (192,)
                cls_target[matched_row] = 1
                cls_loss = cls_loss + self.focal_loss(
                    pred[:, :2], cls_target             # (192, 2) vs (192,)
                ).sum() / target.shape[0]

                # --- Regression: start_y, start_x, theta, length ---
                reg_yxtl = pred[matched_row, 2:6].clone()                           # (K, 4)
                reg_yxtl[:, 0] *= n_strips              # start_y -> strips
                reg_yxtl[:, 1] *= (img_w - 1)          # start_x -> px
                reg_yxtl[:, 2] *= 180                   # theta -> degrees
                reg_yxtl[:, 3] *= n_strips              # length -> strips

                target_yxtl = target[matched_col, 2:6].clone()                      # (K, 4)

                # Adjust target length relative to prediction start position
                with torch.no_grad():
                    pred_starts = (pred[matched_row, 2] * n_strips).round().long().clamp(0, n_strips)              # (K,)
                    tgt_starts = (target[matched_col, 2] * n_strips).round().long()      # (K,)
                    target_yxtl[:, 3] -= (pred_starts - tgt_starts).float() / n_strips

                target_yxtl[:, 0] *= n_strips           # start_y -> strips
                target_yxtl[:, 1] *= (img_w - 1)       # start_x -> px
                target_yxtl[:, 2] *= 180                # theta -> degrees
                target_yxtl[:, 3] *= n_strips           # length -> strips

                reg_xytl_loss = reg_xytl_loss + torch.nn.functional.smooth_l1_loss(
                    reg_yxtl, target_yxtl, reduction='none').mean()

                # --- Line IoU ---
                pred_xs = pred[matched_row, 6:] * (img_w - 1)                       # (K, 72) px
                tgt_xs = target[matched_col, 6:] * (img_w - 1)                      # (K, 72) px
                iou_loss = iou_loss + (1 - self.line_iou(
                    pred_xs, tgt_xs, img_w, length=15, aligned=True)).mean()

        denom = bs * num_stages
        cls_loss /= denom
        reg_xytl_loss /= denom
        iou_loss /= denom

        total = (cls_loss * LaneDetectionLossCalculator.cls_loss_weight +
                 reg_xytl_loss * LaneDetectionLossCalculator.xyt_loss_weight +
                 iou_loss * LaneDetectionLossCalculator.iou_loss_weight).reshape(1)

        return total, {
            'lane_cls_loss': (cls_loss * LaneDetectionLossCalculator.cls_loss_weight).item(),
            'lane_reg_loss': (reg_xytl_loss * LaneDetectionLossCalculator.xyt_loss_weight).item(),
            'lane_iou_loss': (iou_loss * LaneDetectionLossCalculator.iou_loss_weight).item(),
        }

    def __call__(self, loss_items:LaneDetectionLossItems):
        
        img_h, img_w = loss_items.image_size
        
        lane_det_loss, lane_det_loss_items = self.compute_lane_det_loss(
            loss_items.lane_detection_logits,
            loss_items.targets_detections,
            img_w=img_w, img_h=img_h            
        )

        lane_det_loss += LaneDetectionLossCalculator.loss_weights.get("lane_detection", 1.0)

        if loss_items.lane_seg_logits is not None and loss_items.targets_seg_masks is not None:
            seg_loss = torch.nn.functional.nll_loss(
                torch.nn.functional.log_softmax(loss_items.lane_seg_logits, dim=1),
                loss_items.targets_seg_masks
            )

            lane_det_loss += seg_loss * LaneDetectionLossCalculator.loss_weights.get("lane_seg", 1.0)
            lane_det_loss_items["lane_seg_loss"] = seg_loss.item()

        return lane_det_loss, lane_det_loss_items