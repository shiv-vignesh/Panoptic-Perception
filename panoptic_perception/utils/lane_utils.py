"""
CLRNet-style lane target utilities.
Converts augmented lane polylines to CLRNet target format.
"""

import numpy as np
import torch
from scipy.interpolate import interp1d

from enum import Enum

class BDD100KLaneCategories(Enum):
    SINGLE_WHITE = 0
    SINGLE_YELLOW = 1
    DOUBLE_YELLOW = 2
    ROAD_CURB = 3

    @classmethod
    def from_id(cls, class_id: int):
        try:
            return cls(class_id).name.lower()
        except ValueError:
            return None

    @classmethod
    def from_label(cls, label: str):
        """Map BDD100K lane category string to class id.

        Accepts either the raw BDD100K format (e.g. 'lane/single white')
        or the enum name (e.g. 'SINGLE_WHITE').
        """
        if label.startswith("lane/"):
            label = label[len("lane/"):]
        return cls[label.upper().replace(" ", "_")].value

NUM_LANE_POINTS = 72
MAX_LANES = 8


def polyline_to_lane_target(points, img_h, img_w, n_offsets=NUM_LANE_POINTS):
    """
    Convert a single lane polyline to a CLRNet-style target vector.

    Args:
        points: np.array of shape (N, 2) [x, y] in pixel coordinates
        img_h, img_w: image dimensions (after augmentation)
        n_offsets: number of uniformly sampled y positions (72)

    Returns:
        target: np.array of shape (4 + n_offsets,)
                [start_y, start_x, theta, length, x0, x1, ..., x71]
                All values normalized. Invalid x positions = -1e5
        or None if lane is too short / invalid
    """
    if len(points) < 2:
        return None

    pts = points.copy()

    # Sort by y descending (bottom of image = largest y = "start" of lane)
    pts = pts[pts[:, 1].argsort()[::-1]]

    # Remove duplicate y values (keep first occurrence)
    _, unique_idx = np.unique(pts[:, 1], return_index=True)
    pts = pts[np.sort(unique_idx)]
    pts = pts[pts[:, 1].argsort()[::-1]]

    if len(pts) < 2:
        return None

    # Build interpolation function: y -> x (y ascending for interp1d)
    pts_asc = pts[::-1]
    try:
        interp_func = interp1d(
            pts_asc[:, 1], pts_asc[:, 0],
            kind='linear', fill_value='extrapolate'
        )
    except ValueError:
        return None

    # Sample at n_offsets uniformly spaced y positions
    # CLRNet convention: prior_ys goes from 1.0 (bottom) to 0.0 (top)
    ys_normalized = np.linspace(1.0, 0.0, n_offsets)
    ys_pixel = ys_normalized * (img_h - 1)

    # Determine valid y range from actual polyline
    y_min, y_max = pts[:, 1].min(), pts[:, 1].max()

    xs_pixel = np.full(n_offsets, -1e5)  # invalid marker
    valid_count = 0
    for i, yp in enumerate(ys_pixel):
        if y_min <= yp <= y_max:
            x = float(interp_func(yp))
            if 0 <= x < img_w:
                xs_pixel[i] = x
                valid_count += 1

    if valid_count < 2:
        return None

    # Normalize x to [0, 1]
    xs_normalized = np.where(xs_pixel > -1e4, xs_pixel / (img_w - 1), -1e5)

    # Start point = bottom-most valid point
    valid_mask = xs_pixel > -1e4
    valid_indices = np.where(valid_mask)[0]
    start_idx = valid_indices[0]

    start_y = ys_normalized[start_idx]
    start_x = xs_normalized[start_idx]

    # Compute theta from start to end direction
    end_idx = valid_indices[-1]
    dx = xs_pixel[end_idx] - xs_pixel[start_idx]
    dy = ys_pixel[end_idx] - ys_pixel[start_idx]

    theta = np.arctan2(abs(dy), dx + 1e-5)
    theta_normalized = np.clip(theta / np.pi, 0.01, 0.99)

    length = valid_count

    target = np.zeros(4 + n_offsets, dtype=np.float32)
    target[0] = start_y
    target[1] = start_x
    target[2] = theta_normalized
    target[3] = length / (n_offsets - 1)
    target[4:] = xs_normalized

    return target


def build_lane_targets(lane_polys, img_h, img_w,
                       max_lanes=MAX_LANES, n_offsets=NUM_LANE_POINTS):
    """
    Convert augmented lane polylines to CLRNet target tensors.

    Args:
        lane_polys: list of dicts {"points": np.array(N,2), "category": str}
                    points in pixel coordinates of the augmented image
        img_h, img_w: augmented image dimensions
        max_lanes: pad/truncate to this many lanes
        n_offsets: number of y-sample positions

    Returns:
        targets: Tensor (max_lanes, 6 + n_offsets)
                 [valid, cls, start_y, start_x, theta, length, x0...x71]
        categories: Tensor (max_lanes,) lane category indices (-1 for empty)
    """
    targets = np.zeros((max_lanes, 6 + n_offsets), dtype=np.float32)
    categories = np.full(max_lanes, -1, dtype=np.int64)

    if lane_polys is None:
        return torch.from_numpy(targets), torch.from_numpy(categories)

    lane_idx = 0
    for poly in lane_polys:
        if lane_idx >= max_lanes:
            break

        cat = poly["category"]
        if cat == "lane/crosswalk":
            continue  # handled by drivable seg head

        points = poly["points"]
        result = polyline_to_lane_target(points, img_h, img_w, n_offsets)
        if result is None:
            continue

        targets[lane_idx, 0] = 1.0                     # valid
        targets[lane_idx, 1] = 1.0                     # cls = positive
        targets[lane_idx, 2:6] = result[:4]             # start_y, start_x, theta, length
        targets[lane_idx, 6:] = result[4:]              # x offsets

        try:
            categories[lane_idx] = BDD100KLaneCategories.from_label(cat)
        except KeyError:
            categories[lane_idx] = 0

        lane_idx += 1

    return torch.from_numpy(targets), torch.from_numpy(categories)


# ----- Lane Detection -----
# Adapted from: https://github.com/Turoad/CLRNet

class LaneDetectionLossCalculator:

    cls_loss_weight = 2.0
    xyt_loss_weight = 0.5
    iou_loss_weight = 2.0

    @staticmethod
    def focal_loss(inputs, targets, alpha=0.25, gamma=2.0, reduction='none'):
        """
        Softmax-based focal loss for multi-class (2-class bg/fg) classification.
        Note: DetectionLossCalculator.focal_loss is BCE-based (binary), not reusable here.
        """
        # inputs: (N, C) raw logits, targets: (N,) long labels
        p = torch.softmax(inputs, dim=1) + 1e-8                                    # (N, C)
        ce = torch.nn.functional.cross_entropy(inputs, targets, reduction='none')   # (N,)
        p_t = p[torch.arange(len(targets), device=targets.device), targets]         # (N,)
        loss = alpha * (1 - p_t) ** gamma * ce                                      # (N,)
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        return loss

    @staticmethod
    def line_iou(pred, target, img_w, length=15, aligned=True):
        """
        1D lane IoU with rectangular buffer strips.
        Only target validity is checked (predictions can be out-of-bounds and
        naturally receive 0 overlap via geometry).
        """
        # pred: (N, 72) pixels, target: (N, 72) or (M, 72) pixels
        px1, px2 = pred - length, pred + length                                     # (N, 72)
        tx1, tx2 = target - length, target + length

        if aligned:
            invalid = (target < 0) | (target >= img_w)                              # (N, 72)
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

        valid = (~invalid).float()
        ovr = ovr.clamp(min=0) * valid
        union = union.clamp(min=0) * valid
        return ovr.sum(-1) / (union.sum(-1) + 1e-9)                                # (N,) or (N, M)

    @staticmethod
    def _distance_cost(predictions, targets, img_w):
        """Mean absolute x-distance between pred and GT at valid y-levels (pixel space)."""
        # predictions[:, 6:] and targets[:, 6:] already in pixel space (scaled by assign)
        pred_xs = predictions[:, 6:].unsqueeze(1)                                   # (P, 1, 72)
        tgt_xs = targets[:, 6:].unsqueeze(0)                                        # (1, T, 72)

        valid = (tgt_xs >= 0) & (tgt_xs < img_w)                                   # (P, T, 72)
        dist = torch.abs(pred_xs - tgt_xs) * valid.float()                          # (P, T, 72)
        return dist.sum(-1) / valid.sum(-1).float().clamp(min=1)                    # (P, T)

    @staticmethod
    def _focal_cost(cls_pred, gt_labels, alpha=0.25, gamma=2.0, eps=1e-12):
        """Classification cost for assignment (takes softmaxed probabilities)."""
        # cls_pred: (P, 2) softmaxed, gt_labels: (T,) all 1s for valid lanes
        neg_cost = -(1 - cls_pred + eps).log() * (1 - alpha) * cls_pred.pow(gamma)  # (P, 2)
        pos_cost = -(cls_pred + eps).log() * alpha * (1 - cls_pred).pow(gamma)      # (P, 2)
        return pos_cost[:, gt_labels] - neg_cost[:, gt_labels]                      # (P, T)

    @staticmethod
    def _dynamic_k_assign(cost, pair_wise_ious):
        """Dynamic k matching: top-4 IoU per GT determines number of assigned priors."""
        # cost: (P, T), pair_wise_ious: (P, T)
        matching = torch.zeros_like(cost)                                           # (P, T)
        n_candidate_k = min(4, cost.shape[0])

        topk_ious, _ = torch.topk(pair_wise_ious, n_candidate_k, dim=0)            # (4, T)
        dynamic_ks = topk_ious.sum(0).int().clamp(min=1)                            # (T,)

        for gt_idx in range(cost.shape[1]):
            _, pos_idx = torch.topk(
                cost[:, gt_idx], k=dynamic_ks[gt_idx].item(), largest=False)
            matching[pos_idx, gt_idx] = 1.0

        # Resolve conflicts: prior matched to multiple GTs -> keep lowest cost
        conflict_rows = torch.where(matching.sum(1) > 1)[0]                         # (C,)
        if len(conflict_rows) > 0:
            _, best_gt = cost[conflict_rows].min(dim=1)                             # (C,)
            matching[conflict_rows] = 0
            matching[conflict_rows, best_gt] = 1.0

        matched_row = matching.sum(1).nonzero().flatten()                           # (K,)
        matched_col = matching[matched_row].argmax(-1).flatten()                    # (K,)
        return matched_row, matched_col

    @staticmethod
    def assign(predictions, targets, img_w, img_h,
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
            predictions[:, 3] *= (img_w - 1)                                       # start_x -> px
            predictions[:, 6:] *= (img_w - 1)                                      # x-coords -> px
            targets[:, 3] *= (img_w - 1)                                           # start_x -> px
            targets[:, 6:] *= (img_w - 1)                                          # x-coords -> px

            # 1. Distance similarity
            distances = LaneDetectionLossCalculator._distance_cost(
                predictions, targets, img_w)                                        # (P, T)
            distances = 1 - (distances / distances.max().clamp(min=1e-4)) + 1e-2   # -> similarity

            # 2. Classification cost
            cls_pred = torch.softmax(predictions[:, :2], dim=1)                     # (P, 2)
            gt_labels = torch.ones(num_targets, dtype=torch.long,
                                   device=targets.device)                           # (T,)
            cls_cost = LaneDetectionLossCalculator._focal_cost(
                cls_pred, gt_labels)                                                # (P, T)

            # 3. Start point similarity (scale start_y to pixels for comparable L2)
            pred_start = predictions[:, 2:4].clone()                                # (P, 2)
            pred_start[:, 0] *= (img_h - 1)                                        # start_y -> px
            tgt_start = targets[:, 2:4].clone()                                     # (T, 2)
            tgt_start[:, 0] *= (img_h - 1)                                         # start_y -> px
            start_xys = torch.cdist(pred_start, tgt_start, p=2).reshape(
                num_priors, num_targets)                                            # (P, T)
            start_xys = (1 - start_xys / start_xys.max().clamp(min=1e-4)) + 1e-2  # -> similarity

            # 4. Theta similarity (scale to degrees for magnitude)
            thetas = torch.cdist(
                predictions[:, 4:5], targets[:, 4:5], p=1).reshape(
                num_priors, num_targets) * 180                                      # (P, T)
            thetas = (1 - thetas / thetas.max().clamp(min=1e-4)) + 1e-2            # -> similarity

            # Combined cost: high similarity product -> large negative -> selected by topk(largest=False)
            cost = -(distances * start_xys * thetas) ** 2 * distance_cost_weight \
                   + cls_cost * cls_cost_weight                                     # (P, T)

            # Line IoU for dynamic-k
            pair_wise_ious = LaneDetectionLossCalculator.line_iou(
                predictions[:, 6:], targets[:, 6:],
                img_w, aligned=False)                                               # (P, T)

        return LaneDetectionLossCalculator._dynamic_k_assign(cost, pair_wise_ious)

    @staticmethod
    def compute_lane_det_loss(predictions_lists, targets, img_w, img_h, n_strips=71):
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
            preds_stage = predictions_lists[stage]                                  # (bs, 192, 78)

            for b in range(bs):
                pred = preds_stage[b]                                               # (192, 78)
                target = targets[b]                                                 # (max_lanes, 78)
                target = target[target[:, 0] == 1]                                  # (T, 78) valid

                if len(target) == 0:
                    cls_target = pred.new_zeros(pred.shape[0]).long()                # (192,)
                    cls_loss = cls_loss + LaneDetectionLossCalculator.focal_loss(
                        pred[:, :2], cls_target).sum()
                    continue

                with torch.no_grad():
                    matched_row, matched_col = LaneDetectionLossCalculator.assign(
                        pred, target, img_w, img_h)

                # --- Classification ---
                cls_target = pred.new_zeros(pred.shape[0]).long()                   # (192,)
                cls_target[matched_row] = 1
                cls_loss = cls_loss + LaneDetectionLossCalculator.focal_loss(
                    pred[:, :2], cls_target                                         # (192, 2) vs (192,)
                ).sum() / target.shape[0]

                # --- Regression: start_y, start_x, theta, length ---
                reg_yxtl = pred[matched_row, 2:6].clone()                           # (K, 4)
                reg_yxtl[:, 0] *= n_strips                                          # start_y -> strips
                reg_yxtl[:, 1] *= (img_w - 1)                                      # start_x -> px
                reg_yxtl[:, 2] *= 180                                               # theta -> degrees
                reg_yxtl[:, 3] *= n_strips                                          # length -> strips

                target_yxtl = target[matched_col, 2:6].clone()                      # (K, 4)

                # Adjust target length relative to prediction start position
                with torch.no_grad():
                    pred_starts = (pred[matched_row, 2] * n_strips
                                   ).round().long().clamp(0, n_strips)              # (K,)
                    tgt_starts = (target[matched_col, 2] * n_strips
                                  ).round().long()                                  # (K,)
                    target_yxtl[:, 3] -= (pred_starts - tgt_starts).float() / n_strips

                target_yxtl[:, 0] *= n_strips                                       # start_y -> strips
                target_yxtl[:, 1] *= (img_w - 1)                                   # start_x -> px
                target_yxtl[:, 2] *= 180                                            # theta -> degrees
                target_yxtl[:, 3] *= n_strips                                       # length -> strips

                reg_xytl_loss = reg_xytl_loss + torch.nn.functional.smooth_l1_loss(
                    reg_yxtl, target_yxtl, reduction='none').mean()

                # --- Line IoU ---
                pred_xs = pred[matched_row, 6:] * (img_w - 1)                       # (K, 72) px
                tgt_xs = target[matched_col, 6:] * (img_w - 1)                      # (K, 72) px
                iou_loss = iou_loss + (1 - LaneDetectionLossCalculator.line_iou(
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


# ----- Lane NMS & Conversion -----

def lane_nms(predictions, img_w, conf_threshold=0.5, nms_threshold=0.5):
    """
    Greedy lane NMS based on line IoU.
    predictions: (N, 78) activated — [bg_prob, fg_prob, start_y, start_x, theta, length, x0..x71]
    Returns: (K, 78) surviving lanes sorted by confidence descending.
    """
    scores = predictions[:, 1]                                         # (N,) fg confidence
    keep_mask = scores > conf_threshold
    predictions = predictions[keep_mask]
    scores = scores[keep_mask]

    if len(predictions) == 0:
        return predictions

    order = scores.argsort(descending=True)
    predictions = predictions[order]
    scores = scores[order]

    xs_px = predictions[:, 6:] * (img_w - 1)                          # (K, 72) pixel space

    keep = []
    alive = torch.ones(len(predictions), dtype=torch.bool, device=predictions.device)

    for i in range(len(predictions)):
        if not alive[i]:
            continue
        keep.append(i)
        if i == len(predictions) - 1:
            break

        remaining_idx = alive.clone()
        remaining_idx[i] = False
        remaining_idx = remaining_idx.nonzero(as_tuple=True)[0]
        if len(remaining_idx) == 0:
            break

        ious = LaneDetectionLossCalculator.line_iou(
            xs_px[i:i+1].expand(len(remaining_idx), -1),
            xs_px[remaining_idx],
            img_w, aligned=True
        )                                                              # (R,)
        suppress = remaining_idx[ious > nms_threshold]
        alive[suppress] = False

    return predictions[keep]                                           # (K', 78)


def predictions_to_lanes(predictions, img_h, img_w, n_offsets=NUM_LANE_POINTS):
    """
    Convert activated + NMS'd lane predictions to polyline dicts.
    predictions: (K, 78) activated lanes
    Returns: list of dicts with keys 'points', 'confidence', 'start_y', 'length'
    """
    if len(predictions) == 0:
        return []

    lanes = []
    prior_ys = torch.linspace(1.0, 0.0, steps=n_offsets, device=predictions.device)

    for pred in predictions:
        conf = pred[1].item()                                          # fg prob
        start_y = pred[2].item()
        length = pred[5].item()
        x_coords = pred[6:]                                            # (72,) normalized

        # Valid range: from start_y downward for `length` fraction of the strip range
        start_idx = (prior_ys <= start_y + 0.01).nonzero(as_tuple=True)[0]
        if len(start_idx) == 0:
            continue
        start_idx = start_idx[0].item()
        num_valid = int(round(length * (n_offsets - 1)))
        end_idx = min(start_idx + num_valid + 1, n_offsets)

        points = []
        for j in range(start_idx, end_idx):
            x_px = x_coords[j].item() * (img_w - 1)
            y_px = prior_ys[j].item() * (img_h - 1)
            if x_px < 0 or x_px >= img_w:
                continue
            points.append((x_px, y_px))

        if len(points) < 2:
            continue

        lanes.append({
            'points': points,
            'confidence': conf,
            'start_y': start_y,
            'length': length
        })

    return lanes


def _make_activated_gt(valid_targets):
    """
    Format GT targets for reuse with predictions_to_lanes().
    valid_targets: (T, 78) with [valid=1, category, start_y, start_x, theta, length, x0..x71]
    Returns: (T, 78) with [bg_prob=0, fg_prob=1, start_y, start_x, theta, length, x0..x71]
    """
    activated = valid_targets.clone()
    activated[:, 0] = 0.0   # bg prob
    activated[:, 1] = 1.0   # fg prob (confidence)
    return activated


def polyline_iou(pred_points, gt_points, img_h, img_w, lane_width=15):
    """
    Compute IoU between two polylines by interpolating to a common y-grid.
    pred_points, gt_points: list of (x, y) tuples in pixel space
    Returns: float IoU value
    """
    if len(pred_points) < 2 or len(gt_points) < 2:
        return 0.0

    pred_ys = [p[1] for p in pred_points]
    pred_xs = [p[0] for p in pred_points]
    gt_ys = [p[1] for p in gt_points]
    gt_xs = [p[0] for p in gt_points]

    # Common y range
    y_min = max(min(pred_ys), min(gt_ys))
    y_max = min(max(pred_ys), max(gt_ys))

    if y_max <= y_min:
        return 0.0

    # Sample at integer y positions in the shared range
    ys = np.arange(int(np.ceil(y_min)), int(np.floor(y_max)) + 1)
    if len(ys) == 0:
        return 0.0

    # Interpolate x at shared y positions
    try:
        pred_interp = interp1d(pred_ys, pred_xs, bounds_error=False, fill_value="extrapolate")
        gt_interp = interp1d(gt_ys, gt_xs, bounds_error=False, fill_value="extrapolate")
    except ValueError:
        return 0.0

    pred_x_at_ys = pred_interp(ys)
    gt_x_at_ys = gt_interp(ys)

    # 1D IoU with buffer strips at each y
    px1, px2 = pred_x_at_ys - lane_width, pred_x_at_ys + lane_width
    tx1, tx2 = gt_x_at_ys - lane_width, gt_x_at_ys + lane_width

    ovr = np.maximum(0, np.minimum(px2, tx2) - np.maximum(px1, tx1))
    union = np.maximum(px2, tx2) - np.minimum(px1, tx1)
    union = np.maximum(union, 1e-9)

    return float(ovr.sum() / union.sum())