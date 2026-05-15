import torch
import math

from collections import defaultdict

from panoptic_perception.models.common import LaneDetect
from panoptic_perception.models.models import YOLOP


def create_sample_targets(batch_size, max_lanes=5, img_h=640, img_w=640,
                          num_points=72, lanes_per_image=3):
    """
    Create synthetic CLRNet-style lane targets for unit testing.

    Target format per lane (78 values):
        [0]    = valid flag (1.0 = real lane, 0.0 = padding)
        [1]    = category (0-3 for BDD100K lane types)
        [2]    = start_y (normalized, where the lane begins from bottom)
        [3]    = start_x (normalized, x-position at start_y)
        [4]    = theta (normalized angle, maps to [0, pi])
        [5]    = length (fraction of y-levels with valid x-coords)
        [6:78] = x-coordinates at 72 y-levels (normalized by img_w-1, invalid = -1e5)

    Returns:
        targets: (batch_size, max_lanes, 78)
    """
    n_strips = num_points - 1
    targets = torch.zeros(batch_size, max_lanes, 2 + 4 + num_points)

    # y-levels: bottom (1.0) to top (0.0), same as LaneDetect.prior_ys
    ys = torch.linspace(1, 0, num_points)

    # Predefined lane templates (realistic driving scenarios)
    lane_templates = [
        # (start_x, theta, curve, end_y) - typical highway lanes
        (0.25, 0.45, -0.10, 0.55),  # left lane, slight left curve
        (0.40, 0.50,  0.00, 0.60),  # center-left, straight
        (0.50, 0.50,  0.00, 0.65),  # center, straight
        (0.60, 0.50,  0.05, 0.60),  # center-right, slight right curve
        (0.75, 0.55,  0.10, 0.55),  # right lane, slight right curve
    ]

    categories = [0, 1, 2, 3]  # single white, single yellow, double yellow, road curb

    for b in range(batch_size):
        num_lanes = min(lanes_per_image, max_lanes)

        # Pick lanes (offset slightly per batch for variety)
        selected = torch.randperm(len(lane_templates))[:num_lanes]

        for l_idx, tmpl_idx in enumerate(selected):
            start_x, theta, curve, end_y = lane_templates[tmpl_idx]

            # Add slight randomness
            start_x += (torch.rand(1).item() - 0.5) * 0.05
            theta += (torch.rand(1).item() - 0.5) * 0.05
            start_y = torch.rand(1).item() * 0.05  # near bottom

            # Compute x-coordinates at each y-level
            xs = torch.full((num_points,), -1e5)
            valid_count = 0

            for i in range(num_points):
                y = ys[i].item()
                if y < start_y or y > end_y:
                    continue

                # Line equation: x = start_x + dy/tan(theta*pi) + curve*dy^2
                dy = y - start_y
                x = start_x + dy * img_h / (math.tan(theta * math.pi + 1e-5) * (img_w - 1))
                x += curve * dy * dy  # quadratic curve

                if 0.0 <= x <= 1.0:
                    xs[i] = x
                    valid_count += 1

            length = valid_count / n_strips

            targets[b, l_idx, 0] = 1.0          # valid lane
            targets[b, l_idx, 1] = categories[l_idx % len(categories)]
            targets[b, l_idx, 2] = start_y
            targets[b, l_idx, 3] = start_x
            targets[b, l_idx, 4] = theta
            targets[b, l_idx, 5] = length
            targets[b, l_idx, 6:] = xs

    return targets


def test_forward_pass():
    
    in_channels = [256, 512, 512]
    feature_map_sizes = [(80, 80), (40, 40), (20, 20)]
    bs = 8
    
    img_w, img_h = 640, 640

    lane_detect = LaneDetect(in_channels=in_channels, 
                             img_h=img_h,
                             img_w=img_w)

    print(f'lane detection - Priors: {lane_detect.priors.shape} Priors_on_featmap: {lane_detect.priors_on_featmap.shape}')

    features = []
    for (feat_h, feat_w), ch in zip(feature_map_sizes, in_channels):
        features.append(
            torch.rand(bs, ch, feat_h, feat_w)
        )

    lane_detect(
        features, (img_h, img_w)
    )    

def test_integrated_model():
    
    cfg_path = "/Users/shivvignesh/Documents/PersonalProjects/PanopticPerceptionProject/panoptic_perception/configs/models/yolo-lane-detect.cfg"
    model = YOLOP(
        cfg_path
    )
    
    bs = 8
    img_h = 640
    img_w = 640
    max_lanes = 5
    
    targets = create_sample_targets(bs, max_lanes=max_lanes,
                                    img_h=img_h, img_w=img_w)    

    targets_dict = defaultdict()
    targets_dict["lanes_detections"] = targets
    targets_dict["detections"] = None
    targets_dict["drivable_area_seg"] = None

    x = torch.rand(bs, 3, img_h, img_w)
    model(x, targets_dict)


def test_target_creation():
    """Verify synthetic targets have correct shapes and valid ranges."""
    bs = 4
    max_lanes = 5
    img_h, img_w = 640, 640

    targets = create_sample_targets(bs, max_lanes=max_lanes,
                                    img_h=img_h, img_w=img_w)

    assert targets.shape == (bs, max_lanes, 78), f"Expected (4,5,78), got {targets.shape}"

    # Check valid lanes have reasonable values
    for b in range(bs):
        for l in range(max_lanes):
            if targets[b, l, 0] == 1.0:
                start_y = targets[b, l, 2]
                start_x = targets[b, l, 3]
                theta = targets[b, l, 4]
                length = targets[b, l, 5]
                xs = targets[b, l, 6:]

                assert 0 <= start_y <= 0.1, f"start_y={start_y} out of range"
                assert 0 <= start_x <= 1.0, f"start_x={start_x} out of range"
                assert 0 < theta < 1.0, f"theta={theta} out of range"
                assert length > 0, f"length should be > 0 for valid lane"

                valid_xs = xs[xs > -1e4]
                assert len(valid_xs) > 0, "Valid lane should have some valid x-coords"
                assert (valid_xs >= 0).all() and (valid_xs <= 1).all(), \
                    f"Valid x-coords should be in [0,1], got min={valid_xs.min()}, max={valid_xs.max()}"

    print(f"Target shape: {targets.shape}")
    print(f"Valid lanes per image: {[int((targets[b, :, 0] == 1).sum()) for b in range(bs)]}")
    print(f"Sample lane [0,0]: start_y={targets[0,0,2]:.3f} start_x={targets[0,0,3]:.3f} "
          f"theta={targets[0,0,4]:.3f} length={targets[0,0,5]:.3f}")


def test_forward_with_targets():
    """Test forward pass output shape matches target shape for loss computation."""
    in_channels = [256, 512, 512]
    feature_map_sizes = [(80, 80), (40, 40), (20, 20)]
    bs = 4
    max_lanes = 5
    img_h, img_w = 640, 640

    lane_detect = LaneDetect(in_channels=in_channels, img_h=img_h, img_w=img_w)
    lane_detect.train()

    features = []
    for (feat_h, feat_w), ch in zip(feature_map_sizes, in_channels):
        features.append(torch.rand(bs, ch, feat_h, feat_w))

    predictions_lists = lane_detect(features, (img_h, img_w))
    targets = create_sample_targets(bs, max_lanes=max_lanes,
                                    img_h=img_h, img_w=img_w)

    # predictions_lists: List[(bs, 192, 78)] per stage, targets: (bs, max_lanes, 78)
    assert isinstance(predictions_lists, list), "Forward should return list of stages"
    assert len(predictions_lists) == 3, f"Expected 3 stages, got {len(predictions_lists)}"

    predictions = predictions_lists[-1]  # last stage
    assert predictions.shape == (bs, 192, 78), f"Pred shape: {predictions.shape}"
    assert targets.shape == (bs, max_lanes, 78), f"Target shape: {targets.shape}"

    # Verify prediction components are separable
    cls_logits = predictions[:, :, :2]       # (bs, 192, 2)
    x_offsets  = predictions[:, :, 6:]       # (bs, 192, 72)

    assert cls_logits.shape == (bs, 192, 2)
    assert x_offsets.shape == (bs, 192, 72)

    print(f"Stages: {len(predictions_lists)}, Last stage: {predictions.shape}, Targets: {targets.shape}")
    print(f"Cls logits: {cls_logits.shape}, Pred x-offsets: {x_offsets.shape}")


def test_loss_computation():
    """Test full forward + loss pipeline with synthetic targets."""
    from panoptic_perception.utils.lane_utils import LaneDetectionLossCalculator

    in_channels = [256, 512, 512]
    feature_map_sizes = [(80, 80), (40, 40), (20, 20)]
    bs = 4
    max_lanes = 5
    img_h, img_w = 640, 640

    lane_detect = LaneDetect(in_channels=in_channels, img_h=img_h, img_w=img_w)
    lane_detect.train()

    features = []
    for (feat_h, feat_w), ch in zip(feature_map_sizes, in_channels):
        features.append(torch.rand(bs, ch, feat_h, feat_w))

    predictions_lists = lane_detect(features, (img_h, img_w))         # List[(bs, 192, 78)]
    targets = create_sample_targets(bs, max_lanes=max_lanes,
                                    img_h=img_h, img_w=img_w)        # (bs, max_lanes, 78)

    total_loss, loss_dict = LaneDetectionLossCalculator.compute_lane_det_loss(
        predictions_lists, targets, img_w, img_h
    )

    assert total_loss.requires_grad, "Loss should have gradients"
    assert total_loss.item() > 0, f"Loss should be positive, got {total_loss.item()}"
    assert 'lane_cls_loss' in loss_dict
    assert 'lane_reg_loss' in loss_dict
    assert 'lane_iou_loss' in loss_dict

    # Verify backward pass works
    total_loss.backward()
    grad_norms = {name: p.grad.norm().item()
                  for name, p in lane_detect.named_parameters()
                  if p.grad is not None}
    assert len(grad_norms) > 0, "Should have gradients after backward"

    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Loss breakdown: { {k: f'{v:.4f}' for k, v in loss_dict.items()} }")
    print(f"Parameters with gradients: {len(grad_norms)}")


def test_loss_no_targets():
    """Test loss with zero valid lanes (all padding)."""
    from panoptic_perception.utils.lane_utils import LaneDetectionLossCalculator

    bs = 2
    img_h, img_w = 640, 640

    # All-zero targets = no valid lanes
    targets = torch.zeros(bs, 5, 78)
    predictions = torch.randn(bs, 192, 78, requires_grad=True)

    total_loss, loss_dict = LaneDetectionLossCalculator.compute_lane_det_loss(
        predictions, targets, img_w, img_h
    )

    assert total_loss.requires_grad
    assert loss_dict['lane_reg_loss'] == 0.0, "No regression loss with 0 targets"
    assert loss_dict['lane_iou_loss'] == 0.0, "No IoU loss with 0 targets"
    assert loss_dict['lane_cls_loss'] > 0.0, "Should still have cls loss (all bg)"

    print(f"No-target loss: {total_loss.item():.4f} (cls only)")


def main():

    # test_target_creation()
    # test_forward_with_targets()
    # test_loss_computation()
    # test_loss_no_targets()

    test_integrated_model()

if __name__ == "__main__":
    main()