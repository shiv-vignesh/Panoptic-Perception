from panoptic_perception.models.models import YOLOP
import torch
import numpy as np

# ------- YOLOP Model Configuration ------- #
# ----- https://github.com/hustvl/YOLOP/blob/main/lib/models/YOLOP.py#L553 ---- #

YOLOP_CFG = [
    [24, 33, 42],   #Det_out_idx, Da_Segout_idx, LL_Segout_idx
    [ -1, "Focus", [3, 32, 3]],   #0
    [ -1, "Conv", [32, 64, 3, 2]],    #1
    [ -1, "BottleneckCSP", [64, 64, 1]],  #2
    [ -1, "Conv", [64, 128, 3, 2]],   #3
    [ -1, "BottleneckCSP", [128, 128, 3]],    #4
    [ -1, "Conv", [128, 256, 3, 2]],  #5
    [ -1, "BottleneckCSP", [256, 256, 3]],    #6
    [ -1, "Conv", [256, 512, 3, 2]],  #7
    [ -1, "SPP", [512, 512, [5, 9, 13]]],     #8
    [ -1, "BottleneckCSP", [512, 512, 1, False]],     #9
    [ -1, "Conv",[512, 256, 1, 1]],   #10
    [ -1, "Upsample", [None, 2, 'nearest']],  #11
    [ [-1, 6], "Concat", [1]],    #12
    [ -1, "BottleneckCSP", [512, 256, 1, False]], #13
    [ -1, "Conv", [256, 128, 1, 1]],  #14
    [ -1, "Upsample", [None, 2, 'nearest']],  #15
    [ [-1,4], "Concat", [1]],     #16         #Encoder

    [ -1, "BottleneckCSP", [256, 128, 1, False]],     #17
    [ -1, "Conv", [128, 128, 3, 2]],      #18
    [ [-1, 14], "Concat", [1]],       #19
    [ -1, "BottleneckCSP", [256, 256, 1, False]],     #20
    [ -1, "Conv", [256, 256, 3, 2]],      #21
    [ [-1, 10], "Concat", [1]],   #22
    [ -1, "BottleneckCSP", [512, 512, 1, False]],     #23
    [ [17, 20, 23], "Detect",  [1, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [128, 256, 512]]], #Detection head 24

    [ 16, "Conv", [256, 128, 3, 1]],   #25
    [ -1, "Upsample", [None, 2, 'nearest']],  #26
    [ -1, "BottleneckCSP", [128, 64, 1, False]],  #27
    [ -1, "Conv", [64, 32, 3, 1]],    #28
    [ -1, "Upsample", [None, 2, 'nearest']],  #29
    [ -1, "Conv", [32, 16, 3, 1]],    #30
    [ -1, "BottleneckCSP", [16, 8, 1, False]],    #31
    [ -1, "Upsample", [None, 2, 'nearest']],  #32
    [ -1, "Conv", [8, 2, 3, 1]], #33 Driving area segmentation head
    
    [ 16, "Conv", [256, 128, 3, 1]],   #33
    [ -1, "Upsample", [None, 2, 'nearest']],  #34
    [ -1, "BottleneckCSP", [128, 64, 1, False]],  #35
    [ -1, "Conv", [64, 32, 3, 1]],    #36
    [ -1, "Upsample", [None, 2, 'nearest']],  #37
    [ -1, "Conv", [32, 16, 3, 1]],    #38
    [ -1, "BottleneckCSP", [16, 8, 1, False]],    #39
    [ -1, "Upsample", [None, 2, 'nearest']],  #40
    [ -1, "Conv", [8, 2, 3, 1]], #41 Driving area segmentation head
]

def create_detection_targets(batch_size=2, num_targets_per_image=5, num_classes=3):
    """
    Create detection targets for testing YOLOP.

    YOLOP anchors are designed for small objects (vehicles on roads):
    - Layer 0 (stride=8):  anchors ~3-20 pixels  (for small, distant objects)
    - Layer 1 (stride=16): anchors ~6-39 pixels  (for medium objects)
    - Layer 2 (stride=32): anchors ~19-157 pixels (for large, close objects)

    Args:
        batch_size: Number of images in batch
        num_targets_per_image: Number of objects per image
        num_classes: Number of object classes

    Returns:
        torch.Tensor: Shape (num_targets, 6) where each row is:
                      [batch_idx, class, x_center, y_center, width, height]
                      All coordinates normalized to [0, 1]
    """
    targets_list = []

    np.random.seed(42)

    for batch_idx in range(batch_size):
        for _ in range(num_targets_per_image):
            # Random class
            class_id = np.random.randint(0, num_classes)

            # Random bbox (normalized coordinates)
            # IMPORTANT: Keep boxes small to match YOLOP's anchor design
            # Typical vehicle on road: 20-100 pixels on 640px image = 0.03-0.15 normalized
            x_center = np.random.uniform(0.2, 0.8)
            y_center = np.random.uniform(0.4, 0.9)  # Vehicles typically in lower half

            # Small boxes that match anchor sizes
            # Width: 10-80 pixels on 640px image = 0.015-0.125 normalized
            # Height: 10-60 pixels on 640px image = 0.015-0.094 normalized
            width = np.random.uniform(0.02, 0.12)   # Reduced from 0.05-0.3
            height = np.random.uniform(0.02, 0.10)  # Reduced from 0.05-0.3

            # Create target: [batch_idx, class, x, y, w, h]
            target = [float(batch_idx), float(class_id), x_center, y_center, width, height]
            targets_list.append(target)

    targets = torch.tensor(targets_list, dtype=torch.float32)

    return targets

def test_parse_basic_structure():
    """Test that parse_model_config_2 returns correct basic structure."""
    model = YOLOP(YOLOP_CFG)
    module_defs = model.module_defs

    assert isinstance(module_defs, list), "Module definitions should be a list"
    assert len(module_defs) > 0, "Module definitions should not be empty"
    
    return model

def parse_cfg_file():
    cfg_path = 'panoptic_perception/configs/models/yolop.cfg'
    model = YOLOP(cfg_path)
    module_defs = model.module_defs

    assert isinstance(module_defs, list), "Module definitions should be a list"
    assert len(module_defs) > 0, "Module definitions should not be empty"
    
    return model
    
def test_parse_cfg_file():
    return parse_cfg_file()
    
def test_forward_pass():
    """Test forward pass in evaluation mode."""
    model = test_parse_cfg_file()
    # model = test_parse_basic_structure()

    model.eval()

    x = torch.randn(2, model.in_channels, 640, 640)

    predictions = model(x)
    print(f"\nEvaluation mode outputs: {predictions.keys()}")

    return predictions


def test_forward_pass_with_targets():
    """Test forward pass in training mode with targets."""
    model = test_parse_cfg_file()
    model.train()

    batch_size = 2
    img_size = 640

    # Create input
    x = torch.randn(batch_size, model.in_channels, img_size, img_size)

    # Create detection targets: (num_targets, 6) -> [batch_idx, class, x, y, w, h]
    det_targets = create_detection_targets(
        batch_size=batch_size,
        num_targets_per_image=5,
        num_classes=1  # YOLOP uses 1 class for detection
    )

    print(f"\nDetection targets shape: {det_targets.shape}")
    print(f"First 3 targets:\n{det_targets[:3]}")

    # Create segmentation targets (placeholder for now)
    da_seg_targets = torch.randint(0, 2, (batch_size, img_size, img_size), dtype=torch.long)
    lane_seg_targets = torch.randint(0, 2, (batch_size, img_size, img_size), dtype=torch.long)

    # Targets dict
    targets = {
        "detections": det_targets,
        "drivable_area_seg": da_seg_targets,
        "lane_seg": lane_seg_targets
    }
    
    model.eval()

    # Forward pass
    print(f"\nRunning forward pass with targets...")
    outputs = model(x, targets=targets)

    print(f"\nTraining mode outputs: {outputs.keys()}")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape if hasattr(value, 'shape') else value}")
        elif isinstance(value, list):
            print(f"  {key}: list of {len(value)} tensors")

    return outputs


if __name__ == "__main__":
    # Test evaluation mode
    # test_forward_pass()

    # Test training mode with targets
    test_forward_pass_with_targets()