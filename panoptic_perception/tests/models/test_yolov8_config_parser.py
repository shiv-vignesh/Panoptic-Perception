from panoptic_perception.models.models import YOLOv8P
from panoptic_perception.models.common import DetectV8
from panoptic_perception.utils.detection_utils import DetectionLossCalculator

from pprint import pprint
import torch

def parse_cfg_file():
    cfg_path = 'panoptic_perception/configs/models/yolov8-detection.cfg'
    model = YOLOv8P(cfg_path)
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

    print(model)

    x = torch.randn(2, model.in_channels, 640, 640)

    predictions = model(x)
    print(f"\nEvaluation mode outputs: {predictions.keys()}")
    
    if predictions.detection_predictions is not None:
        for detection in predictions.detection_predictions:
            print(detection.shape)

    return predictions

def test_detection_head():
    
    in_channels=[512, 256, 256]
    shapes = [(80, 80), (40, 40), (20, 20)]
    strides = [8, 16, 32]
    image_size = (640, 640)
        
    targets = torch.tensor([
        [0, 1, 125., 180., 150., 240.],  # (50,60,200,300)
        [0, 0, 185., 195., 130., 230.],  # (120,80,250,310)
        [0, 4,  65.,  80.,  70.,  80.],  # (30,40,100,120)
        [0, 2,  50.,  65.,  80.,  90.],  # (10,20,90,110)
        [0, 3, 275., 330., 250., 340.],  # (150,160,400,500)
    ], dtype=torch.float32)

    targets[:, 2:] = targets[:, 2:]/image_size[0]

    detect_v8 = DetectV8(num_classes=6, 
                        in_channels=in_channels)
    
    logits = []

    for i, in_channel in enumerate(in_channels):
        shape = shapes[i]
        stride = strides[i]

        x = torch.rand(size=(1, in_channel, shape[0], shape[1]))
        anchor_tensor, stride_tensor = detect_v8.compute_anchors(shape[0], shape[1], stride)

        logits.append(x)

        print(f'[TEST] {shape[0]}x{shape[1]}   {anchor_tensor.shape} {stride_tensor.shape}')

    bbox_outputs, cls_outputs, anchor_points, strides = detect_v8(logits, image_size)
    print('[TEST]', bbox_outputs.shape, cls_outputs.shape, anchor_points.shape, strides.shape)

    loss, loss_dict = DetectionLossCalculator.compute_detection_loss_v8(
        cls_outputs, bbox_outputs,
        anchor_points, strides,
        targets.to(bbox_outputs.device), image_size
    )
    
    print(loss, loss_dict)

if __name__ == "__main__":
    
    # test_forward_pass()
    test_detection_head()