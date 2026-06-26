import json

from panoptic_perception.dataset.bdd100k_dataset import BDDPreprocessor


def test_load_detection_preserves_xyxy_order(tmp_path):
    label_path = tmp_path / "sample.json"
    label_path.write_text(json.dumps({
        "frames": [{
            "objects": [{
                "category": "car",
                "box2d": {
                    "x1": 10.0,
                    "y1": 20.0,
                    "x2": 110.0,
                    "y2": 220.0,
                },
            }],
        }],
        "attributes": {"weather": "clear"},
    }))

    detections = BDDPreprocessor({}).load_detection(str(label_path))

    assert len(detections.detections) == 1
    assert detections.detections[0].bbox.to_list() == [10.0, 20.0, 110.0, 220.0]
