import torch

from panoptic_perception.utils.evaluation_helper import DetectionMetrics


def test_compute_stats_counts_false_negatives_when_image_has_no_predictions():
    detections_by_image = {0: None}
    gt_by_image = {
        0: torch.tensor([[0.0, 0.0, 10.0, 10.0, 0.0]], dtype=torch.float32)
    }

    ap_results, stats_per_class = DetectionMetrics.compute_stats(
        detections_by_image,
        gt_by_image,
        iou_threshold=0.5,
        num_classes=1,
    )

    assert ap_results["AP_class_0"] == 0.0
    assert ap_results["mAP"] == 0.0
    assert stats_per_class[0]["total_gt"] == 1.0
    assert stats_per_class[0]["true_positives"] == 0.0
    assert stats_per_class[0]["false_positives"] == 0.0
    assert stats_per_class[0]["false_negatives"] == 1.0

