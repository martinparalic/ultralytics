# Ultralytics YOLO 🚀, AGPL-3.0 license
from typing import List

import torch

from ultralytics.models.fastsam.utils import bbox_iou
from ultralytics.models.yolo.segment import SegmentationPredictor


class FastSAMPredictor(SegmentationPredictor):
    """
    FastSAMPredictor is specialized for fast SAM (Segment Anything Model) segmentation prediction tasks in Ultralytics
    YOLO framework.

    This class extends the SegmentationPredictor, customizing the prediction pipeline specifically for fast SAM. It
    adjusts post-processing steps to incorporate mask prediction and non-max suppression while optimizing for single-
    class segmentation.
    """

    def postprocess(self, preds, img, orig_imgs):
        """Applies box postprocess for FastSAM predictions."""
        results = super().postprocess(preds, img, orig_imgs)
        for result in results:
            full_box = torch.tensor([0, 0, result.orig_shape[1], result.orig_shape[0]], device=preds[0].device, dtype=torch.float32)
            idx = bbox_iou(full_box, result.boxes.xyxy, iou_thres=0.9, image_shape=result.orig_shape)
            if idx.numel() != 0:
                result.boxes.xyxy[idx] = full_box
        return results
