"""Evaluation metrics for segmentation models."""

import numpy as np
import torch
from monai.metrics import DiceMetric


class SegmentationEvaluator:
    """Simple evaluator with Dice and IoU metrics."""

    def __init__(self, num_classes: int, include_background: bool = False):
        self.num_classes = num_classes
        self.dice_metric = DiceMetric(include_background=include_background, reduction="mean")
        self.results = {"dice": [], "iou": []}

    def compute_dice(self, prediction: np.ndarray, target: np.ndarray) -> float:
        """Compute Dice coefficient."""
        pred_tensor = torch.from_numpy(prediction).unsqueeze(0).unsqueeze(0).float()
        target_tensor = torch.from_numpy(target).unsqueeze(0).unsqueeze(0).float()
        dice = self.dice_metric(pred_tensor, target_tensor)
        return dice.item() if isinstance(dice, torch.Tensor) else dice

    def compute_iou(self, prediction: np.ndarray, target: np.ndarray) -> float:
        """Compute Intersection over Union."""
        intersection = np.logical_and(prediction, target).sum()
        union = np.logical_or(prediction, target).sum()
        return intersection / union if union > 0 else 0.0

    def evaluate(self, predictions: np.ndarray, targets: np.ndarray) -> dict:
        """Evaluate predictions against targets."""
        dice = self.compute_dice(predictions, targets)
        iou = self.compute_iou(predictions, targets)

        metrics = {"dice": dice, "iou": iou}
        self.results["dice"].append(dice)
        self.results["iou"].append(iou)

        return metrics

    def get_summary(self) -> dict:
        """Get summary statistics."""
        summary = {}
        for metric_name, values in self.results.items():
            if values:
                arr = np.array(values)
                summary[metric_name] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                }
        return summary

    def reset(self):
        """Reset stored results."""
        self.results = {"dice": [], "iou": []}
