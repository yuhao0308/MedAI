"""
Model Evaluation and Metrics
Computes segmentation metrics and generates evaluation reports
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from monai.metrics import DiceMetric, HausdorffDistanceMetric
import torch

logger = logging.getLogger(__name__)


def compute_metrics(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    num_classes: Optional[int] = None,
    include_hausdorff: bool = False,
    spacing: Optional[Tuple[float, float, float]] = None
) -> Dict[str, float]:
    """
    Computes comprehensive segmentation metrics.
    
    Args:
        prediction: Prediction mask (H, W, D) with class indices
        ground_truth: Ground truth mask (H, W, D) with class indices
        num_classes: Number of classes (if None, inferred from data)
        include_hausdorff: Whether to compute Hausdorff distance (slow)
        spacing: Voxel spacing in mm for Hausdorff distance
        
    Returns:
        Dictionary with metrics
    """
    if num_classes is None:
        num_classes = max(int(prediction.max()), int(ground_truth.max())) + 1
    
    metrics = {}
    
    # Flatten arrays
    pred_flat = prediction.flatten()
    gt_flat = ground_truth.flatten()
    
    # Overall accuracy
    accuracy = np.sum(pred_flat == gt_flat) / len(pred_flat)
    metrics["accuracy"] = float(accuracy)
    
    # Per-class metrics
    class_metrics = {}
    
    for c in range(num_classes):
        pred_c = (prediction == c).astype(np.uint8)
        gt_c = (ground_truth == c).astype(np.uint8)
        
        # Skip if class doesn't exist in either
        if np.sum(pred_c) == 0 and np.sum(gt_c) == 0:
            continue
        
        # Dice coefficient
        intersection = np.sum(pred_c & gt_c)
        union = np.sum(pred_c) + np.sum(gt_c)
        dice = 2.0 * intersection / (union + 1e-8)
        
        # IoU (Jaccard Index)
        intersection_iou = np.sum(pred_c & gt_c)
        union_iou = np.sum(pred_c | gt_c)
        iou = intersection_iou / (union_iou + 1e-8)
        
        # Sensitivity (Recall)
        tp = intersection
        fn = np.sum((gt_c == 1) & (pred_c == 0))
        sensitivity = tp / (tp + fn + 1e-8)
        
        # Specificity
        tn = np.sum((gt_c == 0) & (pred_c == 0))
        fp = np.sum((pred_c == 1) & (gt_c == 0))
        specificity = tn / (tn + fp + 1e-8)
        
        # Precision
        precision = tp / (tp + fp + 1e-8)
        
        class_metrics[f"class_{c}"] = {
            "dice": float(dice),
            "iou": float(iou),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "precision": float(precision),
            "volume_pred": int(np.sum(pred_c)),
            "volume_gt": int(np.sum(gt_c))
        }
        
        # Hausdorff distance (for foreground classes only)
        if include_hausdorff and c > 0 and spacing is not None:
            try:
                hausdorff = compute_hausdorff_distance(
                    pred_c, gt_c, spacing
                )
                class_metrics[f"class_{c}"]["hausdorff_95"] = float(hausdorff)
            except Exception as e:
                logger.warning(f"Failed to compute Hausdorff for class {c}: {e}")
    
    metrics["per_class"] = class_metrics
    
    # Average Dice (excluding background)
    if num_classes > 1:
        dice_scores = [
            class_metrics[f"class_{c}"]["dice"]
            for c in range(1, num_classes)
            if f"class_{c}" in class_metrics
        ]
        if dice_scores:
            metrics["mean_dice"] = float(np.mean(dice_scores))
            metrics["std_dice"] = float(np.std(dice_scores))
    
    return metrics


def compute_hausdorff_distance(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    spacing: Tuple[float, float, float]
) -> float:
    """
    Computes 95th percentile Hausdorff distance.
    
    Args:
        pred_mask: Binary prediction mask
        gt_mask: Binary ground truth mask
        spacing: Voxel spacing in mm
        
    Returns:
        95th percentile Hausdorff distance in mm
    """
    from scipy.ndimage import distance_transform_edt
    
    # Get surface points
    pred_surface = pred_mask.astype(bool) & ~np.isclose(
        np.pad(pred_mask, 1, mode='constant'), pred_mask[..., None, None, None]
    ).any(axis=(0, 1, 2))[1:-1, 1:-1, 1:-1]
    
    gt_surface = gt_mask.astype(bool) & ~np.isclose(
        np.pad(gt_mask, 1, mode='constant'), gt_mask[..., None, None, None]
    ).any(axis=(0, 1, 2))[1:-1, 1:-1, 1:-1]
    
    if np.sum(pred_surface) == 0 or np.sum(gt_surface) == 0:
        return float('inf')
    
    # Compute distance transforms
    spacing_array = np.array(spacing)
    pred_dt = distance_transform_edt(~pred_surface, sampling=spacing_array)
    gt_dt = distance_transform_edt(~gt_surface, sampling=spacing_array)
    
    # Get distances from surfaces
    pred_distances = pred_dt[gt_surface]
    gt_distances = gt_dt[pred_surface]
    
    if len(pred_distances) == 0 or len(gt_distances) == 0:
        return float('inf')
    
    # Compute 95th percentile
    hd_95 = max(
        np.percentile(pred_distances, 95),
        np.percentile(gt_distances, 95)
    )
    
    return float(hd_95)


def evaluate_dataset(
    predictions: List[str],
    ground_truths: List[str],
    num_classes: Optional[int] = None,
    include_hausdorff: bool = False,
    spacing: Optional[Tuple[float, float, float]] = None
) -> Dict[str, any]:
    """
    Evaluates predictions on a dataset.
    
    Args:
        predictions: List of paths to prediction NIfTI files
        ground_truths: List of paths to ground truth NIfTI files
        num_classes: Number of classes
        include_hausdorff: Whether to compute Hausdorff distance
        spacing: Voxel spacing (if None, read from first image)
        
    Returns:
        Dictionary with per-sample and aggregate metrics
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths must have same length")
    
    results = []
    
    for pred_path, gt_path in zip(predictions, ground_truths):
        logger.info(f"Evaluating {Path(pred_path).name}...")
        
        # Load images
        pred_img = nib.load(pred_path)
        gt_img = nib.load(gt_path)
        
        pred_data = pred_img.get_fdata().astype(np.uint8)
        gt_data = gt_img.get_fdata().astype(np.uint8)
        
        # Get spacing if not provided
        if spacing is None:
            spacing = gt_img.header.get_zooms()[:3]
        
        # Compute metrics
        metrics = compute_metrics(
            prediction=pred_data,
            ground_truth=gt_data,
            num_classes=num_classes,
            include_hausdorff=include_hausdorff,
            spacing=spacing
        )
        
        results.append({
            "prediction_path": str(pred_path),
            "ground_truth_path": str(gt_path),
            "metrics": metrics
        })
    
    # Aggregate metrics
    aggregate = compute_aggregate_metrics(results)
    
    return {
        "per_sample": results,
        "aggregate": aggregate,
        "num_samples": len(results)
    }


def compute_aggregate_metrics(results: List[Dict]) -> Dict[str, float]:
    """
    Computes aggregate metrics across all samples.
    
    Args:
        results: List of per-sample results from evaluate_dataset
        
    Returns:
        Dictionary with aggregate metrics
    """
    if len(results) == 0:
        return {}
    
    # Collect all per-class dice scores
    all_dice_scores = {}
    all_iou_scores = {}
    
    for result in results:
        per_class = result["metrics"].get("per_class", {})
        for class_name, class_metrics in per_class.items():
            if class_name not in all_dice_scores:
                all_dice_scores[class_name] = []
                all_iou_scores[class_name] = []
            
            all_dice_scores[class_name].append(class_metrics["dice"])
            all_iou_scores[class_name].append(class_metrics["iou"])
    
    # Compute means and stds
    aggregate = {}
    
    for class_name in all_dice_scores:
        dice_scores = all_dice_scores[class_name]
        iou_scores = all_iou_scores[class_name]
        
        aggregate[f"{class_name}_dice_mean"] = float(np.mean(dice_scores))
        aggregate[f"{class_name}_dice_std"] = float(np.std(dice_scores))
        aggregate[f"{class_name}_iou_mean"] = float(np.mean(iou_scores))
        aggregate[f"{class_name}_iou_std"] = float(np.std(iou_scores))
    
    # Overall mean dice
    if "mean_dice" in results[0]["metrics"]:
        mean_dices = [r["metrics"]["mean_dice"] for r in results]
        aggregate["overall_mean_dice"] = float(np.mean(mean_dices))
        aggregate["overall_std_dice"] = float(np.std(mean_dices))
    
    return aggregate


def generate_confusion_matrix(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    num_classes: Optional[int] = None
) -> np.ndarray:
    """
    Generates confusion matrix.
    
    Args:
        prediction: Prediction mask
        ground_truth: Ground truth mask
        num_classes: Number of classes
        
    Returns:
        Confusion matrix (num_classes x num_classes)
    """
    if num_classes is None:
        num_classes = max(int(prediction.max()), int(ground_truth.max())) + 1
    
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    pred_flat = prediction.flatten()
    gt_flat = ground_truth.flatten()
    
    for i in range(len(pred_flat)):
        cm[int(gt_flat[i]), int(pred_flat[i])] += 1
    
    return cm


def save_evaluation_report(
    evaluation_results: Dict[str, any],
    output_path: str,
    format: str = "json"
) -> None:
    """
    Saves evaluation report to file.
    
    Args:
        evaluation_results: Results from evaluate_dataset
        output_path: Path to save report
        format: Output format ("json" or "csv")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "json":
        import json
        with open(output_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        logger.info(f"Saved evaluation report to {output_path}")
    
    elif format == "csv":
        import pandas as pd
        
        # Flatten per-sample results
        rows = []
        for result in evaluation_results["per_sample"]:
            row = {
                "prediction": Path(result["prediction_path"]).name,
                "ground_truth": Path(result["ground_truth_path"]).name
            }
            
            # Add per-class metrics
            per_class = result["metrics"].get("per_class", {})
            for class_name, metrics in per_class.items():
                for metric_name, value in metrics.items():
                    row[f"{class_name}_{metric_name}"] = value
            
            # Add overall metrics
            if "mean_dice" in result["metrics"]:
                row["mean_dice"] = result["metrics"]["mean_dice"]
            if "accuracy" in result["metrics"]:
                row["accuracy"] = result["metrics"]["accuracy"]
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved evaluation report to {output_path}")
    
    else:
        raise ValueError(f"Unknown format: {format}")


