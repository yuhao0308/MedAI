"""
Reporting Utilities
Quantitative metrics extraction and visualization export
"""

from pathlib import Path
from typing import Dict, List, Optional
import json
import logging

logger = logging.getLogger(__name__)


def extract_segmentation_metrics(
    segmentation_path: str,
    label_values: Optional[List[int]] = None
) -> Dict:
    """
    Extracts quantitative metrics from segmentation.
    
    Args:
        segmentation_path: Path to segmentation NIfTI file
        label_values: List of label values to analyze (if None, analyzes all)
        
    Returns:
        Dictionary with metrics
    """
    import nibabel as nib
    import numpy as np
    
    seg = nib.load(segmentation_path)
    seg_data = seg.get_fdata()
    
    voxel_spacing = seg.header.get_zooms()[:3]
    voxel_volume_mm3 = np.prod(voxel_spacing)
    
    if label_values is None:
        label_values = [int(v) for v in np.unique(seg_data) if v != 0]
    
    metrics = {
        "file": segmentation_path,
        "voxel_spacing_mm": list(voxel_spacing),
        "voxel_volume_mm3": float(voxel_volume_mm3),
        "labels": {}
    }
    
    for label in label_values:
        mask = seg_data == label
        count = np.sum(mask)
        volume_mm3 = count * voxel_volume_mm3
        
        # Calculate centroid
        coords = np.where(mask)
        if len(coords[0]) > 0:
            centroid = [
                float(np.mean(coords[i])) * voxel_spacing[i]
                for i in range(3)
            ]
        else:
            centroid = [0, 0, 0]
        
        metrics["labels"][int(label)] = {
            "voxel_count": int(count),
            "volume_mm3": float(volume_mm3),
            "centroid_mm": centroid
        }
    
    return metrics


def export_metrics_to_json(
    metrics: Dict,
    output_path: str
) -> None:
    """
    Exports metrics to JSON file.
    
    Args:
        metrics: Metrics dictionary
        output_path: Path to save JSON file
    """
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Exported metrics to: {output_path}")


def create_summary_report(
    metrics_list: List[Dict],
    output_path: str
) -> None:
    """
    Creates a summary report from multiple segmentation metrics.
    
    Args:
        metrics_list: List of metrics dictionaries
        output_path: Path to save summary report
    """
    summary = {
        "total_subjects": len(metrics_list),
        "summary_statistics": {}
    }
    
    # Aggregate statistics across all subjects
    all_volumes = {}
    
    for metrics in metrics_list:
        for label, label_metrics in metrics.get("labels", {}).items():
            if label not in all_volumes:
                all_volumes[label] = []
            all_volumes[label].append(label_metrics["volume_mm3"])
    
    # Calculate statistics
    import numpy as np
    
    for label, volumes in all_volumes.items():
        summary["summary_statistics"][f"label_{label}"] = {
            "mean_volume_mm3": float(np.mean(volumes)),
            "std_volume_mm3": float(np.std(volumes)),
            "min_volume_mm3": float(np.min(volumes)),
            "max_volume_mm3": float(np.max(volumes)),
            "count": len(volumes)
        }
    
    # Save summary
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Created summary report: {output_path}")


