"""
Diagnostic Report Generation
Creates structured JSON and natural language reports
"""

import json
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def generate_quantitative_report(
    segmentation_path: str,
    uncertainty_path: Optional[str] = None,
    metadata: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Generates a structured JSON report with quantitative metrics.
    
    Args:
        segmentation_path: Path to segmentation mask NIfTI file
        uncertainty_path: Optional path to uncertainty map
        metadata: Optional additional metadata
        
    Returns:
        Dictionary with quantitative metrics
    """
    import nibabel as nib
    import numpy as np
    
    # Load segmentation
    seg_img = nib.load(segmentation_path)
    seg_data = seg_img.get_fdata()
    
    # Calculate metrics
    unique_labels = np.unique(seg_data)
    label_counts = {}
    label_volumes = {}
    
    # Get voxel spacing from NIfTI header
    voxel_spacing = seg_img.header.get_zooms()[:3]  # x, y, z spacing
    voxel_volume_mm3 = np.prod(voxel_spacing)
    
    for label in unique_labels:
        if label == 0:  # Background
            continue
        
        count = np.sum(seg_data == label)
        volume_mm3 = count * voxel_volume_mm3
        
        label_counts[int(label)] = int(count)
        label_volumes[int(label)] = float(volume_mm3)
    
    report = {
        "segmentation_file": segmentation_path,
        "total_labels": len([l for l in unique_labels if l != 0]),
        "label_counts": label_counts,
        "label_volumes_mm3": label_volumes,
        "voxel_spacing_mm": list(voxel_spacing),
        "image_shape": list(seg_data.shape)
    }
    
    # Add uncertainty metrics if available
    if uncertainty_path and Path(uncertainty_path).exists():
        unc_img = nib.load(uncertainty_path)
        unc_data = unc_img.get_fdata()
        
        report["uncertainty"] = {
            "mean_uncertainty": float(np.mean(unc_data)),
            "max_uncertainty": float(np.max(unc_data)),
            "high_uncertainty_voxels": int(np.sum(unc_data > np.percentile(unc_data, 95)))
        }
    
    # Add metadata
    if metadata:
        report["metadata"] = metadata
    
    return report


def save_quantitative_report(
    report: Dict[str, Any],
    output_path: str
) -> None:
    """
    Saves quantitative report to JSON file.
    
    Args:
        report: Report dictionary
        output_path: Path to save JSON file
    """
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Saved quantitative report: {output_path}")


def generate_natural_language_summary(
    quantitative_report: Dict[str, Any],
    template: Optional[str] = None
) -> str:
    """
    Generates a natural language summary from quantitative report.
    
    This is a foundation implementation. Full version would use VLM.
    
    Args:
        quantitative_report: Quantitative report dictionary
        template: Optional custom template
        
    Returns:
        Natural language summary string
    """
    if template is None:
        template = """
Segmentation Analysis Report

Found {total_labels} distinct labeled region(s).

Volume Analysis:
{volume_summary}

Image Information:
- Voxel spacing: {spacing} mm
- Image dimensions: {shape}

{uncertainty_summary}
"""
    
    # Extract information
    total_labels = quantitative_report.get("total_labels", 0)
    volumes = quantitative_report.get("label_volumes_mm3", {})
    spacing = quantitative_report.get("voxel_spacing_mm", [])
    shape = quantitative_report.get("image_shape", [])
    
    # Format volume summary
    volume_lines = []
    for label, volume in volumes.items():
        volume_lines.append(f"  Label {label}: {volume:.2f} mm³")
    volume_summary = "\n".join(volume_lines) if volume_lines else "  No labeled regions found"
    
    # Format uncertainty summary
    uncertainty_info = quantitative_report.get("uncertainty", {})
    if uncertainty_info:
        uncertainty_summary = f"""
Uncertainty Analysis:
- Mean uncertainty: {uncertainty_info.get('mean_uncertainty', 0):.4f}
- Max uncertainty: {uncertainty_info.get('max_uncertainty', 0):.4f}
- High uncertainty voxels: {uncertainty_info.get('high_uncertainty_voxels', 0)}
"""
    else:
        uncertainty_summary = ""
    
    # Format summary
    summary = template.format(
        total_labels=total_labels,
        volume_summary=volume_summary,
        spacing=" x ".join([f"{s:.2f}" for s in spacing]),
        shape=" x ".join([str(s) for s in shape]),
        uncertainty_summary=uncertainty_summary
    )
    
    return summary


