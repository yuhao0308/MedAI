"""
Nodule Detection Report Generator
Creates structured reports for lung nodule detection results
Supports JSON, CSV, and summary formats
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import io
import logging

logger = logging.getLogger(__name__)


def generate_nodule_report(
    nodule_candidates: List[Any],
    nodule_stats: Dict[str, Any],
    image_info: Dict[str, Any],
    model_info: Dict[str, Any],
    output_format: str = "json"
) -> str:
    """
    Generate a structured nodule detection report.
    
    Args:
        nodule_candidates: List of NoduleCandidate objects
        nodule_stats: Aggregate statistics dictionary
        image_info: Information about the input image
        model_info: Information about the model used
        output_format: "json", "csv", or "summary"
        
    Returns:
        Report string in the specified format
    """
    if output_format == "json":
        return generate_json_report(nodule_candidates, nodule_stats, image_info, model_info)
    elif output_format == "csv":
        return generate_csv_report(nodule_candidates, image_info, model_info)
    elif output_format == "summary":
        return generate_summary_report(nodule_candidates, nodule_stats, image_info, model_info)
    else:
        raise ValueError(f"Unknown output format: {output_format}")


def generate_json_report(
    nodule_candidates: List[Any],
    nodule_stats: Dict[str, Any],
    image_info: Dict[str, Any],
    model_info: Dict[str, Any]
) -> str:
    """Generate a comprehensive JSON report"""
    
    report = {
        "report_metadata": {
            "generated_at": datetime.now().isoformat(),
            "report_type": "lung_nodule_detection",
            "version": "1.0"
        },
        "study_info": {
            "image_path": image_info.get("image_path", ""),
            "image_shape": list(image_info.get("original_shape", [])),
            "voxel_spacing_mm": list(image_info.get("original_spacing", [])),
            "subject_id": image_info.get("subject_id", "")
        },
        "model_info": {
            "model_name": model_info.get("name", ""),
            "model_type": model_info.get("type", ""),
            "inference_timestamp": model_info.get("timestamp", "")
        },
        "summary": {
            "total_nodules_detected": nodule_stats.get("total_nodules", 0),
            "suspicious_nodules_count": nodule_stats.get("suspicious_count", 0),
            "total_nodule_volume_mm3": nodule_stats.get("total_volume_mm3", 0),
            "max_nodule_diameter_mm": nodule_stats.get("max_diameter_mm", 0),
            "mean_nodule_diameter_mm": nodule_stats.get("mean_diameter_mm", 0),
            "mean_confidence": nodule_stats.get("mean_confidence", 0),
            "lung_rads_distribution": nodule_stats.get("category_counts", {})
        },
        "findings": []
    }
    
    # Add individual nodule findings
    for candidate in nodule_candidates:
        finding = {
            "nodule_id": candidate.id,
            "location": {
                "centroid_voxel": list(candidate.centroid_voxel),
                "centroid_mm": list(candidate.centroid_mm),
                "bounding_box": [list(bb) for bb in candidate.bounding_box]
            },
            "measurements": {
                "volume_mm3": round(candidate.volume_mm3, 2),
                "equivalent_diameter_mm": round(candidate.equivalent_diameter_mm, 2),
                "max_diameter_mm": round(candidate.max_diameter_mm, 2),
                "sphericity": round(candidate.sphericity, 3)
            },
            "classification": {
                "lung_rads_category": candidate.lung_rads_category,
                "confidence_mean": round(candidate.mean_confidence, 3),
                "confidence_min": round(candidate.min_confidence, 3),
                "confidence_max": round(candidate.max_confidence, 3)
            },
            "recommendation": get_lung_rads_recommendation(candidate.lung_rads_category)
        }
        report["findings"].append(finding)
    
    # Add overall recommendation
    report["overall_recommendation"] = get_overall_recommendation(nodule_candidates, nodule_stats)
    
    return json.dumps(report, indent=2)


def generate_csv_report(
    nodule_candidates: List[Any],
    image_info: Dict[str, Any],
    model_info: Dict[str, Any]
) -> str:
    """Generate a CSV report for easy spreadsheet analysis"""
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Header
    headers = [
        "Nodule_ID", "Centroid_X_mm", "Centroid_Y_mm", "Centroid_Z_mm",
        "Volume_mm3", "Equiv_Diameter_mm", "Max_Diameter_mm", "Sphericity",
        "Lung_RADS", "Confidence_Mean", "Confidence_Min", "Confidence_Max",
        "Image_Path", "Model_Name", "Timestamp"
    ]
    writer.writerow(headers)
    
    # Data rows
    for candidate in nodule_candidates:
        row = [
            candidate.id,
            round(candidate.centroid_mm[0], 2),
            round(candidate.centroid_mm[1], 2),
            round(candidate.centroid_mm[2], 2),
            round(candidate.volume_mm3, 2),
            round(candidate.equivalent_diameter_mm, 2),
            round(candidate.max_diameter_mm, 2),
            round(candidate.sphericity, 3),
            candidate.lung_rads_category,
            round(candidate.mean_confidence, 3),
            round(candidate.min_confidence, 3),
            round(candidate.max_confidence, 3),
            image_info.get("image_path", ""),
            model_info.get("name", ""),
            model_info.get("timestamp", "")
        ]
        writer.writerow(row)
    
    return output.getvalue()


def generate_summary_report(
    nodule_candidates: List[Any],
    nodule_stats: Dict[str, Any],
    image_info: Dict[str, Any],
    model_info: Dict[str, Any]
) -> str:
    """Generate a human-readable summary report"""
    
    lines = []
    lines.append("=" * 60)
    lines.append("LUNG NODULE DETECTION REPORT")
    lines.append("=" * 60)
    lines.append("")
    
    # Study Information
    lines.append("STUDY INFORMATION")
    lines.append("-" * 40)
    lines.append(f"Image: {image_info.get('image_path', 'N/A')}")
    lines.append(f"Subject: {image_info.get('subject_id', 'N/A')}")
    lines.append(f"Image Shape: {image_info.get('original_shape', 'N/A')}")
    lines.append(f"Voxel Spacing: {image_info.get('original_spacing', 'N/A')} mm")
    lines.append("")
    
    # Model Information
    lines.append("MODEL INFORMATION")
    lines.append("-" * 40)
    lines.append(f"Model: {model_info.get('name', 'N/A')}")
    lines.append(f"Type: {model_info.get('type', 'N/A')}")
    lines.append(f"Analysis Date: {model_info.get('timestamp', datetime.now().isoformat())}")
    lines.append("")
    
    # Summary
    lines.append("SUMMARY")
    lines.append("-" * 40)
    total = nodule_stats.get("total_nodules", 0)
    suspicious = nodule_stats.get("suspicious_count", 0)
    lines.append(f"Total Nodules Detected: {total}")
    lines.append(f"Suspicious Nodules (4A+): {suspicious}")
    
    if total > 0:
        lines.append(f"Total Nodule Volume: {nodule_stats.get('total_volume_mm3', 0):.1f} mm³")
        lines.append(f"Largest Nodule: {nodule_stats.get('max_diameter_mm', 0):.1f} mm diameter")
        lines.append(f"Mean Diameter: {nodule_stats.get('mean_diameter_mm', 0):.1f} mm")
        lines.append(f"Mean Confidence: {nodule_stats.get('mean_confidence', 0):.2f}")
    lines.append("")
    
    # Lung-RADS Distribution
    category_counts = nodule_stats.get("category_counts", {})
    if category_counts:
        lines.append("LUNG-RADS DISTRIBUTION")
        lines.append("-" * 40)
        for cat in ["1", "2", "3", "4A", "4B", "4X"]:
            count = category_counts.get(cat, 0)
            if count > 0:
                lines.append(f"  Category {cat}: {count} nodule(s)")
        lines.append("")
    
    # Individual Findings
    if nodule_candidates:
        lines.append("INDIVIDUAL FINDINGS")
        lines.append("-" * 40)
        
        for candidate in nodule_candidates:
            lines.append(f"\nNodule #{candidate.id}")
            lines.append(f"  Location: ({candidate.centroid_mm[0]:.1f}, {candidate.centroid_mm[1]:.1f}, {candidate.centroid_mm[2]:.1f}) mm")
            lines.append(f"  Diameter: {candidate.equivalent_diameter_mm:.1f} mm")
            lines.append(f"  Volume: {candidate.volume_mm3:.1f} mm³")
            lines.append(f"  Lung-RADS: Category {candidate.lung_rads_category}")
            lines.append(f"  Confidence: {candidate.mean_confidence:.2f}")
            lines.append(f"  Recommendation: {get_lung_rads_recommendation(candidate.lung_rads_category)}")
    
    lines.append("")
    
    # Overall Recommendation
    lines.append("OVERALL RECOMMENDATION")
    lines.append("-" * 40)
    lines.append(get_overall_recommendation(nodule_candidates, nodule_stats))
    lines.append("")
    
    # Disclaimer
    lines.append("=" * 60)
    lines.append("DISCLAIMER: This is an AI-assisted analysis. All findings")
    lines.append("should be reviewed and verified by a qualified radiologist.")
    lines.append("=" * 60)
    
    return "\n".join(lines)


def get_lung_rads_recommendation(category: str) -> str:
    """Get management recommendation based on Lung-RADS category"""
    
    recommendations = {
        "1": "Continue annual screening with low-dose CT",
        "2": "Continue annual screening with low-dose CT",
        "3": "6-month follow-up low-dose CT recommended",
        "4A": "3-month follow-up CT or consider PET/CT; tissue sampling may be considered",
        "4B": "Consider tissue sampling or surgical consultation; PET/CT may be performed",
        "4X": "Additional workup as indicated by concerning features"
    }
    
    return recommendations.get(category, "Clinical correlation recommended")


def get_overall_recommendation(
    nodule_candidates: List[Any],
    nodule_stats: Dict[str, Any]
) -> str:
    """Generate overall recommendation based on findings"""
    
    if not nodule_candidates:
        return "No significant nodules detected. Continue routine screening as clinically indicated."
    
    suspicious_count = nodule_stats.get("suspicious_count", 0)
    total_count = nodule_stats.get("total_nodules", 0)
    max_diameter = nodule_stats.get("max_diameter_mm", 0)
    
    if suspicious_count > 0:
        return (
            f"ATTENTION: {suspicious_count} suspicious nodule(s) detected (Lung-RADS 4A or higher). "
            f"Largest nodule: {max_diameter:.1f} mm. "
            "Recommend urgent radiologist review and appropriate follow-up as per Lung-RADS guidelines."
        )
    elif total_count > 0:
        # Find highest category
        category_counts = nodule_stats.get("category_counts", {})
        if category_counts.get("3", 0) > 0:
            return (
                f"{total_count} nodule(s) detected, including Lung-RADS Category 3 finding(s). "
                "Recommend 6-month follow-up CT and radiologist review."
            )
        else:
            return (
                f"{total_count} nodule(s) detected, all Lung-RADS Category 2 or below. "
                "Continue annual screening. Radiologist verification recommended."
            )
    
    return "Findings require clinical correlation and radiologist review."


def save_report(
    report_content: str,
    output_path: str,
    output_format: str = "json"
) -> str:
    """
    Save report to file.
    
    Args:
        report_content: Report string content
        output_path: Output file path
        output_format: Format for extension determination
        
    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure correct extension
    ext_map = {"json": ".json", "csv": ".csv", "summary": ".txt"}
    expected_ext = ext_map.get(output_format, ".txt")
    
    if not str(output_path).endswith(expected_ext):
        output_path = output_path.with_suffix(expected_ext)
    
    with open(output_path, 'w') as f:
        f.write(report_content)
    
    logger.info(f"Report saved to {output_path}")
    return str(output_path)


