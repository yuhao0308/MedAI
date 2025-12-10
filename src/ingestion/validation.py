"""
BIDS Dataset Validation and Quality Checks
Validates structure, files, and data consistency
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import logging
import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)


def validate_bids_structure(bids_root: str) -> Dict[str, any]:
    """
    Validates BIDS directory structure.
    
    Args:
        bids_root: Root directory of BIDS dataset
        
    Returns:
        Dictionary with validation results
    """
    bids_path = Path(bids_root)
    issues = []
    warnings = []
    
    # Check if root exists
    if not bids_path.exists():
        return {
            "valid": False,
            "errors": [f"BIDS root directory does not exist: {bids_root}"],
            "warnings": [],
            "summary": {}
        }
    
    # Check for dataset_description.json
    dataset_desc = bids_path / "dataset_description.json"
    if not dataset_desc.exists():
        warnings.append("dataset_description.json not found (recommended but not required)")
    else:
        try:
            with open(dataset_desc, 'r') as f:
                desc = json.load(f)
                if "BIDSVersion" not in desc:
                    warnings.append("dataset_description.json missing BIDSVersion")
        except json.JSONDecodeError:
            issues.append("dataset_description.json is not valid JSON")
    
    # Check for participants.tsv
    participants_tsv = bids_path / "participants.tsv"
    if not participants_tsv.exists():
        warnings.append("participants.tsv not found (recommended but not required)")
    
    # Find subject directories
    subject_dirs = [d for d in bids_path.iterdir() if d.is_dir() and d.name.startswith("sub-")]
    
    if len(subject_dirs) == 0:
        issues.append("No subject directories (sub-*) found in BIDS root")
        return {
            "valid": False,
            "errors": issues,
            "warnings": warnings,
            "summary": {"num_subjects": 0}
        }
    
    # Validate each subject
    valid_subjects = 0
    subjects_with_images = 0
    subjects_with_labels = 0
    
    for sub_dir in subject_dirs:
        sub_id = sub_dir.name
        
        # Check for anat directory
        anat_dir = sub_dir / "anat"
        if not anat_dir.exists():
            warnings.append(f"{sub_id}: Missing 'anat' directory")
            continue
        
        # Check for image files
        image_files = list(anat_dir.glob("*.nii.gz")) + list(anat_dir.glob("*.nii"))
        if len(image_files) == 0:
            warnings.append(f"{sub_id}: No image files found in anat/")
            continue
        
        subjects_with_images += 1
        
        # Check for labels in derivatives
        label_dir = bids_path / "derivatives" / "labels" / sub_id / "anat"
        if label_dir.exists():
            label_files = list(label_dir.glob("*_mask.nii.gz"))
            if len(label_files) > 0:
                subjects_with_labels += 1
        
        valid_subjects += 1
    
    summary = {
        "num_subjects": len(subject_dirs),
        "valid_subjects": valid_subjects,
        "subjects_with_images": subjects_with_images,
        "subjects_with_labels": subjects_with_labels
    }
    
    return {
        "valid": len(issues) == 0,
        "errors": issues,
        "warnings": warnings,
        "summary": summary
    }


def validate_nifti_files(bids_root: str) -> Dict[str, any]:
    """
    Validates that all NIfTI files are readable and have valid headers.
    
    Args:
        bids_root: Root directory of BIDS dataset
        
    Returns:
        Dictionary with validation results
    """
    bids_path = Path(bids_root)
    issues = []
    warnings = []
    file_stats = []
    
    # Check all image files
    subject_dirs = [d for d in bids_path.iterdir() if d.is_dir() and d.name.startswith("sub-")]
    
    for sub_dir in subject_dirs:
        anat_dir = sub_dir / "anat"
        if not anat_dir.exists():
            continue
        
        image_files = list(anat_dir.glob("*.nii.gz")) + list(anat_dir.glob("*.nii"))
        
        for img_file in image_files:
            try:
                img = nib.load(str(img_file))
                data = img.get_fdata()
                
                file_stats.append({
                    "file": str(img_file.relative_to(bids_path)),
                    "shape": data.shape,
                    "dtype": str(data.dtype),
                    "spacing": img.header.get_zooms()[:3],
                    "valid": True
                })
                
                # Check for unusual values
                if np.any(np.isnan(data)):
                    warnings.append(f"{img_file.name}: Contains NaN values")
                if np.any(np.isinf(data)):
                    warnings.append(f"{img_file.name}: Contains Inf values")
                
            except Exception as e:
                issues.append(f"{img_file.name}: Failed to load - {str(e)}")
                file_stats.append({
                    "file": str(img_file.relative_to(bids_path)),
                    "valid": False,
                    "error": str(e)
                })
    
    # Check label files
    label_dir = bids_path / "derivatives" / "labels"
    if label_dir.exists():
        for sub_dir in label_dir.iterdir():
            if not sub_dir.is_dir():
                continue
            
            anat_label_dir = sub_dir / "anat"
            if not anat_label_dir.exists():
                continue
            
            label_files = list(anat_label_dir.glob("*_mask.nii.gz"))
            
            for label_file in label_files:
                try:
                    label_img = nib.load(str(label_file))
                    label_data = label_img.get_fdata()
                    
                    # Check that labels are integers
                    if not np.all(np.equal(np.mod(label_data, 1), 0)):
                        warnings.append(f"{label_file.name}: Contains non-integer values")
                    
                    # Check label range
                    unique_labels = np.unique(label_data)
                    if len(unique_labels) > 100:
                        warnings.append(f"{label_file.name}: Has {len(unique_labels)} unique labels (might be continuous instead of discrete)")
                    
                except Exception as e:
                    issues.append(f"{label_file.name}: Failed to load - {str(e)}")
    
    return {
        "valid": len(issues) == 0,
        "errors": issues,
        "warnings": warnings,
        "file_stats": file_stats
    }


def check_data_consistency(bids_root: str) -> Dict[str, any]:
    """
    Checks consistency between images and labels (size, spacing, etc.).
    
    For multi-organ datasets, organ masks may have different z-dimensions
    (only covering slices where the organ exists). This is expected and
    should be a warning, not an error.
    
    Args:
        bids_root: Root directory of BIDS dataset
        
    Returns:
        Dictionary with consistency check results
    """
    bids_path = Path(bids_root)
    issues = []
    warnings = []
    
    subject_dirs = [d for d in bids_path.iterdir() if d.is_dir() and d.name.startswith("sub-")]
    
    for sub_dir in subject_dirs:
        sub_id = sub_dir.name
        anat_dir = sub_dir / "anat"
        
        if not anat_dir.exists():
            continue
        
        # Get image files
        image_files = list(anat_dir.glob("*.nii.gz")) + list(anat_dir.glob("*.nii"))
        image_files = [f for f in image_files if "seg" not in f.name and "label" not in f.name]
        
        if len(image_files) == 0:
            continue
        
        # Get corresponding labels
        label_dir = bids_path / "derivatives" / "labels" / sub_id / "anat"
        if not label_dir.exists():
            continue
        
        label_files = list(label_dir.glob("*_mask.nii.gz"))
        
        if len(label_files) == 0:
            warnings.append(f"{sub_id}: Has images but no labels")
            continue
        
        # Load first image for comparison
        try:
            img_file = image_files[0]
            img = nib.load(str(img_file))
            img_data = img.get_fdata()
            img_shape = img_data.shape
            img_spacing = img.header.get_zooms()[:3]
            
            # Check each label against the image
            for label_file in label_files:
                try:
                    label_img = nib.load(str(label_file))
                    label_data = label_img.get_fdata()
                    label_shape = label_data.shape
                    label_spacing = label_img.header.get_zooms()[:3]
                    
                    # Check x,y dimensions match (should always match)
                    if img_shape[0] != label_shape[0] or img_shape[1] != label_shape[1]:
                        issues.append(
                            f"{sub_id}: XY dimension mismatch - Image {img_shape[:2]} vs Label {label_shape[:2]} "
                            f"({label_file.name})"
                        )
                    
                    # Check z-dimension (z can differ for organ-specific masks)
                    if img_shape[2] != label_shape[2]:
                        # This is expected for organ-specific masks - make it a warning
                        warnings.append(
                            f"{sub_id}: Z-dimension differs - Image {img_shape[2]} vs Label {label_shape[2]} slices "
                            f"({label_file.name}) - This is normal for organ-specific masks"
                        )
                    
                    # Check spacing match (with tolerance)
                    spacing_diff = np.abs(np.array(img_spacing) - np.array(label_spacing))
                    if np.any(spacing_diff > 0.01):  # 0.01mm tolerance
                        warnings.append(
                            f"{sub_id}: Spacing mismatch - Image {img_spacing} vs Label {label_spacing} "
                            f"({label_file.name})"
                        )
                    
                    # Check that label z-dimension is not larger than image (would be an error)
                    if label_shape[2] > img_shape[2]:
                        issues.append(
                            f"{sub_id}: Label has more slices than image - Image {img_shape[2]} vs Label {label_shape[2]} "
                            f"({label_file.name})"
                        )
                    
                except Exception as e:
                    issues.append(f"{sub_id}: Failed to load label {label_file.name} - {str(e)}")
                    
        except Exception as e:
            issues.append(f"{sub_id}: Failed to load image {img_file.name} - {str(e)}")
    
    return {
        "valid": len(issues) == 0,
        "errors": issues,
        "warnings": warnings
    }


def generate_data_summary(bids_root: str) -> Dict[str, any]:
    """
    Generates summary statistics about the dataset.
    
    Args:
        bids_root: Root directory of BIDS dataset
        
    Returns:
        Dictionary with summary statistics
    """
    bids_path = Path(bids_root)
    summary = {
        "num_subjects": 0,
        "num_images": 0,
        "num_labels": 0,
        "image_shapes": [],
        "spacings": [],
        "intensity_ranges": [],
        "label_counts": []
    }
    
    subject_dirs = [d for d in bids_path.iterdir() if d.is_dir() and d.name.startswith("sub-")]
    summary["num_subjects"] = len(subject_dirs)
    
    for sub_dir in subject_dirs:
        anat_dir = sub_dir / "anat"
        if not anat_dir.exists():
            continue
        
        # Count images
        image_files = list(anat_dir.glob("*.nii.gz")) + list(anat_dir.glob("*.nii"))
        image_files = [f for f in image_files if "seg" not in f.name and "label" not in f.name]
        summary["num_images"] += len(image_files)
        
        # Analyze first image
        if len(image_files) > 0:
            try:
                img = nib.load(str(image_files[0]))
                data = img.get_fdata()
                summary["image_shapes"].append(data.shape)
                summary["spacings"].append(img.header.get_zooms()[:3])
                summary["intensity_ranges"].append((float(data.min()), float(data.max())))
            except:
                pass
        
        # Count labels
        label_dir = bids_path / "derivatives" / "labels" / sub_dir.name / "anat"
        if label_dir.exists():
            label_files = list(label_dir.glob("*_mask.nii.gz"))
            summary["num_labels"] += len(label_files)
            
            # Analyze labels
            for label_file in label_files:
                try:
                    label_img = nib.load(str(label_file))
                    label_data = label_img.get_fdata()
                    unique_labels = np.unique(label_data)
                    summary["label_counts"].append(len(unique_labels))
                except:
                    pass
    
    # Compute statistics
    if summary["image_shapes"]:
        summary["common_shape"] = max(set(summary["image_shapes"]), key=summary["image_shapes"].count)
    
    if summary["spacings"]:
        summary["common_spacing"] = max(set([tuple(s) for s in summary["spacings"]]), 
                                       key=[tuple(s) for s in summary["spacings"]].count)
    
    if summary["intensity_ranges"]:
        all_mins = [r[0] for r in summary["intensity_ranges"]]
        all_maxs = [r[1] for r in summary["intensity_ranges"]]
        summary["overall_intensity_range"] = (min(all_mins), max(all_maxs))
    
    return summary


def run_all_validations(bids_root: str) -> Dict[str, any]:
    """
    Runs all validation checks and returns comprehensive report.
    
    Args:
        bids_root: Root directory of BIDS dataset
        
    Returns:
        Complete validation report
    """
    logger.info("Running BIDS structure validation...")
    structure_result = validate_bids_structure(bids_root)
    
    logger.info("Running NIfTI file validation...")
    nifti_result = validate_nifti_files(bids_root)
    
    logger.info("Running data consistency checks...")
    consistency_result = check_data_consistency(bids_root)
    
    logger.info("Generating data summary...")
    summary = generate_data_summary(bids_root)
    
    # Combine results
    all_errors = structure_result["errors"] + nifti_result["errors"] + consistency_result["errors"]
    all_warnings = structure_result["warnings"] + nifti_result["warnings"] + consistency_result["warnings"]
    
    return {
        "bids_root": bids_root,
        "valid": len(all_errors) == 0,
        "structure_validation": structure_result,
        "nifti_validation": nifti_result,
        "consistency_validation": consistency_result,
        "summary": summary,
        "all_errors": all_errors,
        "all_warnings": all_warnings,
        "num_errors": len(all_errors),
        "num_warnings": len(all_warnings)
    }

