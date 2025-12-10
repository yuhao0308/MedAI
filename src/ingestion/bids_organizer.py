"""
BIDS (Brain Imaging Data Structure) Organization
2025 Standard: Organizes NIfTI files into BIDS format for reproducible AI
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import shutil
import logging

logger = logging.getLogger(__name__)


def create_bids_structure(bids_root: str) -> None:
    """
    Creates the standard BIDS directory structure.
    
    Args:
        bids_root: Root directory for the BIDS dataset
    """
    bids_path = Path(bids_root)
    
    # Main directories
    bids_path.mkdir(parents=True, exist_ok=True)
    
    # Derivatives structure
    derivatives_path = bids_path / "derivatives" / "labels"
    derivatives_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Created BIDS structure at {bids_root}")


def create_dataset_description(
    bids_root: str,
    dataset_name: str = "Medical Imaging AI Dataset",
    bids_version: str = "1.9.0",
    authors: Optional[List[str]] = None,
    description: Optional[str] = None
) -> None:
    """
    Creates dataset_description.json file required by BIDS.
    
    Args:
        bids_root: Root directory of BIDS dataset
        dataset_name: Name of the dataset
        bids_version: BIDS version
        authors: List of author names
        description: Dataset description
    """
    dataset_desc = {
        "Name": dataset_name,
        "BIDSVersion": bids_version,
        "DatasetType": "raw",
        "License": "CC0",
    }
    
    if authors:
        dataset_desc["Authors"] = authors
    
    if description:
        dataset_desc["Description"] = description
    
    dataset_desc_path = Path(bids_root) / "dataset_description.json"
    
    with open(dataset_desc_path, "w") as f:
        json.dump(dataset_desc, f, indent=2)
    
    logger.info(f"Created dataset_description.json at {dataset_desc_path}")
    
    # Also create for derivatives
    derivatives_desc = {
        "Name": f"{dataset_name} - Labels",
        "BIDSVersion": bids_version,
        "DatasetType": "derivative",
        "GeneratedBy": [{
            "Name": "MONAI Label",
            "Version": "0.5.0"
        }]
    }
    
    derivatives_path = Path(bids_root) / "derivatives" / "labels" / "dataset_description.json"
    derivatives_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(derivatives_path, "w") as f:
        json.dump(derivatives_desc, f, indent=2)


def create_participants_tsv(
    bids_root: str,
    participants_data: List[Dict[str, str]]
) -> None:
    """
    Creates participants.tsv file (the data dictionary).
    
    Args:
        bids_root: Root directory of BIDS dataset
        participants_data: List of dictionaries with participant metadata
                          Each dict should have at least 'participant_id'
    """
    if not participants_data:
        logger.warning("No participant data provided, creating empty participants.tsv")
        participants_data = [{"participant_id": "sub-001"}]
    
    df = pd.DataFrame(participants_data)
    
    # Ensure participant_id is the first column
    if "participant_id" in df.columns:
        cols = ["participant_id"] + [c for c in df.columns if c != "participant_id"]
        df = df[cols]
    else:
        logger.warning("participant_id not found in data, adding default")
        df.insert(0, "participant_id", [f"sub-{i+1:03d}" for i in range(len(df))])
    
    participants_path = Path(bids_root) / "participants.tsv"
    df.to_csv(participants_path, sep="\t", index=False)
    
    logger.info(f"Created participants.tsv with {len(df)} participants")


def organize_nifti_to_bids(
    nifti_file: str,
    bids_root: str,
    participant_id: str,
    modality: str = "anat",
    suffix: str = "T1w",
    session: Optional[str] = None
) -> str:
    """
    Organizes a NIfTI file into BIDS structure.
    
    BIDS naming: sub-{participant_id}[_ses-{session}]_{modality}_{suffix}.nii.gz
    
    Args:
        nifti_file: Path to NIfTI file
        bids_root: Root directory of BIDS dataset
        participant_id: BIDS participant ID (e.g., "001")
        modality: Modality label (e.g., "anat", "func", "dwi")
        suffix: Suffix label (e.g., "T1w", "T2w", "FLAIR")
        session: Optional session ID
        
    Returns:
        Path to the organized NIfTI file in BIDS structure
    """
    bids_path = Path(bids_root)
    
    # Create participant directory
    participant_dir = bids_path / f"sub-{participant_id}"
    if session:
        participant_dir = participant_dir / f"ses-{session}"
    modality_dir = participant_dir / modality
    modality_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate BIDS-compliant filename
    if session:
        bids_filename = f"sub-{participant_id}_ses-{session}_{suffix}.nii.gz"
    else:
        bids_filename = f"sub-{participant_id}_{suffix}.nii.gz"
    
    bids_filepath = modality_dir / bids_filename
    
    # Copy or move the file
    if os.path.exists(nifti_file):
        shutil.copy2(nifti_file, bids_filepath)
        logger.info(f"Organized {nifti_file} -> {bids_filepath}")
    else:
        raise FileNotFoundError(f"NIfTI file not found: {nifti_file}")
    
    return str(bids_filepath)


def create_bids_readme(bids_root: str, content: Optional[str] = None) -> None:
    """
    Creates a README file for the BIDS dataset.
    
    Args:
        bids_root: Root directory of BIDS dataset
        content: Optional custom README content
    """
    if content is None:
        content = """# Medical Imaging AI Dataset

This dataset is organized according to the Brain Imaging Data Structure (BIDS) standard.

## Dataset Information

This dataset contains medical imaging data for AI model training and validation.

## Data Organization

- `sub-*/anat/`: Anatomical imaging data
- `derivatives/labels/`: Segmentation masks and annotations

## Usage

This dataset is compatible with:
- MONAI Auto3DSeg
- MONAI Label
- Standard BIDS tools

For more information, see: https://bids.neuroimaging.io/
"""
    
    readme_path = Path(bids_root) / "README"
    
    with open(readme_path, "w") as f:
        f.write(content)
    
    logger.info(f"Created README at {readme_path}")


def validate_bids_structure(bids_root: str) -> bool:
    """
    Validates that the BIDS structure is correct.
    
    Note: Requires bids-validator to be installed.
    
    Args:
        bids_root: Root directory of BIDS dataset
        
    Returns:
        True if valid, False otherwise
    """
    try:
        import subprocess
        result = subprocess.run(
            ["bids-validator", bids_root],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            logger.info("BIDS structure validation passed")
            return True
        else:
            logger.warning(f"BIDS validation warnings: {result.stdout}")
            return False
            
    except FileNotFoundError:
        logger.warning("bids-validator not found. Install with: pip install bids-validator")
        return False
    except Exception as e:
        logger.warning(f"BIDS validation error: {e}")
        return False


