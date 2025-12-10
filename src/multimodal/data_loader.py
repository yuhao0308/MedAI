"""
Multimodal Data Integration
Loads imaging, EHR, and text data for multimodal models
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


def load_ehr_data(ehr_file: str) -> pd.DataFrame:
    """
    Loads structured EHR (Electronic Health Record) data.
    
    Supports CSV, TSV, and JSON formats.
    
    Args:
        ehr_file: Path to EHR data file
        
    Returns:
        DataFrame with EHR data
    """
    ehr_path = Path(ehr_file)
    
    if ehr_path.suffix == ".csv":
        ehr_df = pd.read_csv(ehr_file)
    elif ehr_path.suffix == ".tsv":
        ehr_df = pd.read_csv(ehr_file, sep="\t")
    elif ehr_path.suffix == ".json":
        ehr_df = pd.read_json(ehr_file)
    else:
        raise ValueError(f"Unsupported EHR file format: {ehr_path.suffix}")
    
    logger.info(f"Loaded EHR data: {len(ehr_df)} records")
    return ehr_df


def load_clinical_notes(notes_file: str) -> Dict[str, str]:
    """
    Loads clinical notes/text reports.
    
    Args:
        notes_file: Path to JSON file with patient_id -> note_text mapping
        
    Returns:
        Dictionary mapping patient_id to note text
    """
    with open(notes_file, "r") as f:
        notes_dict = json.load(f)
    
    logger.info(f"Loaded clinical notes for {len(notes_dict)} patients")
    return notes_dict


def pair_image_text_data(
    image_paths: List[str],
    ehr_data: Optional[pd.DataFrame] = None,
    clinical_notes: Optional[Dict[str, str]] = None,
    patient_id_key: str = "patient_id"
) -> List[Dict[str, Union[str, Dict]]]:
    """
    Pairs imaging data with corresponding EHR and text data.
    
    Args:
        image_paths: List of paths to image files (BIDS format)
        ehr_data: Optional DataFrame with EHR data
        clinical_notes: Optional dictionary with clinical notes
        patient_id_key: Column name for patient ID in EHR data
        
    Returns:
        List of dictionaries with paired data
    """
    paired_data = []
    
    for image_path in image_paths:
        # Extract participant ID from BIDS path
        # Format: sub-001/anat/sub-001_T1w.nii.gz
        path_parts = Path(image_path).parts
        participant_id = None
        
        for part in path_parts:
            if part.startswith("sub-"):
                participant_id = part.replace("sub-", "")
                break
        
        if participant_id is None:
            logger.warning(f"Could not extract participant ID from {image_path}")
            continue
        
        data_dict = {
            "image": image_path,
            "participant_id": participant_id
        }
        
        # Add EHR data if available
        if ehr_data is not None and patient_id_key in ehr_data.columns:
            patient_ehr = ehr_data[ehr_data[patient_id_key] == participant_id]
            if not patient_ehr.empty:
                data_dict["ehr"] = patient_ehr.iloc[0].to_dict()
        
        # Add clinical notes if available
        if clinical_notes is not None:
            note_key = f"sub-{participant_id}"
            if note_key in clinical_notes:
                data_dict["clinical_note"] = clinical_notes[note_key]
            elif participant_id in clinical_notes:
                data_dict["clinical_note"] = clinical_notes[participant_id]
        
        paired_data.append(data_dict)
    
    logger.info(f"Paired {len(paired_data)} image-text/EHR records")
    return paired_data


