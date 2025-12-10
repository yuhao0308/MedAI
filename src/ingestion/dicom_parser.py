"""
DICOM Series Discovery and Parsing
2025 Standard: Uses pydicom for metadata and SimpleITK for volume loading
"""

import os
import pydicom
import pydicom.errors
from typing import Dict, List, Optional
import SimpleITK as sitk
from pathlib import Path


def get_series_metadata(dicom_root_dir: str) -> Dict[str, Dict]:
    """
    Scans a directory and groups all DICOM files by their series.
    
    Uses pydicom for fast metadata-only parsing (stops before pixel data).
    
    Args:
        dicom_root_dir: Root directory containing DICOM files
        
    Returns:
        Dictionary mapping SeriesInstanceUID to metadata dict containing:
        - files: List of file paths in the series
        - patient_id: Patient ID from DICOM
        - study_uid: StudyInstanceUID
        - series_desc: SeriesDescription
    """
    series_map = {}
    dicom_root = Path(dicom_root_dir)
    
    for root, _, files in os.walk(dicom_root):
        for file in files:
            # Skip non-DICOM files
            if not (file.lower().endswith('.dcm') or file.lower().endswith('.dicom')):
                continue
                
            filepath = os.path.join(root, file)
            try:
                # Read metadata only, stops before pixel data for speed
                # Use stop_before_pixels if available (pydicom >= 2.0)
                # Check if parameter exists in function signature
                import inspect
                sig = inspect.signature(pydicom.dcmread)
                if 'stop_before_pixels' in sig.parameters:
                    dcm = pydicom.dcmread(filepath, stop_before_pixels=True)
                else:
                    # Fallback for older pydicom versions
                    dcm = pydicom.dcmread(filepath)
                
                patient_id = str(dcm.get("PatientID", "UNKNOWN"))
                study_uid = str(dcm.get("StudyInstanceUID", "UNKNOWN"))
                series_uid = str(dcm.get("SeriesInstanceUID", "UNKNOWN"))
                series_desc = str(dcm.get("SeriesDescription", "UNKNOWN"))
                
                if series_uid not in series_map:
                    series_map[series_uid] = {
                        "files": [],
                        "patient_id": patient_id,
                        "study_uid": study_uid,
                        "series_desc": series_desc,
                        "modality": str(dcm.get("Modality", "UNKNOWN")),
                        "series_number": str(dcm.get("SeriesNumber", "UNKNOWN")),
                    }
                series_map[series_uid]["files"].append(filepath)
                
            except pydicom.errors.InvalidDicomError:
                continue  # Skip non-DICOM files
            except Exception as e:
                print(f"Warning: Error reading {filepath}: {e}")
                continue
    
    return series_map


def filter_series(
    series_map: Dict[str, Dict],
    target_description: Optional[str] = None,
    target_modality: Optional[str] = None,
    case_sensitive: bool = False
) -> List[str]:
    """
    Filters the discovered series map for target criteria.
    
    Args:
        series_map: Output from get_series_metadata()
        target_description: Substring to match in SeriesDescription
        target_modality: Exact match for Modality (e.g., "MR", "CT")
        case_sensitive: Whether matching should be case-sensitive
        
    Returns:
        List of unique directory paths containing filtered series
    """
    filtered_series = []
    
    for series_uid, metadata in series_map.items():
        match = True
        
        if target_description:
            desc = metadata["series_desc"]
            if not case_sensitive:
                desc = desc.lower()
                target_description = target_description.lower()
            if target_description not in desc:
                match = False
        
        if target_modality and metadata["modality"] != target_modality:
            match = False
        
        if match:
            # Get the directory containing the first file
            if metadata["files"]:
                dicom_directory = os.path.dirname(metadata["files"][0])
                filtered_series.append(dicom_directory)
    
    return list(set(filtered_series))  # Return unique directories


def load_dicom_volume(dicom_directory: str) -> sitk.Image:
    """
    Loads a directory of DICOM files into a 3D SimpleITK Image.
    
    Uses SimpleITK.ImageSeriesReader which correctly handles:
    - Complex slice ordering
    - B-vector orientation
    - Spatial geometry
    
    Args:
        dicom_directory: Directory containing DICOM files for a single series
        
    Returns:
        SimpleITK Image object (3D or 4D volume)
        
    Raises:
        RuntimeError: If no DICOM series found or loading fails
    """
    reader = sitk.ImageSeriesReader()
    
    try:
        # Get DICOM series file names
        dicom_names = reader.GetGDCMSeriesFileNames(dicom_directory)
        
        if not dicom_names:
            raise RuntimeError(f"No DICOM series found in {dicom_directory}")
        
        reader.SetFileNames(dicom_names)
        
        # Execute the reader to load the volume
        image_volume = reader.Execute()
        
        return image_volume
        
    except Exception as e:
        raise RuntimeError(f"Failed to load DICOM volume from {dicom_directory}: {e}")


def extract_participant_metadata(dicom_file: str) -> Dict[str, str]:
    """
    Extracts metadata for participants.tsv from a DICOM file.
    
    Args:
        dicom_file: Path to a DICOM file
        
    Returns:
        Dictionary with participant metadata fields
    """
    try:
        # Use stop_before_pixels if available (pydicom >= 2.0)
        import inspect
        sig = inspect.signature(pydicom.dcmread)
        if 'stop_before_pixels' in sig.parameters:
            dcm = pydicom.dcmread(dicom_file, stop_before_pixels=True)
        else:
            # Fallback for older pydicom versions
            dcm = pydicom.dcmread(dicom_file)
        
        metadata = {
            "patient_id": str(dcm.get("PatientID", "unknown")),
            "age": str(dcm.get("PatientAge", "unknown")),
            "sex": str(dcm.get("PatientSex", "unknown")),
            "modality": str(dcm.get("Modality", "unknown")),
            "study_date": str(dcm.get("StudyDate", "unknown")),
        }
        
        return metadata
        
    except Exception as e:
        print(f"Warning: Could not extract metadata from {dicom_file}: {e}")
        return {
            "patient_id": "unknown",
            "age": "unknown",
            "sex": "unknown",
            "modality": "unknown",
            "study_date": "unknown",
        }

