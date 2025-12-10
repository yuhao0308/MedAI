"""
Simple Data Ingestion Script
Processes DICOM files from raw_dicom/ and organizes into BIDS format
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.dicom_parser import get_series_metadata, filter_series, extract_participant_metadata
from src.ingestion.dicom_to_nifti import convert_dicom_to_nifti
from src.ingestion.bids_organizer import (
    create_bids_structure,
    create_dataset_description,
    create_participants_tsv,
    organize_nifti_to_bids,
    create_bids_readme
)
import logging
import os
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Ingest DICOM data into BIDS format")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="./data/raw_dicom",
        help="Directory containing raw DICOM files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/bids_dataset",
        help="Output BIDS dataset directory"
    )
    parser.add_argument(
        "--modality",
        type=str,
        default=None,
        help="Filter by modality (e.g., 'CT', 'MR')"
    )
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        help="Maximum number of subjects to process (for testing)"
    )
    
    args = parser.parse_args()
    
    # Step 1: Parse DICOM files
    logger.info(f"Parsing DICOM files from {args.input_dir}...")
    if not os.path.exists(args.input_dir):
        logger.error(f"Input directory not found: {args.input_dir}")
        logger.info("Please download DICOM data first. See QUICKSTART.md")
        return
    
    series_map = get_series_metadata(args.input_dir)
    logger.info(f"Found {len(series_map)} DICOM series")
    
    if len(series_map) == 0:
        logger.error("No DICOM series found. Please check your input directory.")
        return
    
    # Step 2: Filter series
    if args.modality:
        target_series = filter_series(series_map, target_modality=args.modality)
        logger.info(f"Filtered to {len(target_series)} {args.modality} series")
    else:
        # Use all series, but group by patient
        target_series = []
        seen_patients = set()
        for series_uid, metadata in series_map.items():
            patient_id = metadata.get("patient_id", "unknown")
            if patient_id not in seen_patients:
                # Use first series per patient
                if metadata.get("files"):
                    target_series.append(os.path.dirname(metadata["files"][0]))
                    seen_patients.add(patient_id)
        logger.info(f"Selected {len(target_series)} series (one per patient)")
    
    if len(target_series) == 0:
        logger.error("No target series found after filtering")
        return
    
    # Limit for testing
    if args.max_subjects:
        target_series = target_series[:args.max_subjects]
        logger.info(f"Processing first {len(target_series)} series for testing")
    
    # Step 3: Create BIDS structure
    logger.info(f"Creating BIDS structure at {args.output_dir}...")
    create_bids_structure(args.output_dir)
    create_dataset_description(
        args.output_dir,
        dataset_name="Medical Imaging AI Dataset",
        description="Dataset processed from DICOM files"
    )
    create_bids_readme(args.output_dir)
    
    # Step 4: Convert and organize
    temp_nifti_dir = os.path.join(args.output_dir, "..", "temp_nifti")
    os.makedirs(temp_nifti_dir, exist_ok=True)
    
    participants_data = []
    successful = 0
    
    for idx, series_dir in enumerate(target_series):
        participant_id = f"{idx+1:03d}"
        
        try:
            # Get modality from first DICOM file
            import pydicom
            sample_file = None
            for root, _, files in os.walk(series_dir):
                for file in files:
                    if file.endswith('.dcm') or file.endswith('.DCM'):
                        sample_file = os.path.join(root, file)
                        break
                if sample_file:
                    break
            
            if sample_file:
                dcm = pydicom.dcmread(sample_file, stop_before_pixel_data=True)
                modality = str(dcm.get("Modality", "UNKNOWN"))
                series_desc = str(dcm.get("SeriesDescription", "T1w"))
            else:
                modality = "UNKNOWN"
                series_desc = "T1w"
            
            # Determine suffix from series description
            suffix = "T1w"
            if "T2" in series_desc.upper():
                suffix = "T2w"
            elif "FLAIR" in series_desc.upper():
                suffix = "FLAIR"
            elif "CT" in modality:
                suffix = "CT"
            
            logger.info(f"Processing {idx+1}/{len(target_series)}: {series_dir}")
            
            # Convert to NIfTI
            nifti_path = convert_dicom_to_nifti(
                input_dir=series_dir,
                output_dir=temp_nifti_dir,
                output_filename=f"sub-{participant_id}_{suffix}",
                bids_compliant=True
            )
            
            # Organize into BIDS
            organize_nifti_to_bids(
                nifti_file=nifti_path,
                bids_root=args.output_dir,
                participant_id=participant_id,
                modality="anat",
                suffix=suffix
            )
            
            # Collect participant metadata
            participants_data.append({
                "participant_id": f"sub-{participant_id}",
                "modality": modality,
                "series_description": series_desc
            })
            
            successful += 1
            
        except Exception as e:
            logger.error(f"Failed to process {series_dir}: {e}")
            continue
    
    # Step 5: Create participants.tsv
    if participants_data:
        create_participants_tsv(args.output_dir, participants_data)
        logger.info(f"Successfully processed {successful}/{len(target_series)} subjects")
        logger.info(f"BIDS dataset created at: {args.output_dir}")
    else:
        logger.error("No subjects were successfully processed")


if __name__ == "__main__":
    main()


