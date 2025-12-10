"""
Specialized ingestion script for the thorax CT dataset structure
Handles: New_thorax_ct_dicom/Thrx-CT###/dicom/Thrx-CT###/image/ and mask/ folders
"""

import sys
from pathlib import Path
import os
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.dicom_parser import get_series_metadata
from src.ingestion.dicom_to_nifti import convert_dicom_to_nifti
from src.ingestion.bids_organizer import (
    create_bids_structure,
    create_dataset_description,
    create_participants_tsv,
    organize_nifti_to_bids,
    create_bids_readme
)
import argparse
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_thorax_dataset(input_dir: str, output_dir: str, max_subjects: int = None):
    """
    Processes the thorax CT dataset with its specific structure.
    
    Structure:
    New_thorax_ct_dicom/
      Thrx-CT001/
        dicom/
          Thrx-CT001/
            image/  (CT scan DICOM files)
            mask/   (Segmentation masks for different organs)
    
    Args:
        input_dir: Root directory containing Thrx-CT### folders
        output_dir: Output BIDS dataset directory
        max_subjects: Maximum number of subjects to process (for testing)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Find all patient folders
    patient_folders = sorted([d for d in input_path.iterdir() 
                             if d.is_dir() and d.name.startswith("Thrx-CT")])
    
    if not patient_folders:
        logger.error(f"No patient folders found in {input_dir}")
        return
    
    logger.info(f"Found {len(patient_folders)} patient folders")
    
    if max_subjects:
        patient_folders = patient_folders[:max_subjects]
        logger.info(f"Processing first {len(patient_folders)} patients for testing")
    
    # Create BIDS structure
    logger.info(f"Creating BIDS structure at {output_dir}...")
    create_bids_structure(str(output_path))
    create_dataset_description(
        str(output_path),
        dataset_name="Thorax CT Dataset",
        description="Thorax CT scans with multi-organ segmentation masks"
    )
    create_bids_readme(str(output_path))
    
    # Create temp directory for NIfTI conversion
    temp_nifti_dir = output_path.parent / "temp_nifti"
    temp_nifti_dir.mkdir(exist_ok=True)
    
    participants_data = []
    successful = 0
    
    for idx, patient_folder in enumerate(patient_folders):
        patient_id = f"{idx+1:03d}"
        patient_name = patient_folder.name  # e.g., "Thrx-CT001"
        
        logger.info(f"Processing {idx+1}/{len(patient_folders)}: {patient_name}")
        
        try:
            # Find image directory
            image_dir = patient_folder / "dicom" / patient_name / "image"
            if not image_dir.exists():
                logger.warning(f"Image directory not found: {image_dir}")
                continue
            
            # Check if there are DICOM files
            dicom_files = list(image_dir.glob("*.dcm")) + list(image_dir.glob("*.DCM"))
            if not dicom_files:
                logger.warning(f"No DICOM files found in {image_dir}")
                continue
            
            logger.info(f"  Found {len(dicom_files)} DICOM slices")
            
            # Convert image to NIfTI
            logger.info(f"  Converting image to NIfTI...")
            nifti_image = convert_dicom_to_nifti(
                input_dir=str(image_dir),
                output_dir=str(temp_nifti_dir),
                output_filename=f"sub-{patient_id}_CT",
                bids_compliant=True
            )
            
            # Organize image into BIDS
            bids_image_path = organize_nifti_to_bids(
                nifti_file=nifti_image,
                bids_root=str(output_path),
                participant_id=patient_id,
                modality="anat",
                suffix="CT"
            )
            
            # Process segmentation masks
            mask_dir = patient_folder / "dicom" / patient_name / "mask"
            if mask_dir.exists():
                logger.info(f"  Processing segmentation masks...")
                
                # Find all mask subdirectories (LLg, Med, PC, etc.)
                mask_subdirs = [d for d in mask_dir.iterdir() if d.is_dir()]
                
                for mask_subdir in mask_subdirs:
                    mask_name = mask_subdir.name  # e.g., "LLg", "PC", "T-AS"
                    
                    # Check if there are DICOM files
                    mask_dicom_files = list(mask_subdir.glob("*.dcm")) + list(mask_subdir.glob("*.DCM"))
                    if not mask_dicom_files:
                        continue
                    
                    logger.info(f"    Converting mask: {mask_name} ({len(mask_dicom_files)} slices)")
                    
                    # Convert mask to NIfTI
                    nifti_mask = convert_dicom_to_nifti(
                        input_dir=str(mask_subdir),
                        output_dir=str(temp_nifti_dir),
                        output_filename=f"sub-{patient_id}_CT_seg-{mask_name}_mask",
                        bids_compliant=True
                    )
                    
                    # Organize mask into BIDS derivatives
                    bids_mask_dir = output_path / "derivatives" / "labels" / f"sub-{patient_id}" / "anat"
                    bids_mask_dir.mkdir(parents=True, exist_ok=True)
                    
                    bids_mask_filename = f"sub-{patient_id}_CT_seg-{mask_name}_mask.nii.gz"
                    bids_mask_path = bids_mask_dir / bids_mask_filename
                    
                    shutil.copy2(nifti_mask, bids_mask_path)
                    logger.info(f"    Saved mask: {bids_mask_path}")
            
            # Collect participant metadata
            participants_data.append({
                "participant_id": f"sub-{patient_id}",
                "original_id": patient_name,
                "modality": "CT",
                "anatomy": "thorax"
            })
            
            successful += 1
            logger.info(f"  ✓ Successfully processed {patient_name}")
            
        except Exception as e:
            logger.error(f"  ✗ Failed to process {patient_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create participants.tsv
    if participants_data:
        create_participants_tsv(str(output_path), participants_data)
        logger.info(f"\n{'='*60}")
        logger.info(f"SUCCESS: Processed {successful}/{len(patient_folders)} subjects")
        logger.info(f"BIDS dataset created at: {output_path}")
        logger.info(f"{'='*60}\n")
        
        # Clean up temp directory
        if temp_nifti_dir.exists():
            logger.info(f"Cleaning up temporary files...")
            shutil.rmtree(temp_nifti_dir)
    else:
        logger.error("No subjects were successfully processed")


def main():
    parser = argparse.ArgumentParser(
        description="Ingest thorax CT DICOM data into BIDS format"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="./New_thorax_ct_dicom",
        help="Directory containing Thrx-CT### folders"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/bids_dataset",
        help="Output BIDS dataset directory"
    )
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        help="Maximum number of subjects to process (for testing)"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        logger.error(f"Input directory not found: {args.input_dir}")
        return
    
    process_thorax_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_subjects=args.max_subjects
    )


if __name__ == "__main__":
    main()


