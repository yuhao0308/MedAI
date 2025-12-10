"""
LUNA16 Dataset Preprocessor
Converts LUNA16 .mhd/.raw files to NIfTI format and creates binary segmentation masks.

Processes first 50 scans from subset0 and organizes into BIDS-compatible structure.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk
import nibabel as nib
from tqdm import tqdm
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def read_mhd(mhd_path):
    """Read .mhd/.raw file using SimpleITK."""
    try:
        image = sitk.ReadImage(str(mhd_path))
        return image
    except Exception as e:
        logger.error(f"Failed to read {mhd_path}: {e}")
        return None


def sitk_to_nifti(sitk_image, output_path):
    """Convert SimpleITK image to NIfTI format using nibabel."""
    # Get numpy array (Z, Y, X)
    array = sitk.GetArrayFromImage(sitk_image)
    
    # Get spacing and origin
    spacing = sitk_image.GetSpacing()  # (X, Y, Z)
    origin = sitk_image.GetOrigin()    # (X, Y, Z)
    direction = sitk_image.GetDirection()
    
    # Create affine matrix
    # Convert spacing from (X,Y,Z) to (Z,Y,X) for nibabel
    affine = np.eye(4)
    affine[0, 0] = spacing[0]
    affine[1, 1] = spacing[1]
    affine[2, 2] = spacing[2]
    affine[0, 3] = origin[0]
    affine[1, 3] = origin[1]
    affine[2, 3] = origin[2]
    
    # Create NIfTI image
    nifti_img = nib.Nifti1Image(array, affine)
    
    # Save
    nib.save(nifti_img, str(output_path))
    logger.debug(f"Saved NIfTI: {output_path}")
    
    return array.shape, spacing


def resample_image(sitk_image, new_spacing=(1.5, 1.5, 1.5), interpolator=sitk.sitkLinear):
    """Resample image to new spacing."""
    original_spacing = sitk_image.GetSpacing()
    original_size = sitk_image.GetSize()
    
    # Calculate new size
    new_size = [
        int(round(osz * osp / nsp))
        for osz, osp, nsp in zip(original_size, original_spacing, new_spacing)
    ]
    
    # Resample
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(sitk_image.GetDirection())
    resampler.SetOutputOrigin(sitk_image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(sitk_image.GetPixelIDValue())
    resampler.SetInterpolator(interpolator)
    
    return resampler.Execute(sitk_image)


def create_nodule_mask(image_shape, spacing, origin, nodule_annotations):
    """
    Create binary mask for nodules.
    
    Args:
        image_shape: (Z, Y, X) shape of the image array
        spacing: (X, Y, Z) spacing from SimpleITK
        origin: (X, Y, Z) origin from SimpleITK
        nodule_annotations: DataFrame with columns [coordX, coordY, coordZ, diameter_mm]
    
    Returns:
        mask: Binary numpy array of same shape as image
    """
    mask = np.zeros(image_shape, dtype=np.uint8)
    
    if len(nodule_annotations) == 0:
        logger.warning("No nodules found for this scan")
        return mask
    
    for _, nodule in nodule_annotations.iterrows():
        # World coordinates (mm)
        world_x = nodule['coordX']
        world_y = nodule['coordY']
        world_z = nodule['coordZ']
        diameter = nodule['diameter_mm']
        
        # Convert world coordinates to voxel indices
        # Voxel = (World - Origin) / Spacing
        voxel_x = int(round((world_x - origin[0]) / spacing[0]))
        voxel_y = int(round((world_y - origin[1]) / spacing[1]))
        voxel_z = int(round((world_z - origin[2]) / spacing[2]))
        
        # Radius in voxels (diameter to radius, then convert to voxel units)
        radius_mm = diameter / 2.0
        radius_x = int(np.ceil(radius_mm / spacing[0]))
        radius_y = int(np.ceil(radius_mm / spacing[1]))
        radius_z = int(np.ceil(radius_mm / spacing[2]))
        
        # Create spherical mask
        z_min = max(0, voxel_z - radius_z)
        z_max = min(image_shape[0], voxel_z + radius_z + 1)
        y_min = max(0, voxel_y - radius_y)
        y_max = min(image_shape[1], voxel_y + radius_y + 1)
        x_min = max(0, voxel_x - radius_x)
        x_max = min(image_shape[2], voxel_x + radius_x + 1)
        
        for z in range(z_min, z_max):
            for y in range(y_min, y_max):
                for x in range(x_min, x_max):
                    # Check if point is inside sphere
                    dist_sq = (
                        ((x - voxel_x) * spacing[0]) ** 2 +
                        ((y - voxel_y) * spacing[1]) ** 2 +
                        ((z - voxel_z) * spacing[2]) ** 2
                    )
                    if dist_sq <= radius_mm ** 2:
                        mask[z, y, x] = 1
    
    nodule_count = len(nodule_annotations)
    voxel_count = np.sum(mask)
    logger.debug(f"Created mask with {nodule_count} nodules, {voxel_count} voxels marked")
    
    return mask


def process_luna16_scan(mhd_path, annotations_df, output_base_dir, subject_id, 
                        target_spacing=(1.5, 1.5, 1.5)):
    """
    Process a single LUNA16 scan: convert to NIfTI and create mask.
    
    Args:
        mhd_path: Path to .mhd file
        annotations_df: Full annotations DataFrame
        output_base_dir: Base output directory
        subject_id: Subject ID (e.g., "001")
        target_spacing: Target spacing for resampling
    
    Returns:
        success: Boolean indicating if processing succeeded
    """
    try:
        # Extract seriesuid from filename (e.g., "1.3.6.1.4.1.14519.5.2.1.6279.6001.123.mhd")
        seriesuid = mhd_path.stem
        
        logger.info(f"\nProcessing {subject_id}: {seriesuid[:40]}...")
        
        # Read image
        sitk_image = read_mhd(mhd_path)
        if sitk_image is None:
            return False
        
        original_spacing = sitk_image.GetSpacing()
        original_size = sitk_image.GetSize()
        logger.info(f"  Original size: {original_size}, spacing: {original_spacing}")
        
        # Resample to consistent spacing
        logger.info(f"  Resampling to spacing {target_spacing}...")
        resampled_image = resample_image(sitk_image, new_spacing=target_spacing)
        
        new_size = resampled_image.GetSize()
        new_spacing = resampled_image.GetSpacing()
        logger.info(f"  Resampled size: {new_size}, spacing: {new_spacing}")
        
        # Create output directories
        image_dir = output_base_dir / f"sub-{subject_id}" / "anat"
        label_dir = output_base_dir / "derivatives" / "labels" / f"sub-{subject_id}" / "anat"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        
        # Save image as NIfTI
        image_output_path = image_dir / f"sub-{subject_id}_CT.nii.gz"
        image_shape, _ = sitk_to_nifti(resampled_image, image_output_path)
        logger.info(f"  ✓ Saved image: {image_output_path} (shape: {image_shape})")
        
        # Get nodule annotations for this scan
        scan_annotations = annotations_df[annotations_df['seriesuid'] == seriesuid]
        
        # Create mask
        logger.info(f"  Creating mask with {len(scan_annotations)} nodules...")
        array = sitk.GetArrayFromImage(resampled_image)
        mask = create_nodule_mask(
            image_shape=array.shape,
            spacing=new_spacing,
            origin=resampled_image.GetOrigin(),
            nodule_annotations=scan_annotations
        )
        
        # Verify shapes match
        assert mask.shape == array.shape, f"Shape mismatch: mask {mask.shape} vs image {array.shape}"
        
        # Save mask as NIfTI
        mask_output_path = label_dir / f"sub-{subject_id}_CT_seg-nodule_mask.nii.gz"
        affine = np.eye(4)
        affine[0, 0] = new_spacing[0]
        affine[1, 1] = new_spacing[1]
        affine[2, 2] = new_spacing[2]
        affine[0, 3] = resampled_image.GetOrigin()[0]
        affine[1, 3] = resampled_image.GetOrigin()[1]
        affine[2, 3] = resampled_image.GetOrigin()[2]
        
        mask_nifti = nib.Nifti1Image(mask, affine)
        nib.save(mask_nifti, str(mask_output_path))
        logger.info(f"  ✓ Saved mask: {mask_output_path} (shape: {mask.shape})")
        
        # Verify both files exist and have same shape
        img_check = nib.load(str(image_output_path))
        mask_check = nib.load(str(mask_output_path))
        assert img_check.shape == mask_check.shape, \
            f"Final shape mismatch: {img_check.shape} vs {mask_check.shape}"
        
        logger.info(f"  ✓ Verified: Image and mask shapes match: {img_check.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to process {mhd_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_bids_structure(output_dir):
    """Create BIDS-compatible directory structure and metadata files."""
    
    # Create dataset_description.json
    dataset_desc = {
        "Name": "LUNA16 Lung Nodule Detection Dataset",
        "BIDSVersion": "1.6.0",
        "DatasetType": "raw",
        "License": "CC BY-SA 4.0",
        "Authors": ["LUNA16 Challenge Organizers"],
        "Acknowledgements": "LUNA16 Grand Challenge",
        "HowToAcknowledge": "Please cite the LUNA16 challenge paper",
        "DatasetDOI": "10.1109/TMI.2017.2737841",
        "Description": "Preprocessed LUNA16 dataset for lung nodule segmentation"
    }
    
    desc_path = output_dir / "dataset_description.json"
    with open(desc_path, 'w') as f:
        json.dump(dataset_desc, f, indent=2)
    logger.info(f"Created: {desc_path}")
    
    # Create README
    readme_content = """# LUNA16 Preprocessed Dataset

This dataset contains preprocessed CT scans from the LUNA16 lung nodule detection challenge.

## Processing Steps
1. Converted from .mhd/.raw to NIfTI format
2. Resampled to consistent spacing (1.5mm x 1.5mm x 1.5mm)
3. Created binary segmentation masks from nodule annotations
4. Organized in BIDS-compatible structure

## Structure
- sub-XXX/anat/sub-XXX_CT.nii.gz: CT scan
- derivatives/labels/sub-XXX/anat/sub-XXX_CT_seg-nodule_mask.nii.gz: Binary nodule mask

## Reference
Setio, A. A. A., et al. (2017). Validation, comparison, and combination of algorithms 
for automatic detection of pulmonary nodules in computed tomography images: 
The LUNA16 challenge. Medical image analysis, 42, 1-13.
"""
    
    readme_path = output_dir / "README"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    logger.info(f"Created: {readme_path}")
    
    # Create derivatives dataset_description.json
    derivatives_dir = output_dir / "derivatives" / "labels"
    derivatives_dir.mkdir(parents=True, exist_ok=True)
    
    derivatives_desc = {
        "Name": "LUNA16 Nodule Segmentation Labels",
        "BIDSVersion": "1.6.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": "LUNA16 Preprocessing Script",
                "Version": "1.0",
                "Description": "Binary masks created from nodule annotations"
            }
        ]
    }
    
    deriv_desc_path = derivatives_dir / "dataset_description.json"
    with open(deriv_desc_path, 'w') as f:
        json.dump(derivatives_desc, f, indent=2)
    logger.info(f"Created: {deriv_desc_path}")


def create_participants_tsv(output_dir, participants_info):
    """Create participants.tsv file."""
    df = pd.DataFrame(participants_info)
    tsv_path = output_dir / "participants.tsv"
    df.to_csv(tsv_path, sep='\t', index=False)
    logger.info(f"Created: {tsv_path}")


def main():
    """Main preprocessing pipeline."""
    
    # Set paths
    project_root = Path(__file__).parent.parent
    input_dir = project_root / "data" / "luna16_raw"
    output_dir = project_root / "data" / "luna16_processed"
    
    subset_dir = input_dir / "subset0"
    annotations_path = input_dir / "annotations.csv"
    
    # Check inputs
    if not subset_dir.exists():
        logger.error(f"Subset directory not found: {subset_dir}")
        logger.error("Please run download_luna16.py first")
        return
    
    if not annotations_path.exists():
        logger.error(f"Annotations file not found: {annotations_path}")
        logger.error("Please download annotations.csv and place it in {input_dir}")
        return
    
    # Read annotations
    logger.info("Reading annotations...")
    annotations_df = pd.read_csv(annotations_path)
    logger.info(f"Found {len(annotations_df)} nodule annotations for {annotations_df['seriesuid'].nunique()} scans")
    
    # Get list of .mhd files
    mhd_files = sorted(list(subset_dir.glob("*.mhd")))
    logger.info(f"Found {len(mhd_files)} .mhd files in subset0")
    
    if len(mhd_files) == 0:
        logger.error("No .mhd files found")
        return
    
    # Limit to first 50 scans
    max_scans = 50
    mhd_files = mhd_files[:max_scans]
    logger.info(f"Processing first {len(mhd_files)} scans")
    
    # Create output structure
    output_dir.mkdir(parents=True, exist_ok=True)
    create_bids_structure(output_dir)
    
    # Process each scan
    participants_info = []
    successful = 0
    failed = 0
    
    print("\n" + "="*70)
    print(f"PROCESSING {len(mhd_files)} LUNA16 SCANS")
    print("="*70)
    
    for idx, mhd_path in enumerate(tqdm(mhd_files, desc="Processing scans")):
        subject_id = f"{idx+1:03d}"
        
        success = process_luna16_scan(
            mhd_path=mhd_path,
            annotations_df=annotations_df,
            output_base_dir=output_dir,
            subject_id=subject_id,
            target_spacing=(1.5, 1.5, 1.5)
        )
        
        if success:
            # Get nodule count for this scan
            seriesuid = mhd_path.stem
            nodule_count = len(annotations_df[annotations_df['seriesuid'] == seriesuid])
            
            participants_info.append({
                "participant_id": f"sub-{subject_id}",
                "seriesuid": seriesuid,
                "nodule_count": nodule_count
            })
            successful += 1
        else:
            failed += 1
    
    # Create participants.tsv
    if participants_info:
        create_participants_tsv(output_dir, participants_info)
    
    # Summary
    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE")
    print("="*70)
    print(f"✓ Successfully processed: {successful}/{len(mhd_files)} scans")
    if failed > 0:
        print(f"✗ Failed: {failed} scans")
    print(f"\nOutput directory: {output_dir}")
    print("\nDataset structure:")
    print("  sub-XXX/anat/sub-XXX_CT.nii.gz")
    print("  derivatives/labels/sub-XXX/anat/sub-XXX_CT_seg-nodule_mask.nii.gz")
    print("\nNext step: Use this dataset for training")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()



