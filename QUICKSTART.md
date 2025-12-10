# Quick Start Guide

## Step 1: Install Dependencies

```bash
# Install Python packages
pip install -r requirements.txt

# Install dcm2niix (required for DICOM conversion)
# macOS:
brew install dcm2niix

# Linux:
sudo apt-get install dcm2niix

# Windows: Download from https://github.com/rordenlab/dcm2niix/releases
```

## Step 2: Download a Public Dataset (< 20GB)

Since you want to replace the MyHead/ folder with common online DICOM data under 20GB, here are your best options:

### Option A: 3DICOM Samples (FASTEST - 50-135MB) ⭐ RECOMMENDED

Perfect for quick testing:

1. Go to: https://3dicomviewer.com/dicom-library/
2. Download "CT Scan of COVID-19 Lung" (135MB) or "CT Scan of Skull Base" (50MB)
3. Extract to `./data/raw_dicom/`

**Or get instructions:**
```bash
python scripts/download_small_datasets.py --source 3dicom
```

### Option B: MIPG Datasets (BEST FOR DEVELOPMENT - 426MB-2GB) ⭐ RECOMMENDED

Includes segmentations, perfect for training:

1. Go to: https://www.mipg.upenn.edu/Vnews/mipg_data_sharing.html
2. Download "Neck Dataset" (426MB) or "Thorax Dataset" (836MB)
3. Extract to `./data/raw_dicom/`

**Or get instructions:**
```bash
python scripts/download_small_datasets.py --source mipg --dataset thorax
```

### Option C: TCIA LIDC-IDRI Subset (5-10GB)

For a more complete dataset:

1. Go to: https://www.cancerimagingarchive.net/collection/lidc-idri/
2. Download first 20-30 patients only (selectively via NBIA Data Retriever)
3. Extract to `./data/raw_dicom/`

**See full guide:** `DATASETS_UNDER_20GB.md`

## Step 3: Process Your Data

Once you have DICOM files in `./data/raw_dicom/`, run:

```python
# Create a simple ingestion script
from src.ingestion.dicom_parser import get_series_metadata, filter_series
from src.ingestion.dicom_to_nifti import convert_dicom_to_nifti, batch_convert_dicom_to_nifti
from src.ingestion.bids_organizer import (
    create_bids_structure,
    create_dataset_description,
    create_participants_tsv,
    organize_nifti_to_bids
)

# 1. Parse DICOM files
print("Parsing DICOM files...")
series_map = get_series_metadata("./data/raw_dicom")
print(f"Found {len(series_map)} series")

# 2. Filter for your target modality (e.g., CT or MR)
target_series = filter_series(series_map, target_modality="CT")  # or "MR"
print(f"Filtered to {len(target_series)} target series")

# 3. Create BIDS structure
print("Creating BIDS structure...")
create_bids_structure("./data/bids_dataset")
create_dataset_description("./data/bids_dataset")

# 4. Convert to NIfTI and organize
participants_data = []
for idx, series_dir in enumerate(target_series[:5]):  # Process first 5 for testing
    participant_id = f"{idx+1:03d}"
    
    # Convert to NIfTI
    nifti_path = convert_dicom_to_nifti(
        input_dir=series_dir,
        output_dir="./data/temp_nifti",
        output_filename=f"sub-{participant_id}_T1w"
    )
    
    # Organize into BIDS
    organize_nifti_to_bids(
        nifti_file=nifti_path,
        bids_root="./data/bids_dataset",
        participant_id=participant_id,
        modality="anat",
        suffix="T1w"
    )
    
    # Collect participant metadata
    participants_data.append({
        "participant_id": f"sub-{participant_id}",
        "modality": "CT"  # or "MR"
    })

# 5. Create participants.tsv
create_participants_tsv("./data/bids_dataset", participants_data)

print("Data ingestion complete!")
```

Save this as `scripts/ingest_data.py` and run:
```bash
python scripts/ingest_data.py
```

## Step 4: Remove MyHead/ Folder (Optional)

Once you have new data processed:

```bash
# Backup first (optional)
mv MyHead MyHead_backup

# Or remove if you're sure
rm -rf MyHead
```

## Step 5: Test the Pipeline

### Test Preprocessing

```python
from src.preprocessing.dataloader import read_bids_dataset
from src.preprocessing.transforms import get_train_transforms

# Load BIDS dataset
data_dicts = read_bids_dataset("./data/bids_dataset")
print(f"Found {len(data_dicts)} image-label pairs")

# Note: You'll need labels for training. If you don't have labels yet,
# you can still test the preprocessing on images only.
```

### Test MONAI Label Setup (For Annotation)

```bash
python scripts/setup_annotation.py --bids-root ./data/bids_dataset --instructions-only
```

## Next Steps

1. **If you have labels**: Proceed to training
2. **If you don't have labels**: Set up MONAI Label for annotation
3. **For multimodal**: Prepare EHR/clinical notes data

## Recommended First Dataset

For a quick start, I recommend:
- **LIDC-IDRI** from TCIA (lung nodule detection - has annotations)
- Or any dataset from the Medical Segmentation Decathlon

These are well-documented and commonly used for testing medical AI pipelines.

