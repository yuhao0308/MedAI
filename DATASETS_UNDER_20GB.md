# DICOM Datasets Under 20GB

## Recommended Datasets (Easiest to Download)

### 1. **MIPG Datasets** (University of Pennsylvania) - **BEST FOR QUICK START**
- **Size**: 426MB - 836MB per dataset
- **Format**: DICOM with segmentations
- **Regions**: Neck, Thorax, Abdomen, Pelvis, Torso
- **Download**: https://www.mipg.upenn.edu/Vnews/mipg_data_sharing.html
- **Why**: Small, well-organized, includes segmentations (labels)

### 2. **3DICOM Sample Library** - **BEST FOR TESTING**
- **CT Scan of COVID-19 Lung**: 135MB
- **CT Scan of Skull Base**: 50MB
- **Circle of Willis**: 20.2MB
- **Download**: https://3dicomviewer.com/dicom-library/
- **Why**: Very small, perfect for pipeline testing

### 3. **Aliza Medical Imaging Samples**
- **MR DWI samples**: Various sizes (typically < 500MB)
- **Multiple manufacturers**: GE, Philips, Siemens
- **Download**: https://sites.google.com/site/alizaviewer/download/datasets
- **Why**: Good for testing DICOM compatibility

## Medium-Sized Datasets (5-15GB)

### 4. **LIDC-IDRI Subset** (TCIA)
- **Full size**: ~120GB
- **Subset option**: Download first 20-30 patients (~5-10GB)
- **Type**: Lung CT with nodule annotations
- **Download**: https://www.cancerimagingarchive.net/collection/lidc-idri/
- **Why**: Well-documented, has annotations, widely used

### 5. **OASIS Datasets** (Neuroimaging)
- **Size**: Various, can select subsets
- **Type**: Brain MRI
- **Download**: https://www.oasis-brains.org/
- **Why**: Standard neuroimaging dataset

### 6. **CT-RATE Subset** (Hugging Face)
- **Full size**: Large
- **Subset option**: Select first 100-200 volumes (~5-10GB)
- **Type**: Chest CT with text reports
- **Download**: https://huggingface.co/datasets
- **Why**: Includes text reports (good for multimodal)

## How to Download

### Option 1: MIPG Datasets (Recommended)

1. Go to: https://www.mipg.upenn.edu/Vnews/mipg_data_sharing.html
2. Click on a dataset (e.g., "Neck Dataset" - 426MB)
3. Download and extract to `./data/raw_dicom/`

### Option 2: 3DICOM Samples (Quick Test)

1. Go to: https://3dicomviewer.com/dicom-library/
2. Download any sample (e.g., "CT Scan of COVID-19 Lung" - 135MB)
3. Extract to `./data/raw_dicom/`

### Option 3: TCIA LIDC-IDRI (Subset)

**Using Web Interface:**
1. Go to: https://www.cancerimagingarchive.net/collection/lidc-idri/
2. Click "Download" → "Download Manifest"
3. Use NBIA Data Retriever to download first 20-30 patients only
4. Save to `./data/raw_dicom/`

**Using our script (if you have NBIA client):**
```bash
# This will download the full dataset, so we'll create a subset script
python scripts/download_tcia_subset.py --collection LIDC-IDRI --max-patients 25 --output ./data/raw_dicom
```

## Quick Start Commands

After downloading any dataset to `./data/raw_dicom/`:

```bash
# Process the DICOM files
python scripts/ingest_data.py --input-dir ./data/raw_dicom --modality CT --max-subjects 10

# Check the BIDS structure
ls -la ./data/bids_dataset/
```

## Size Comparison

| Dataset | Size | Download Time | Best For |
|---------|------|---------------|----------|
| 3DICOM Samples | 20MB - 135MB | < 1 min | Quick testing |
| MIPG Neck | 426MB | ~2 min | Development |
| MIPG Thorax | 836MB | ~5 min | Development |
| LIDC-IDRI (25 patients) | ~5-10GB | ~30-60 min | Full pipeline |
| OASIS Subset | 5-15GB | ~30-60 min | Neuroimaging |

## Recommendation

**For your first test**: Start with **3DICOM COVID-19 Lung CT** (135MB)
- Very fast download
- Perfect for testing the pipeline
- Real medical data

**For development**: Use **MIPG Thorax Dataset** (836MB)
- Includes segmentations
- Well-organized
- Good for training/testing

**For full pipeline**: Use **LIDC-IDRI subset** (5-10GB)
- Industry standard
- Has annotations
- Widely documented


