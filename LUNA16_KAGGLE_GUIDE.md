# LUNA16 Kaggle Training Guide

Complete guide for training lung nodule segmentation on Kaggle with LUNA16 dataset.

## 🎯 Quick Start

1. **Download & Preprocess LUNA16** (on your local machine)
2. **Upload to Kaggle** as a dataset
3. **Copy-paste training script** into Kaggle notebook
4. **Run** and monitor training

---

## 📦 Step 1: Download LUNA16 Dataset

### Option A: Manual Download (Recommended)

1. Register at: https://luna16.grand-challenge.org/Download/
2. Download these files:
   - `subset0.zip` (~40GB) - Contains ~88 CT scans
   - `annotations.csv` - Nodule annotations (coordinates & diameters)
3. Place them in: `/Users/yuhaowang/Documents/Itlize/MedAI/data/luna16_raw/`

### Option B: Use Download Script

```bash
cd /Users/yuhaowang/Documents/Itlize/MedAI
python scripts/download_luna16.py
```

This will create a README with download instructions.

---

## 🔧 Step 2: Preprocess LUNA16

Convert .mhd/.raw files to NIfTI with binary masks:

```bash
python scripts/preprocess_luna16.py
```

**What it does:**
- Converts first 50 scans from subset0
- Creates binary segmentation masks from annotations
- Resamples to consistent spacing (1.5mm isotropic)
- Saves in BIDS-compatible structure
- **Ensures image and mask have EXACT same shape** (critical!)

**Output structure:**
```
data/luna16_processed/
├── sub-001/anat/sub-001_CT.nii.gz
├── sub-002/anat/sub-002_CT.nii.gz
├── ...
├── derivatives/labels/
│   ├── sub-001/anat/sub-001_CT_seg-nodule_mask.nii.gz
│   ├── sub-002/anat/sub-002_CT_seg-nodule_mask.nii.gz
│   └── ...
├── dataset_description.json
├── participants.tsv
└── README
```

**Time:** ~30-60 minutes for 50 scans

---

## ☁️ Step 3: Upload to Kaggle

### Create Kaggle Dataset

1. Compress the processed data:
   ```bash
   cd data
   zip -r luna16_processed.zip luna16_processed/
   ```

2. Upload to Kaggle:
   - Go to: https://www.kaggle.com/datasets
   - Click "New Dataset"
   - Upload `luna16_processed.zip`
   - Title: "LUNA16 Preprocessed (50 samples)"
   - Make it **Private** (dataset is large)

3. Note your dataset URL, e.g.:
   ```
   https://www.kaggle.com/datasets/YOUR_USERNAME/luna16-processed
   ```

---

## 🚀 Step 4: Run Training on Kaggle

### Create New Notebook

1. Go to: https://www.kaggle.com/code
2. Click "New Notebook"
3. **Enable GPU Accelerator:**
   - Settings → Accelerator → GPU T4 x2 (or P100)
4. Add your dataset:
   - Add Data → Your Datasets → Select "LUNA16 Preprocessed"

### Copy Training Script

Open: `scripts/kaggle_luna16_training.py`

**Copy the ENTIRE script** and paste into a Kaggle code cell.

### Update Configuration

Find this line in the script (near line 72):

```python
"bids_root": "/kaggle/input/luna16-processed",
```

Update to match your Kaggle dataset path:
```python
"bids_root": "/kaggle/input/YOUR-DATASET-NAME",
```

### Run!

Click "Run All" and monitor training.

**Expected output:**
```
Loading dataset...
✓ Found 50 valid image-label pairs
Dataset split:
  Train: 35 samples
  Val: 8 samples
  Test: 7 samples

Creating model...
Model parameters: 62,191,042
✓ Forward pass successful!

STARTING TRAINING
Epoch 1/100
  Batch 0/35, Loss: 0.8234
  ...
```

---

## 📊 Training Configuration

### Default Settings

| Parameter | Value | Notes |
|-----------|-------|-------|
| Model | SwinUNETR | State-of-the-art 3D segmentation |
| Input size | 96×96×96 | Patch-based training |
| Batch size | 1 | Typical for 3D medical imaging |
| Learning rate | 1e-4 | Adam optimizer |
| Epochs | 100 | Can reduce for testing |
| Val interval | 2 | Validate every 2 epochs |

### Memory Optimization

If you encounter OOM (Out Of Memory) errors:

1. **Reduce feature size:**
   ```python
   "feature_size": 24,  # Instead of 48
   ```

2. **Reduce input size:**
   ```python
   "img_size": [64, 64, 64],  # Instead of [96, 96, 96]
   "spatial_size": [64, 64, 64],
   ```

3. **Reduce cache rate:**
   ```python
   "cache_rate": 0.25,  # Instead of 0.5
   ```

---

## 🐛 Troubleshooting

### Error: "BIDS root does not exist"

**Problem:** Dataset path is incorrect.

**Solution:** Check the dataset name in Kaggle:
```python
# List available datasets
!ls /kaggle/input/
```

Update `bids_root` in CONFIG accordingly.

---

### Error: "Shape mismatch: inputs (1,1,96,96,96) vs labels (1,1,96,96,95)"

**Problem:** Image and label have different shapes.

**Solution:** This should NOT happen with the preprocessing script. If it does:
1. Re-run preprocessing: `python scripts/preprocess_luna16.py`
2. Check that you uploaded the correct `luna16_processed` folder
3. Verify shapes locally:
   ```python
   import nibabel as nib
   img = nib.load("sub-001/anat/sub-001_CT.nii.gz")
   mask = nib.load("derivatives/labels/sub-001/anat/sub-001_CT_seg-nodule_mask.nii.gz")
   print(f"Image: {img.shape}, Mask: {mask.shape}")
   # Should be identical!
   ```

---

### Error: "CUDA out of memory"

**Problem:** GPU memory insufficient.

**Solution:** See "Memory Optimization" above. Try:
```python
CONFIG["model"]["feature_size"] = 24
CONFIG["model"]["img_size"] = [64, 64, 64]
CONFIG["training"]["spatial_size"] = [64, 64, 64]
```

---

### Error: "No nodules found for this scan"

**Problem:** Scan has no nodules (normal).

**Solution:** This is just a warning. Many scans have no nodules. The model learns background + nodule segmentation.

---

### Training is very slow

**Problem:** CPU being used instead of GPU, or slow data loading.

**Solution:**
1. Verify GPU is enabled:
   ```python
   print(torch.cuda.is_available())  # Should be True
   print(torch.cuda.get_device_name(0))  # Should show GPU name
   ```

2. Reduce data loading workers:
   ```python
   "num_workers": 0,  # Use 0 on Kaggle
   ```

3. Reduce cache rate or disable caching:
   ```python
   "cache_rate": 0.0,  # Disable caching
   ```

---

## 📈 Expected Results

### Training Time

| GPU | Time per Epoch | 100 Epochs |
|-----|----------------|------------|
| T4 x2 | ~8-10 min | ~14 hours |
| P100 | ~5-7 min | ~9 hours |
| V100 | ~3-5 min | ~6 hours |

### Performance

After 100 epochs with 50 samples:

| Metric | Expected Value |
|--------|----------------|
| Train Loss | 0.15 - 0.25 |
| Val Loss | 0.20 - 0.30 |
| Val Dice | 0.60 - 0.75 |

**Note:** LUNA16 nodule segmentation is challenging. Dice > 0.70 is good!

---

## 📁 Output Files

After training, check `/kaggle/working/` for:

| File | Description |
|------|-------------|
| `best_model.pth` | Best model (highest val Dice) |
| `final_model.pth` | Final model after all epochs |
| `training_history.json` | Loss and metrics per epoch |
| `training_curves.png` | Loss and Dice plots |
| `config.json` | Configuration used |
| `data_splits.json` | Train/val/test split |
| `checkpoint_epochX.pth` | Checkpoints every 10 epochs |

---

## 🔍 Validating Results

### Load Best Model

```python
import torch
from monai.networks.nets import SwinUNETR

# Load config and model
checkpoint = torch.load("/kaggle/working/best_model.pth")
config = checkpoint["config"]

model = SwinUNETR(
    img_size=config["model"]["img_size"],
    in_channels=1,
    out_channels=2,
    feature_size=48,
).cuda()

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print(f"Loaded model from epoch {checkpoint['epoch']}")
print(f"Dice score: {checkpoint['dice_score']:.4f}")
```

### Run Inference on Test Set

```python
# Use test split from data_splits.json
import json
with open("/kaggle/working/data_splits.json") as f:
    splits = json.load(f)

test_subjects = splits["test"]
print(f"Test subjects: {test_subjects}")

# Run inference (see inference script in project)
```

---

## 🎓 Advanced: Fine-tuning

### Use Pretrained Weights

SwinUNETR can be initialized with ImageNet pretrained weights:

```python
from monai.networks.nets import SwinUNETR

model = SwinUNETR(
    img_size=[96, 96, 96],
    in_channels=1,
    out_channels=2,
    feature_size=48,
    use_checkpoint=True,
)

# Load pretrained weights (if available)
# weight = torch.load("pretrained_swin_unetr.pth")
# model.load_from(weight)
```

### Adjust Learning Rate Schedule

```python
# Warm-up + cosine annealing
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,  # Restart every 10 epochs
    T_mult=2,
    eta_min=1e-6,
)
```

---

## 📚 Key Fixes Applied

This script fixes ALL common issues from the original code:

### ✅ Shape Mismatch Issues

**Original Problem:** Image and label shapes don't match → training crashes

**Fix:**
1. Preprocessing ensures exact shape matching
2. `ValidateShapesD` transform catches mismatches early
3. All spatial transforms applied to BOTH image and label together

### ✅ Transform Pipeline Issues

**Original Problem:** Transforms modify shapes inconsistently

**Fix:**
1. Shape validation after each critical transform
2. `DivisiblePadd` ensures dimensions are multiples of 32
3. No more `MatchImageLabelShape` custom transform (error-prone)

### ✅ Data Loading Issues

**Original Problem:** Silent failures, unclear errors

**Fix:**
1. Robust `read_luna16_dataset()` with error reporting
2. File existence and size checks
3. Clear error messages with subject IDs

### ✅ Memory Issues

**Original Problem:** OOM errors on Kaggle GPU

**Fix:**
1. Gradient checkpointing enabled
2. Mixed precision (AMP) training
3. Configurable cache rate and batch size
4. Optimal default settings for Kaggle GPUs

---

## 🆘 Still Having Issues?

### Debug Mode

Enable debug printing to see shapes at each step:

```python
# In the script, find get_train_transforms()
# Add this transform after LoadImaged:

class DebugShapesD(Transform):
    def __call__(self, data):
        print(f"Subject: {data.get('subject_id', 'unknown')}")
        print(f"  Image: {data['image'].shape}")
        print(f"  Label: {data['label'].shape}")
        return data

# Add to pipeline:
DebugShapesD(),
```

### Test with Small Dataset

Test with just 3 samples to quickly identify issues:

```python
# Modify split_dataset() to use only first 3 samples
data_dicts = data_dicts[:3]
```

### Check Preprocessing Output

Before uploading to Kaggle, verify preprocessing locally:

```bash
python scripts/preprocess_luna16.py

# Check one sample
python -c "
import nibabel as nib
img = nib.load('data/luna16_processed/sub-001/anat/sub-001_CT.nii.gz')
mask = nib.load('data/luna16_processed/derivatives/labels/sub-001/anat/sub-001_CT_seg-nodule_mask.nii.gz')
print(f'Image shape: {img.shape}')
print(f'Mask shape: {mask.shape}')
print(f'Shapes match: {img.shape == mask.shape}')
"
```

---

## 📞 Contact & Resources

- **LUNA16 Challenge:** https://luna16.grand-challenge.org/
- **MONAI Documentation:** https://docs.monai.io/
- **SwinUNETR Paper:** https://arxiv.org/abs/2201.01266
- **Kaggle GPU Quotas:** https://www.kaggle.com/code

---

## ✅ Checklist

Before running on Kaggle:

- [ ] Downloaded LUNA16 subset0 and annotations.csv
- [ ] Ran preprocessing script successfully
- [ ] Verified shapes match for at least one sample
- [ ] Created Kaggle dataset and uploaded
- [ ] Created new Kaggle notebook with GPU enabled
- [ ] Added dataset to notebook
- [ ] Copied training script
- [ ] Updated `bids_root` path
- [ ] Started training and monitoring

---

**Good luck with your training! 🚀**

If preprocessing and upload are done correctly, the training script should run without errors. All shape mismatch issues have been fixed!



