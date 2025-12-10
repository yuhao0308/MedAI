"""
Test LUNA16 Pipeline
Quick test script to verify preprocessing and training pipeline work correctly.

Run this before uploading to Kaggle to catch any issues early.
"""

import sys
from pathlib import Path
import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("="*70)
print("LUNA16 PIPELINE TEST")
print("="*70 + "\n")

# ============================================
# Test 1: Check Preprocessing Output
# ============================================

print("TEST 1: Checking preprocessing output...")
print("-"*70)

processed_dir = project_root / "data" / "luna16_processed"

if not processed_dir.exists():
    print("❌ Preprocessed data not found!")
    print(f"   Expected: {processed_dir}")
    print("\n   Run preprocessing first:")
    print("   python scripts/preprocess_luna16.py")
    sys.exit(1)

# Check structure
subjects = sorted([d for d in processed_dir.iterdir() if d.is_dir() and d.name.startswith("sub-")])
print(f"✓ Found {len(subjects)} subjects")

if len(subjects) == 0:
    print("❌ No subjects found!")
    sys.exit(1)

# Check first subject
sub_id = subjects[0].name
print(f"\nChecking {sub_id}...")

image_path = processed_dir / sub_id / "anat" / f"{sub_id}_CT.nii.gz"
label_path = processed_dir / "derivatives" / "labels" / sub_id / "anat" / f"{sub_id}_CT_seg-nodule_mask.nii.gz"

if not image_path.exists():
    print(f"❌ Image not found: {image_path}")
    sys.exit(1)

if not label_path.exists():
    print(f"❌ Label not found: {label_path}")
    sys.exit(1)

print(f"✓ Image exists: {image_path.name}")
print(f"✓ Label exists: {label_path.name}")

# Check shapes match
try:
    import nibabel as nib
    
    img = nib.load(str(image_path))
    mask = nib.load(str(label_path))
    
    print(f"\nShape check:")
    print(f"  Image: {img.shape}")
    print(f"  Mask:  {mask.shape}")
    
    if img.shape == mask.shape:
        print("  ✅ Shapes match!")
    else:
        print("  ❌ Shapes DON'T match!")
        sys.exit(1)
    
    # Check spacing
    img_spacing = img.header.get_zooms()[:3]
    mask_spacing = mask.header.get_zooms()[:3]
    
    print(f"\nSpacing check:")
    print(f"  Image: {img_spacing}")
    print(f"  Mask:  {mask_spacing}")
    
    if np.allclose(img_spacing, mask_spacing, atol=0.01):
        print("  ✅ Spacing matches!")
    else:
        print("  ⚠️  Spacing differs (may be OK)")
    
    # Check data ranges
    img_data = img.get_fdata()
    mask_data = mask.get_fdata()
    
    print(f"\nData range check:")
    print(f"  Image: [{img_data.min():.1f}, {img_data.max():.1f}]")
    print(f"  Mask:  [{mask_data.min():.1f}, {mask_data.max():.1f}]")
    
    nodule_voxels = np.sum(mask_data > 0)
    total_voxels = mask_data.size
    nodule_percent = 100 * nodule_voxels / total_voxels
    
    print(f"  Nodule voxels: {nodule_voxels:,} ({nodule_percent:.2f}%)")
    
    if nodule_voxels > 0:
        print("  ✅ Mask contains nodules!")
    else:
        print("  ⚠️  Mask has no nodules (may be normal)")
    
except ImportError:
    print("⚠️  nibabel not installed, skipping detailed checks")
    print("   Install with: pip install nibabel")

print("\n✅ TEST 1 PASSED: Preprocessing output looks good\n")

# ============================================
# Test 2: Load Data with Transforms
# ============================================

print("\nTEST 2: Loading data with transforms...")
print("-"*70)

try:
    from monai.transforms import (
        Compose, LoadImaged, EnsureChannelFirstd,
        Spacingd, Orientationd, ScaleIntensityRanged,
        ToTensord, DivisiblePadd
    )
    
    # Create simple test transform pipeline
    test_transforms = Compose([
        LoadImaged(keys=["image", "label"], image_only=False),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(
            keys=["image", "label"],
            pixdim=[1.5, 1.5, 1.5],
            mode=("bilinear", "nearest"),
        ),
        Orientationd(
            keys=["image", "label"],
            axcodes="RAS",
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-1000.0,
            a_max=400.0,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        DivisiblePadd(
            keys=["image", "label"],
            k=32,
            mode="constant",
        ),
        ToTensord(keys=["image", "label"]),
    ])
    
    # Test on first subject
    test_data = {
        "image": str(image_path),
        "label": str(label_path),
        "subject_id": sub_id,
    }
    
    print(f"Applying transforms to {sub_id}...")
    result = test_transforms(test_data)
    
    print(f"\nResult shapes:")
    print(f"  Image: {result['image'].shape}")
    print(f"  Label: {result['label'].shape}")
    
    # Verify shapes match (except channel dimension)
    if result['image'].shape[1:] == result['label'].shape[1:]:
        print("  ✅ Spatial dimensions match!")
    else:
        print("  ❌ Spatial dimensions DON'T match!")
        sys.exit(1)
    
    # Check divisibility by 32
    spatial_dims = result['image'].shape[2:]
    divisible = all(d % 32 == 0 for d in spatial_dims)
    print(f"\nDivisible by 32: {divisible}")
    if divisible:
        print("  ✅ All dimensions divisible by 32!")
    else:
        print("  ❌ Not all dimensions divisible by 32!")
        sys.exit(1)
    
    print("\n✅ TEST 2 PASSED: Transform pipeline works correctly\n")
    
except ImportError as e:
    print(f"⚠️  MONAI not installed: {e}")
    print("   Install with: pip install monai")
    print("   Skipping transform test")

# ============================================
# Test 3: Model Forward Pass
# ============================================

print("\nTEST 3: Testing model forward pass...")
print("-"*70)

try:
    from monai.networks.nets import SwinUNETR
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create model
    model = SwinUNETR(
        img_size=[96, 96, 96],
        in_channels=1,
        out_channels=2,
        feature_size=48,
        use_checkpoint=True,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Test forward pass
    test_input = torch.randn(1, 1, 96, 96, 96).to(device)
    
    print(f"\nForward pass...")
    print(f"  Input shape: {test_input.shape}")
    
    with torch.no_grad():
        output = model(test_input)
    
    print(f"  Output shape: {output.shape}")
    
    if output.shape == (1, 2, 96, 96, 96):
        print("  ✅ Output shape correct!")
    else:
        print(f"  ❌ Unexpected output shape!")
        sys.exit(1)
    
    print("\n✅ TEST 3 PASSED: Model works correctly\n")
    
except ImportError as e:
    print(f"⚠️  Could not import model: {e}")
    print("   Skipping model test")

# ============================================
# Test 4: Check Dataset Files
# ============================================

print("\nTEST 4: Checking all dataset files...")
print("-"*70)

errors = []
for subject in subjects[:10]:  # Check first 10
    sub_id = subject.name
    
    img_path = processed_dir / sub_id / "anat" / f"{sub_id}_CT.nii.gz"
    lbl_path = processed_dir / "derivatives" / "labels" / sub_id / "anat" / f"{sub_id}_CT_seg-nodule_mask.nii.gz"
    
    if not img_path.exists():
        errors.append(f"{sub_id}: Image missing")
    if not lbl_path.exists():
        errors.append(f"{sub_id}: Label missing")
    
    if img_path.exists() and lbl_path.exists():
        if img_path.stat().st_size == 0:
            errors.append(f"{sub_id}: Image empty")
        if lbl_path.stat().st_size == 0:
            errors.append(f"{sub_id}: Label empty")

if errors:
    print("❌ Found errors:")
    for error in errors:
        print(f"  - {error}")
    sys.exit(1)
else:
    print(f"✅ All {min(10, len(subjects))} checked subjects OK")

print("\n✅ TEST 4 PASSED: Dataset files are valid\n")

# ============================================
# Summary
# ============================================

print("\n" + "="*70)
print("ALL TESTS PASSED! ✅")
print("="*70)
print("\nYour LUNA16 dataset is ready for Kaggle!")
print("\nNext steps:")
print("1. Compress the dataset:")
print("   cd data")
print("   zip -r luna16_processed.zip luna16_processed/")
print("\n2. Upload to Kaggle as a dataset")
print("\n3. Copy scripts/kaggle_luna16_training.py to Kaggle notebook")
print("\n4. Update bids_root path and run!")
print("\nSee LUNA16_KAGGLE_GUIDE.md for detailed instructions.")
print("="*70 + "\n")



