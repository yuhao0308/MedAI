"""
LUNA16 Lung Nodule Segmentation Training Script
Bug-free 3D medical image segmentation with MONAI and SwinUNETR

This is a complete, copy-paste ready script for Kaggle.
All shape mismatch issues have been fixed.

USAGE ON KAGGLE:
1. Upload LUNA16 preprocessed dataset as a Kaggle dataset
2. Create new notebook and enable GPU accelerator
3. Copy-paste this entire script into a code cell
4. Update BIDS_ROOT path to match your dataset
5. Run!
"""

# ============================================
# SECTION 1: ENVIRONMENT SETUP
# ============================================

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
torch.cuda.empty_cache()

import sys
from pathlib import Path
import json
from datetime import datetime
import random
import numpy as np
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# PyTorch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

# MONAI
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.data import DataLoader, CacheDataset, Dataset, decollate_batch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
    Orientationd, ScaleIntensityRanged, RandCropByPosNegLabeld,
    RandRotate90d, RandFlipd, ToTensord, Transform,
    DivisiblePadd, AsDiscrete, CropForegroundd, RandGaussianNoised
)
from monai.inferers import sliding_window_inference

# Scipy & Nibabel
import nibabel as nib
from scipy import ndimage

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

# ============================================
# SECTION 2: CONFIGURATION
# ============================================

CONFIG = {
    "paths": {
        # 🔧 CHANGE THIS to match your Kaggle dataset path
        "bids_root": "/kaggle/input/luna16-processed/luna16_processed",
        "output_dir": "/kaggle/working",
    },
    "selection": {
        "subset_size": 50,
        "target_positive_ratio": 0.6,
        "random_seed": 42,
    },
    "model": {
        "type": "SwinUNETR",
        "in_channels": 1,
        "out_channels": 2,  # Background + Nodule
        "img_size": [96, 96, 96],  # Model input size (must be divisible by 32)
        "feature_size": 48,  # Reduce to 24 if OOM
        "use_checkpoint": True,
    },
    
    "training": {
        "spatial_size": [96, 96, 96],  # Patch size (matches img_size)
        "batch_size": 1,
        "num_epochs": 80,
        "learning_rate": 1e-4,
        "num_workers": 2,
        "cache_rate": 0.5,
        "val_interval": 2,
        "randcrop": {
            "pos": 4,  # Increased from 3 - more positive samples
            "neg": 1,
            "num_samples": 4,
        },
    },
    
    "preprocessing": {
        "spacing": [1.5, 1.5, 1.5],
        "orientation": "RAS",
        "intensity_range": [-1000.0, 400.0],  # Lung CT window
        "target_range": [0.0, 1.0],
        "divisible_k": 32,
    },
}

OUTPUT_DIR = Path(CONFIG["paths"]["output_dir"])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_DIR / "config.json", "w") as f:
    json.dump(CONFIG, f, indent=2)

print("\n" + "="*70)
print("CONFIGURATION")
print("="*70)
print(json.dumps(CONFIG, indent=2))
print("="*70 + "\n")

# ============================================
# SECTION 3: DATA LOADING & STRATIFIED SPLIT
# ============================================

def read_luna16_dataset(bids_root: Path, verbose: bool = True) -> List[Dict[str, str]]:
    """
    Read LUNA16 dataset in BIDS format with robust error handling.
    """
    if verbose:
        print(f"\n📂 Scanning BIDS dataset: {bids_root}")
    
    bids_root = Path(bids_root)
    
    if not bids_root.exists():
        raise ValueError(f"❌ BIDS root does not exist: {bids_root}")
    
    data_dicts = []
    
    # Find all subject directories
    subject_dirs = sorted([d for d in bids_root.iterdir() 
                          if d.is_dir() and d.name.startswith("sub-")])
    
    if len(subject_dirs) == 0:
        raise ValueError(f"❌ No subject directories found in {bids_root}")
    
    if verbose:
        print(f"   Found {len(subject_dirs)} subjects")
    
    for sub_dir in subject_dirs:
        sub_id = sub_dir.name
        try:
            # Look for image and label files
            img_candidates = list((sub_dir / "anat").glob("*CT*.nii*")) + list((sub_dir / "anat").glob("*.nii*"))
            label_dir = bids_root / "derivatives" / "labels" / sub_id / "anat"
            if not label_dir.exists():
                continue
            lbl_candidates = list(label_dir.glob("*seg*mask.nii*")) + list(label_dir.glob("*.nii*"))
            
            if img_candidates and lbl_candidates:
                data_dicts.append({
                    "image": str(img_candidates[0]),
                    "label": str(lbl_candidates[0]),
                    "subject_id": sub_id,
                })
        except Exception:
            continue
    
    if len(data_dicts) == 0:
        raise ValueError("❌ No valid image-label pairs found!")
    
    return data_dicts

def analyze_subject(data_dict):
    """Analyze if subject has nodules."""
    mask = nib.load(data_dict["label"]).get_fdata()
    has_nodules = np.any(mask > 0)
    info = {
        "subject_id": data_dict["subject_id"],
        "has_nodules": has_nodules,
        "data_dict": data_dict,
    }
    return info

def stratified_select_and_split(data_dicts):
    """Select balanced subset and split stratified."""
    print("   Analyzing subjects for stratification...")
    analyses = [analyze_subject(d) for d in data_dicts]
    positive = [a for a in analyses if a["has_nodules"]]
    negative = [a for a in analyses if not a["has_nodules"]]
    
    random.seed(CONFIG["selection"]["random_seed"])
    
    # Select subset
    n_total = CONFIG["selection"]["subset_size"]
    n_pos = min(int(n_total * CONFIG["selection"]["target_positive_ratio"]), len(positive))
    n_neg = min(n_total - n_pos, len(negative))
    
    selected = random.sample(positive, n_pos) + random.sample(negative, n_neg)
    random.shuffle(selected)
    
    # Helper to split list
    def split_list(items, train_ratio=0.7, val_ratio=0.15):
        random.shuffle(items)
        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        return {
            "train": [i["data_dict"] for i in items[:n_train]],
            "val": [i["data_dict"] for i in items[n_train:n_train+n_val]],
            "test": [i["data_dict"] for i in items[n_train+n_val:]],
        }

    splits = {"train": [], "val": [], "test": []}
    pos_splits = split_list([a for a in selected if a["has_nodules"]])
    neg_splits = split_list([a for a in selected if not a["has_nodules"]])

    for key in ["train", "val", "test"]:
        splits[key] = pos_splits[key] + neg_splits[key]
        random.shuffle(splits[key])
        
    summary = {
        "total": len(selected),
        "train": len(splits["train"]),
        "val": len(splits["val"]),
        "test": len(splits["test"]),
        "train_pos": sum(analyze_subject(d)["has_nodules"] for d in splits["train"]),
        "val_pos": sum(analyze_subject(d)["has_nodules"] for d in splits["val"]),
    }
    
    print(f"\n📊 Selection & Split Summary:")
    print(f"   Selected: {summary['total']} subjects")
    print(f"   Train: {summary['train']} (Pos: {summary['train_pos']})")
    print(f"   Val:   {summary['val']} (Pos: {summary['val_pos']})")
    print(f"   Test:  {summary['test']}")
    
    return splits

# Load dataset
print("\n" + "="*70)
print("LOADING DATASET")
print("="*70)

try:
    BIDS_ROOT = Path(CONFIG["paths"]["bids_root"])
    data_dicts = read_luna16_dataset(BIDS_ROOT, verbose=True)
    splits = stratified_select_and_split(data_dicts)
    
    # Save split info
    split_info = {k: [d["subject_id"] for d in v] for k, v in splits.items()}
    with open(OUTPUT_DIR / "data_splits.json", "w") as f:
        json.dump(split_info, f, indent=2)
    
    print("="*70 + "\n")
    
except Exception as e:
    print(f"\n❌ ERROR loading dataset: {e}")
    raise

# ============================================
# SECTION 4: TRANSFORMS
# ============================================

class BinarizeLabeld(Transform):
    def __init__(self, keys):
        self.keys = keys
    def __call__(self, data):
        for key in self.keys:
            data[key] = (data[key] > 0).float()
        return data

def get_train_transforms(config):
    return Compose([
        LoadImaged(keys=["image", "label"], image_only=False),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=config["preprocessing"]["spacing"], mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes=config["preprocessing"]["orientation"]),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=config["preprocessing"]["intensity_range"][0],
            a_max=config["preprocessing"]["intensity_range"][1],
            b_min=config["preprocessing"]["target_range"][0],
            b_max=config["preprocessing"]["target_range"][1],
            clip=True,
        ),
        BinarizeLabeld(keys=["label"]),
        CropForegroundd(keys=["image", "label"], source_key="image", margin=8, allow_smaller=True),
        DivisiblePadd(keys=["image", "label"], k=config["preprocessing"]["divisible_k"]),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=config["training"]["spatial_size"],
            pos=config["training"]["randcrop"]["pos"],
            neg=config["training"]["randcrop"]["neg"],
            num_samples=config["training"]["randcrop"]["num_samples"],
            allow_smaller=True,
        ),
        RandGaussianNoised(keys=["image"], prob=0.1, mean=0.0, std=0.01),
        RandRotate90d(keys=["image", "label"], prob=0.2, spatial_axes=(0, 1)),
        RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=2),
        ToTensord(keys=["image", "label"]),
    ])

def get_val_transforms(config):
    return Compose([
        LoadImaged(keys=["image", "label"], image_only=False),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=config["preprocessing"]["spacing"], mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes=config["preprocessing"]["orientation"]),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=config["preprocessing"]["intensity_range"][0],
            a_max=config["preprocessing"]["intensity_range"][1],
            b_min=config["preprocessing"]["target_range"][0],
            b_max=config["preprocessing"]["target_range"][1],
            clip=True,
        ),
        BinarizeLabeld(keys=["label"]),
        CropForegroundd(keys=["image", "label"], source_key="image", margin=8, allow_smaller=True),
        DivisiblePadd(keys=["image", "label"], k=config["preprocessing"]["divisible_k"]),
        ToTensord(keys=["image", "label"]),
    ])

print("✅ Transform pipelines defined\n")

# ============================================
# SECTION 5: CREATE DATASETS AND DATALOADERS
# ============================================

train_transforms = get_train_transforms(CONFIG)
val_transforms = get_val_transforms(CONFIG)

train_dataset = CacheDataset(
    data=splits["train"],
    transform=train_transforms,
    cache_rate=CONFIG["training"]["cache_rate"],
    num_workers=CONFIG["training"]["num_workers"],
)
val_dataset = CacheDataset(
    data=splits["val"],
    transform=val_transforms,
    cache_rate=CONFIG["training"]["cache_rate"],
    num_workers=CONFIG["training"]["num_workers"],
)

train_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG["training"]["batch_size"],
    shuffle=True,
    num_workers=0,  # Use 0 for Kaggle
    pin_memory=torch.cuda.is_available(),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

print(f"✅ DataLoaders created (Train: {len(train_loader)}, Val: {len(val_loader)})")

# ============================================
# SECTION 6: MODEL DEFINITION
# ============================================

print("="*70)
print("CREATING MODEL")
print("="*70 + "\n")

# Removed img_size to silence deprecation warning
model = SwinUNETR(
    img_size=CONFIG["model"]["img_size"], # Still passing it as it might be needed for internal init in some versions, but can be removed if strict 1.5+
    in_channels=CONFIG["model"]["in_channels"],
    out_channels=CONFIG["model"]["out_channels"],
    feature_size=CONFIG["model"]["feature_size"],
    use_checkpoint=CONFIG["model"]["use_checkpoint"],
).to(device)

# Using DiceCELoss with class weights to handle severe class imbalance
# Nodules are tiny compared to background, so we weight foreground heavily
try:
    ce_weights = torch.tensor([0.1, 0.9]).to(device)  # Heavy emphasis on foreground
    loss_function = DiceCELoss(
        to_onehot_y=True,
        softmax=True,
        lambda_dice=1.0,
        lambda_ce=1.0,
        ce_weight=ce_weights,
    )
    print("   Using weighted DiceCELoss (0.1:0.9 bg:fg)")
except TypeError:
    # Fallback for older MONAI versions without ce_weight
    loss_function = DiceCELoss(
        to_onehot_y=True,
        softmax=True,
        lambda_dice=1.0,
        lambda_ce=0.5,  # Reduce CE contribution to let Dice dominate
    )
    print("   Using DiceCELoss (no class weights - older MONAI)")

dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

# Post-processing transform for predictions only
# Labels will be handled manually since they're already binary masks
post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=CONFIG["training"]["learning_rate"],
    weight_decay=1e-5,
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=CONFIG["training"]["num_epochs"],
)

scaler = GradScaler(enabled=(device.type == "cuda"))

print("\n✅ Model, loss, optimizer ready")
print("="*70 + "\n")

# ============================================
# SECTION 7: TRAINING LOOP
# ============================================

history = {
    "train_loss": [],
    "val_loss": [],
    "val_dice": [],
    "learning_rate": [],
}

best_dice = 0.0
best_epoch = 0

def train_epoch(model, loader, optimizer, loss_fn, device, scaler):
    model.train()
    epoch_loss = 0
    step = 0
    for batch in loader:
        step += 1
        inputs, labels = batch["image"].to(device), batch["label"].to(device)
        optimizer.zero_grad()
        with autocast(enabled=(device.type == "cuda")):
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()
    return epoch_loss / step

def validate_epoch(model, loader, loss_fn, device, epoch_num=0):
    model.eval()
    epoch_loss = 0
    step = 0
    dice_metric.reset()
    
    # Track debug info
    total_pred_foreground = 0
    total_label_foreground = 0
    
    with torch.no_grad():
        for batch in loader:
            step += 1
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            with autocast(enabled=(device.type == "cuda")):
                outputs = sliding_window_inference(
                    inputs, CONFIG["model"]["img_size"], 4, model, overlap=0.5
                )
                loss = loss_fn(outputs, labels)
            epoch_loss += loss.item()
            
            # Decollate batch into individual samples
            outputs_list = decollate_batch(outputs)
            labels_list = decollate_batch(labels)
            
            # Post-process predictions: argmax + one-hot
            outputs_post = [post_pred(i) for i in outputs_list]
            
            # Post-process labels: manually convert binary mask to one-hot
            # Labels are [1, H, W, D] with values 0.0/1.0, we need [2, H, W, D] one-hot
            labels_post = []
            for label in labels_list:
                # label shape: [1, H, W, D] with float values 0.0 or 1.0
                foreground = label  # [1, H, W, D] - this IS the foreground channel
                background = 1.0 - label  # [1, H, W, D] - complement is background
                one_hot_label = torch.cat([background, foreground], dim=0)  # [2, H, W, D]
                labels_post.append(one_hot_label)
            
            # Track statistics for debugging
            for pred, lbl in zip(outputs_post, labels_post):
                total_pred_foreground += pred[1].sum().item()
                total_label_foreground += lbl[1].sum().item()
            
            dice_metric(y_pred=outputs_post, y=labels_post)
    
    mean_dice = dice_metric.aggregate().item()
    dice_metric.reset()
    
    # Print debug info every 10 epochs or first epoch
    if epoch_num == 0 or (epoch_num + 1) % 10 == 0:
        print(f"   [DEBUG] Total pred foreground voxels: {total_pred_foreground:.0f}")
        print(f"   [DEBUG] Total label foreground voxels: {total_label_foreground:.0f}")
    
    return epoch_loss / max(step, 1), mean_dice

print("Starting training...")

try:
    for epoch in range(CONFIG["training"]["num_epochs"]):
        print(f"Epoch {epoch+1}/{CONFIG['training']['num_epochs']}")
        
        train_loss = train_epoch(model, train_loader, optimizer, loss_function, device, scaler)
        history["train_loss"].append(train_loss)
        history["learning_rate"].append(optimizer.param_groups[0]["lr"])
        
        print(f"   Train Loss: {train_loss:.4f}")
        
        if (epoch + 1) % CONFIG["training"]["val_interval"] == 0:
            val_loss, val_dice = validate_epoch(model, val_loader, loss_function, device, epoch_num=epoch)
            history["val_loss"].append(val_loss)
            history["val_dice"].append(val_dice)
            
            print(f"   Val Loss:   {val_loss:.4f}")
            print(f"   Val Dice:   {val_dice:.4f}")
            
            if val_dice > best_dice:
                best_dice = val_dice
                best_epoch = epoch + 1
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "dice_score": best_dice,
                    "config": CONFIG,
                }, OUTPUT_DIR / "best_model.pth")
                print(f"   ✅ New best model saved!")
        
        scheduler.step()
        
        # Periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "history": history,
            }, OUTPUT_DIR / f"checkpoint_epoch{epoch+1}.pth")

except Exception as e:
    print(f"\n❌ Training failed: {e}")
    raise

finally:
    # Save final
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": CONFIG,
    }, OUTPUT_DIR / "final_model.pth")
    
    with open(OUTPUT_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

print("\n" + "="*70)
print(f"TRAINING COMPLETE. Best Dice: {best_dice:.4f} at epoch {best_epoch}")
print("="*70 + "\n")
