"""
MONAI Transform Pipeline
2025 Standard: Dictionary-based transforms for synchronized image/label processing
"""

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    RandCropByPosNegLabeld,
    SpatialPadd,
    RandAffined,
    Rand3DElasticd,
    ToTensord,
)
from typing import Tuple, Optional, Dict, Any
import torch
import numpy as np
from monai.transforms import Transform


class BinaryLabelNormalizationd(Transform):
    """
    Normalizes labels to binary (0/1) for binary segmentation.
    Converts any non-zero label value to 1, keeps 0 as 0.
    This handles labels that may be encoded as 0-255 (8-bit) or other ranges.
    """
    
    def __init__(self, keys: str):
        """
        Args:
            keys: Key(s) to apply transform to (e.g., "label")
        """
        super().__init__()
        self.keys = keys if isinstance(keys, (list, tuple)) else [keys]
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply binary normalization to labels.
        """
        d = dict(data)
        for key in self.keys:
            if key in d:
                label = d[key]
                # Handle numpy arrays
                if isinstance(label, np.ndarray):
                    d[key] = (label > 0).astype(np.float32)
                # Handle torch tensors
                elif isinstance(label, torch.Tensor):
                    d[key] = (label > 0).float()
                else:
                    # Try to convert
                    label_array = np.asarray(label)
                    d[key] = (label_array > 0).astype(np.float32)
        return d


def get_pre_transforms(
    spacing: Tuple[float, float, float] = (1.5, 1.5, 1.5),
    intensity_range: Tuple[float, float] = (-1000.0, 1000.0),
    target_range: Tuple[float, float] = (0.0, 1.0),
    orientation: str = "RAS"
) -> Compose:
    """
    Creates deterministic preprocessing transforms (for caching).
    
    These transforms are applied once and cached, as they don't include
    random augmentations.
    
    Args:
        spacing: Target voxel spacing in mm (x, y, z)
        intensity_range: Source intensity range (a_min, a_max)
        target_range: Target intensity range (b_min, b_max)
        orientation: Target anatomical orientation (default: "RAS")
        
    Returns:
        MONAI Compose transform pipeline
    """
    return Compose([
        # Load NIfTI files
        LoadImaged(keys=["image", "label"]),
        
        # Add channel dimension (required for PyTorch)
        EnsureChannelFirstd(keys=["image", "label"]),
        
        # Reorient to standard anatomical orientation
        # Use labels=None to use meta-tensor space information (new MONAI API)
        Orientationd(keys=["image", "label"], axcodes=orientation, labels=None),
        
        # Resample to uniform voxel spacing (critical for 3D CNNs)
        Spacingd(
            keys=["image", "label"],
            pixdim=spacing,
            mode=("bilinear", "nearest")  # Bilinear for image, nearest for label
        ),
        
        # Intensity normalization
        ScaleIntensityRanged(
            keys=["image"],
            a_min=intensity_range[0],
            a_max=intensity_range[1],
            b_min=target_range[0],
            b_max=target_range[1],
            clip=True
        ),
        
        # Remove background/zero-padding
        CropForegroundd(
            keys=["image", "label"],
            source_key="image"
        ),
        
        # Normalize labels to binary (0/1) for binary segmentation
        # Converts any non-zero label value to 1, keeps 0 as 0
        # This handles labels that may be encoded as 0-255 (8-bit) or other ranges
        BinaryLabelNormalizationd(keys="label"),
    ])


def get_train_transforms(
    spatial_size: Tuple[int, int, int] = (96, 96, 96),
    spacing: Tuple[float, float, float] = (1.5, 1.5, 1.5),
    intensity_range: Tuple[float, float] = (-1000.0, 1000.0),
    target_range: Tuple[float, float] = (0.0, 1.0),
    orientation: str = "RAS",
    pos_neg_ratio: int = 1,
    num_samples: int = 4,
    augmentation_prob: float = 0.1
) -> Compose:
    """
    Creates training transforms with data augmentation.
    
    Args:
        spatial_size: Patch size for training (x, y, z)
        spacing: Target voxel spacing
        intensity_range: Source intensity range
        target_range: Target intensity range
        orientation: Target anatomical orientation
        pos_neg_ratio: Positive to negative patch ratio
        num_samples: Number of patches per volume
        augmentation_prob: Probability of applying augmentations
        
    Returns:
        MONAI Compose transform pipeline for training
    """
    # Start with deterministic preprocessing
    pre_transforms = get_pre_transforms(
        spacing=spacing,
        intensity_range=intensity_range,
        target_range=target_range,
        orientation=orientation
    )
    
    # Add training-specific augmentations
    train_transforms = Compose([
        *pre_transforms.transforms,  # Include all pre_transforms
        
        # Patch-based sampling (required for 3D volumes that don't fit in GPU)
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=spatial_size,
            pos=pos_neg_ratio,
            neg=pos_neg_ratio,
            num_samples=num_samples
        ),
        
        # Pad if patch is smaller than spatial_size
        SpatialPadd(
            keys=["image", "label"],
            spatial_size=spatial_size
        ),
        
        # 3D Affine augmentation (rotation, scaling, translation)
        RandAffined(
            keys=["image", "label"],
            prob=augmentation_prob,
            rotate_range=(0, 0, 3.14),  # Rotation range in radians
            scale_range=(0.1, 0.1, 0.1),  # Scale range
            mode=("bilinear", "nearest")
        ),
        
        # 3D Elastic deformation (mimics tissue deformation)
        Rand3DElasticd(
            keys=["image", "label"],
            prob=augmentation_prob,
            sigma_range=(5, 8),
            magnitude_range=(100, 200),
            mode=("bilinear", "nearest")
        ),
        
        # Convert to PyTorch tensors
        ToTensord(keys=["image", "label"])
    ])
    
    return train_transforms


def get_val_transforms(
    spacing: Tuple[float, float, float] = (1.5, 1.5, 1.5),
    intensity_range: Tuple[float, float] = (-1000.0, 1000.0),
    target_range: Tuple[float, float] = (0.0, 1.0),
    orientation: str = "RAS"
) -> Compose:
    """
    Creates validation transforms (deterministic, no augmentation).
    
    Args:
        spacing: Target voxel spacing
        intensity_range: Source intensity range
        target_range: Target intensity range
        orientation: Target anatomical orientation
        
    Returns:
        MONAI Compose transform pipeline for validation
    """
    val_transforms = Compose([
        *get_pre_transforms(
            spacing=spacing,
            intensity_range=intensity_range,
            target_range=target_range,
            orientation=orientation
        ).transforms,
        
        # Convert to PyTorch tensors
        ToTensord(keys=["image", "label"])
    ])
    
    return val_transforms


def get_inference_transforms(
    spacing: Tuple[float, float, float] = (1.5, 1.5, 1.5),
    intensity_range: Tuple[float, float] = (-1000.0, 1000.0),
    target_range: Tuple[float, float] = (0.0, 1.0),
    orientation: str = "RAS"
) -> Compose:
    """
    Creates inference transforms (same as validation, but only for images).
    
    Args:
        spacing: Target voxel spacing
        intensity_range: Source intensity range
        target_range: Target intensity range
        orientation: Target anatomical orientation
        
    Returns:
        MONAI Compose transform pipeline for inference
    """
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes=orientation, labels=None),
        Spacingd(
            keys=["image"],
            pixdim=spacing,
            mode="bilinear"
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=intensity_range[0],
            a_max=intensity_range[1],
            b_min=target_range[0],
            b_max=target_range[1],
            clip=True
        ),
        CropForegroundd(keys=["image"], source_key="image"),
        ToTensord(keys=["image"])
    ])

