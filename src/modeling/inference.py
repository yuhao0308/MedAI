"""
Inference Pipeline for Medical Imaging Segmentation
Loads trained models and runs inference on new data
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
import nibabel as nib
import numpy as np
import json
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
    Orientationd, ScaleIntensityRanged, CropForegroundd,
    DivisiblePadd, ToTensord, Invertd, SaveImaged, AsDiscreted
)
from monai.data import decollate_batch
from scipy import ndimage

logger = logging.getLogger(__name__)


def load_model_config(config_path: str) -> Dict[str, Any]:
    """
    Loads model configuration from a JSON file.
    
    Args:
        config_path: Path to config.json file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def load_trained_model(
    checkpoint_path: str,
    model_type: str = "SwinUNETR",
    in_channels: int = 1,
    out_channels: int = 2,
    img_size: tuple = (96, 96, 96),
    feature_size: int = 48,
    use_checkpoint: bool = True,
    device: Optional[torch.device] = None
) -> nn.Module:
    """
    Loads a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint (.pth file)
        model_type: Type of model ("SwinUNETR" or "SegResNet")
        in_channels: Number of input channels
        out_channels: Number of output classes
        img_size: Input image size (for model initialization)
        feature_size: Feature size for SwinUNETR (default 48)
        use_checkpoint: Use gradient checkpointing (default True)
        device: Device to load model on
        
    Returns:
        Loaded model in eval mode
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Import here to avoid circular imports
    from src.modeling.trainer import create_model
    
    # Convert img_size to tuple if it's a list
    if isinstance(img_size, list):
        img_size = tuple(img_size)
    
    # Create model architecture
    model = create_model(
        model_type=model_type,
        in_channels=in_channels,
        out_channels=out_channels,
        img_size=img_size,
        feature_size=feature_size,
        use_checkpoint=use_checkpoint
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # Assume entire checkpoint is state_dict
            model.load_state_dict(checkpoint)
    else:
        # Assume checkpoint is state_dict directly
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    logger.info(f"Loaded model from {checkpoint_path}")
    return model


def load_model_from_config(
    model_dir: str,
    device: Optional[torch.device] = None
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Loads a trained model using its config.json file.
    
    Args:
        model_dir: Path to model directory containing config.json and best_model.pth
        device: Device to load model on
        
    Returns:
        Tuple of (loaded model, config dictionary)
    """
    model_dir = Path(model_dir)
    config_path = model_dir / "config.json"
    checkpoint_path = model_dir / "best_model.pth"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    # Load config
    config = load_model_config(str(config_path))
    
    # Extract model parameters from config
    model_config = config.get("model", {})
    model_type = model_config.get("type", "SwinUNETR")
    in_channels = model_config.get("in_channels", 1)
    out_channels = model_config.get("out_channels", 2)
    img_size = model_config.get("img_size", [96, 96, 96])
    feature_size = model_config.get("feature_size", 48)
    use_checkpoint = model_config.get("use_checkpoint", True)
    
    # Load model
    model = load_trained_model(
        checkpoint_path=str(checkpoint_path),
        model_type=model_type,
        in_channels=in_channels,
        out_channels=out_channels,
        img_size=img_size,
        feature_size=feature_size,
        use_checkpoint=use_checkpoint,
        device=device
    )
    
    return model, config


def predict_volume(
    model: nn.Module,
    image: torch.Tensor,
    roi_size: Tuple[int, int, int] = (96, 96, 96),
    sw_batch_size: int = 4,
    overlap: float = 0.25,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Runs inference on a single 3D volume using sliding window.
    
    Args:
        model: Trained model
        image: Input image tensor (1, C, H, W, D)
        roi_size: Patch size for sliding window
        sw_batch_size: Batch size for sliding window
        overlap: Overlap ratio between patches
        device: Device to run inference on
        
    Returns:
        Prediction tensor (1, C, H, W, D)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    model = model.to(device)
    image = image.to(device)
    
    with torch.no_grad():
        pred = sliding_window_inference(
            inputs=image,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=model,
            overlap=overlap
        )
    
    return pred


def batch_inference(
    model: nn.Module,
    image_paths: List[str],
    transform: Compose,
    roi_size: Tuple[int, int, int] = (96, 96, 96),
    sw_batch_size: int = 4,
    overlap: float = 0.25,
    device: Optional[torch.device] = None
) -> List[Dict[str, Any]]:
    """
    Runs inference on multiple volumes.
    
    Args:
        model: Trained model
        image_paths: List of paths to NIfTI image files
        transform: MONAI transform pipeline for preprocessing
        roi_size: Patch size for sliding window
        sw_batch_size: Batch size for sliding window
        overlap: Overlap ratio between patches
        device: Device to run inference on
        
    Returns:
        List of dictionaries with 'image_path', 'prediction', and 'metadata'
    """
    results = []
    
    for image_path in image_paths:
        logger.info(f"Processing {Path(image_path).name}...")
        
        try:
            # Load and preprocess image
            data_dict = {"image": image_path}
            transformed = transform(data_dict)
            image_tensor = transformed["image"].unsqueeze(0)  # Add batch dimension
            
            # Run inference
            pred_tensor = predict_volume(
                model=model,
                image=image_tensor,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                overlap=overlap,
                device=device
            )
            
            # Convert to numpy
            pred_np = pred_tensor.cpu().numpy()[0]  # Remove batch dimension
            
            # Get argmax for class predictions
            if pred_np.shape[0] > 1:  # Multi-class
                pred_seg = np.argmax(pred_np, axis=0).astype(np.uint8)
            else:  # Binary
                pred_seg = (pred_np[0] > 0.5).astype(np.uint8)
            
            results.append({
                "image_path": image_path,
                "prediction": pred_seg,
                "prediction_tensor": pred_np,
                "metadata": {
                    "shape": pred_seg.shape,
                    "dtype": str(pred_seg.dtype),
                    "unique_labels": np.unique(pred_seg).tolist()
                }
            })
            
        except Exception as e:
            logger.error(f"Failed to process {image_path}: {e}")
            results.append({
                "image_path": image_path,
                "error": str(e)
            })
    
    return results


def save_predictions(
    predictions: List[Dict[str, any]],
    output_dir: str,
    reference_images: Optional[List[str]] = None
) -> List[str]:
    """
    Saves predictions as NIfTI files in BIDS derivatives format.
    
    Args:
        predictions: List of prediction dictionaries from batch_inference
        output_dir: Output directory for predictions
        reference_images: Optional list of reference image paths for header copying
        
    Returns:
        List of saved prediction file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    saved_paths = []
    
    for i, pred_dict in enumerate(predictions):
        if "error" in pred_dict:
            continue
        
        image_path = Path(pred_dict["image_path"])
        pred_seg = pred_dict["prediction"]
        
        # Load reference image for header
        if reference_images and i < len(reference_images):
            ref_img = nib.load(reference_images[i])
            affine = ref_img.affine
            header = ref_img.header.copy()
        else:
            # Try to load original image
            try:
                ref_img = nib.load(str(image_path))
                affine = ref_img.affine
                header = ref_img.header.copy()
            except:
                # Create default header
                affine = np.eye(4)
                header = nib.Nifti1Header()
        
        # Create NIfTI image
        pred_img = nib.Nifti1Image(pred_seg, affine, header)
        
        # Generate output filename
        subject_id = image_path.parent.parent.name  # Extract sub-XXX
        image_name = image_path.stem.replace(".nii", "")
        output_filename = f"{subject_id}_{image_name}_pred.nii.gz"
        output_path_full = output_path / output_filename
        
        # Save
        nib.save(pred_img, str(output_path_full))
        saved_paths.append(str(output_path_full))
        
        logger.info(f"Saved prediction to {output_path_full}")
    
    return saved_paths


def compute_basic_metrics(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    num_classes: Optional[int] = None
) -> Dict[str, float]:
    """
    Computes basic segmentation metrics (Dice, IoU).
    
    Args:
        prediction: Prediction mask
        ground_truth: Ground truth mask
        num_classes: Number of classes (if None, inferred from data)
        
    Returns:
        Dictionary with metrics
    """
    if num_classes is None:
        num_classes = max(int(prediction.max()), int(ground_truth.max())) + 1
    
    metrics = {}
    
    # Overall metrics
    pred_flat = prediction.flatten()
    gt_flat = ground_truth.flatten()
    
    # Dice coefficient
    intersection = np.sum(pred_flat == gt_flat)
    union = len(pred_flat)
    dice = 2.0 * intersection / (np.sum(pred_flat > 0) + np.sum(gt_flat > 0) + 1e-8)
    metrics["dice"] = float(dice)
    
    # IoU (Jaccard Index)
    intersection = np.sum((pred_flat > 0) & (gt_flat > 0))
    union = np.sum((pred_flat > 0) | (gt_flat > 0))
    iou = intersection / (union + 1e-8)
    metrics["iou"] = float(iou)
    
    # Per-class metrics
    class_metrics = {}
    for c in range(1, num_classes):  # Skip background (class 0)
        pred_c = (prediction == c).astype(np.uint8)
        gt_c = (ground_truth == c).astype(np.uint8)
        
        intersection_c = np.sum(pred_c & gt_c)
        union_c = np.sum(pred_c | gt_c)
        dice_c = 2.0 * intersection_c / (np.sum(pred_c) + np.sum(gt_c) + 1e-8)
        iou_c = intersection_c / (union_c + 1e-8)
        
        class_metrics[f"class_{c}"] = {
            "dice": float(dice_c),
            "iou": float(iou_c)
        }
    
    metrics["per_class"] = class_metrics
    
    return metrics


def get_invertible_inference_transforms(config: Dict) -> Tuple[Compose, Compose]:
    """
    Build invertible inference transforms from model config.
    Returns both forward transforms and post-processing transforms that can invert.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Tuple of (forward_transforms, post_transforms_with_invert)
    """
    preproc = config.get("preprocessing", {})
    
    spacing = preproc.get("spacing", [1.5, 1.5, 1.5])
    orientation = preproc.get("orientation", "RAS")
    intensity_range = preproc.get("intensity_range", [-1000.0, 400.0])
    target_range = preproc.get("target_range", [0.0, 1.0])
    divisible_k = preproc.get("divisible_k", 32)
    
    # Forward transforms - these track metadata for inversion
    forward_transforms = Compose([
        LoadImaged(keys=["image"], image_only=False),
        EnsureChannelFirstd(keys=["image"]),
        Spacingd(keys=["image"], pixdim=spacing, mode="bilinear"),
        Orientationd(keys=["image"], axcodes=orientation, labels=None),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=intensity_range[0],
            a_max=intensity_range[1],
            b_min=target_range[0],
            b_max=target_range[1],
            clip=True
        ),
        CropForegroundd(keys=["image"], source_key="image", margin=8, allow_smaller=True),
        DivisiblePadd(keys=["image"], k=divisible_k),
        ToTensord(keys=["image"])
    ])
    
    # Post transforms - includes inversion to original space
    post_transforms = Compose([
        Invertd(
            keys=["pred"],
            transform=forward_transforms,
            orig_keys=["image"],
            meta_keys=["pred_meta_dict"],
            orig_meta_keys=["image_meta_dict"],
            meta_key_postfix="meta_dict",
            nearest_interp=True,  # Use nearest for segmentation masks
            to_tensor=True
        ),
        AsDiscreted(keys=["pred"], argmax=True, to_onehot=None)
    ])
    
    return forward_transforms, post_transforms


def predict_with_inversion(
    model: nn.Module,
    image_path: str,
    config: Dict,
    roi_size: Tuple[int, int, int] = (96, 96, 96),
    sw_batch_size: int = 2,
    overlap: float = 0.5,
    device: Optional[torch.device] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Run inference with proper spatial alignment - returns prediction in original image space.
    
    Args:
        model: Trained model
        image_path: Path to input NIfTI image
        config: Model configuration dictionary
        roi_size: Patch size for sliding window
        sw_batch_size: Batch size for sliding window
        overlap: Overlap ratio between patches
        device: Device to run inference on
        
    Returns:
        Tuple of (prediction_array, metadata_dict)
        - prediction_array: Segmentation in original image space
        - metadata_dict: Contains original affine, shape, spacing, etc.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load original image to get metadata
    original_img = nib.load(image_path)
    original_affine = original_img.affine
    original_header = original_img.header
    original_shape = original_img.shape
    original_spacing = original_img.header.get_zooms()[:3]
    
    metadata = {
        "original_affine": original_affine,
        "original_header": original_header,
        "original_shape": original_shape,
        "original_spacing": original_spacing,
        "image_path": str(image_path)
    }
    
    # Get transforms
    forward_transforms, post_transforms = get_invertible_inference_transforms(config)
    
    # Preprocess
    data_dict = {"image": str(image_path)}
    transformed = forward_transforms(data_dict)
    
    # Store preprocessed shape for reference
    metadata["preprocessed_shape"] = tuple(transformed["image"].shape)
    
    # Run inference
    image_tensor = transformed["image"].unsqueeze(0).to(device)  # Add batch dimension
    model.eval()
    model = model.to(device)
    
    with torch.no_grad():
        pred_tensor = sliding_window_inference(
            inputs=image_tensor,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=model,
            overlap=overlap
        )
    
    # Process prediction
    pred_np = pred_tensor.cpu().numpy()[0]  # Remove batch dimension
    
    # Apply softmax and argmax for multi-class
    if pred_np.shape[0] > 1:
        pred_softmax = np.exp(pred_np) / np.sum(np.exp(pred_np), axis=0, keepdims=True)
        pred_seg = np.argmax(pred_softmax, axis=0).astype(np.uint8)
        confidence_map = np.max(pred_softmax, axis=0)
    else:
        pred_seg = (pred_np[0] > 0.5).astype(np.uint8)
        confidence_map = pred_np[0]
    
    # Resample prediction back to original space
    pred_resampled = resample_to_original(
        prediction=pred_seg,
        original_shape=original_shape,
        preprocessed_shape=metadata["preprocessed_shape"][1:],  # Remove channel dim
        mode='nearest'
    )
    
    metadata["confidence_map"] = confidence_map
    metadata["pred_preprocessed_shape"] = pred_seg.shape
    
    return pred_resampled, metadata


def resample_to_original(
    prediction: np.ndarray,
    original_shape: Tuple[int, ...],
    preprocessed_shape: Tuple[int, ...],
    mode: str = 'nearest'
) -> np.ndarray:
    """
    Resample prediction back to original image dimensions.
    
    Args:
        prediction: Prediction array in preprocessed space
        original_shape: Original image shape (3D)
        preprocessed_shape: Preprocessed image shape (3D)
        mode: Interpolation mode ('nearest' for segmentation)
        
    Returns:
        Resampled prediction array matching original shape
    """
    # Calculate zoom factors
    zoom_factors = tuple(o / p for o, p in zip(original_shape[:3], preprocessed_shape[:3]))
    
    # Resample using scipy's zoom
    if mode == 'nearest':
        order = 0
    else:
        order = 1
    
    resampled = ndimage.zoom(prediction, zoom_factors, order=order)
    
    # Ensure exact shape match (handle rounding differences)
    final = np.zeros(original_shape[:3], dtype=prediction.dtype)
    min_shape = tuple(min(a, b) for a, b in zip(resampled.shape, original_shape[:3]))
    final[:min_shape[0], :min_shape[1], :min_shape[2]] = \
        resampled[:min_shape[0], :min_shape[1], :min_shape[2]]
    
    return final


def save_prediction_with_metadata(
    prediction: np.ndarray,
    output_path: str,
    original_affine: np.ndarray,
    original_header: Optional[nib.Nifti1Header] = None
) -> str:
    """
    Save prediction as NIfTI with proper spatial metadata.
    
    Args:
        prediction: Prediction array
        output_path: Output file path
        original_affine: Original image affine matrix
        original_header: Optional original NIfTI header
        
    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create NIfTI image with original spatial information
    if original_header is not None:
        # Copy header but update data type
        new_header = original_header.copy()
        new_header.set_data_dtype(np.uint8)
    else:
        new_header = nib.Nifti1Header()
        new_header.set_data_dtype(np.uint8)
    
    pred_img = nib.Nifti1Image(
        prediction.astype(np.uint8),
        affine=original_affine,
        header=new_header
    )
    
    nib.save(pred_img, str(output_path))
    logger.info(f"Saved prediction with proper spatial alignment to {output_path}")
    
    return str(output_path)

