"""
Uncertainty Quantification
2025 Standard: Monte Carlo Dropout for predictive uncertainty
"""

import torch
import torch.nn as nn
from monai.inferers import sliding_window_inference
from typing import Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


def monte_carlo_dropout_inference(
    model: nn.Module,
    image: torch.Tensor,
    num_passes: int = 20,
    roi_size: tuple = (96, 96, 96),
    sw_batch_size: int = 4,
    overlap: float = 0.25,
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Performs Monte Carlo Dropout inference for uncertainty quantification.
    
    The model must have Dropout layers. This function runs inference
    multiple times with dropout enabled to estimate prediction uncertainty.
    
    Args:
        model: Model with Dropout layers
        image: Input image tensor (1, C, H, W, D)
        num_passes: Number of inference passes
        roi_size: Patch size for sliding window inference
        sw_batch_size: Batch size for sliding window
        overlap: Overlap ratio between patches
        device: Device to run inference on
        
    Returns:
        Tuple of (mean_prediction, uncertainty_map)
        - mean_prediction: Mean of all predictions (softmax probabilities)
        - uncertainty_map: Variance of predictions (uncertainty measure)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.train()  # Enable dropout
    model = model.to(device)
    image = image.to(device)
    
    uncertainty_outputs = []
    
    logger.info(f"Running {num_passes} Monte Carlo Dropout passes...")
    
    for i in range(num_passes):
        with torch.no_grad():  # Still no_grad, we're not training
            pred = sliding_window_inference(
                inputs=image,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                predictor=model,
                overlap=overlap
            )
            
            # Store softmax probabilities
            pred_probs = torch.softmax(pred, dim=1)
            uncertainty_outputs.append(pred_probs)
        
        if (i + 1) % 5 == 0:
            logger.info(f"Completed {i+1}/{num_passes} passes")
    
    # Stack all predictions
    stacked_preds = torch.stack(uncertainty_outputs)  # (num_passes, B, C, H, W, D)
    
    # Calculate mean prediction (final prediction)
    mean_prediction_probs = torch.mean(stacked_preds, dim=0)
    
    # Calculate uncertainty as variance across passes
    # Higher variance = higher uncertainty
    variance_map = torch.var(stacked_preds, dim=0)
    
    # Uncertainty can be computed as:
    # 1. Variance of probabilities
    # 2. Entropy of mean prediction
    # 3. Mutual information between passes
    
    # For simplicity, we'll use variance
    # Sum variance across channels for total uncertainty
    uncertainty_map = torch.sum(variance_map, dim=1, keepdim=True)
    
    logger.info("Monte Carlo Dropout inference complete")
    
    return mean_prediction_probs, uncertainty_map


def get_final_segmentation(
    mean_prediction_probs: torch.Tensor
) -> torch.Tensor:
    """
    Converts mean prediction probabilities to final segmentation mask.
    
    Args:
        mean_prediction_probs: Mean prediction probabilities (B, C, H, W, D)
        
    Returns:
        Segmentation mask (B, 1, H, W, D) with class indices
    """
    final_segmentation = torch.argmax(mean_prediction_probs, dim=1, keepdim=True)
    return final_segmentation


def compute_uncertainty_metrics(
    uncertainty_map: torch.Tensor,
    threshold_percentile: float = 95.0
) -> dict:
    """
    Computes summary statistics for uncertainty map.
    
    Args:
        uncertainty_map: Uncertainty map tensor
        threshold_percentile: Percentile to use as high-uncertainty threshold
        
    Returns:
        Dictionary with uncertainty metrics
    """
    uncertainty_np = uncertainty_map.cpu().numpy()
    
    metrics = {
        "mean_uncertainty": float(np.mean(uncertainty_np)),
        "std_uncertainty": float(np.std(uncertainty_np)),
        "max_uncertainty": float(np.max(uncertainty_np)),
        "min_uncertainty": float(np.min(uncertainty_np)),
        "high_uncertainty_threshold": float(
            np.percentile(uncertainty_np, threshold_percentile)
        ),
        "high_uncertainty_voxels": int(
            np.sum(uncertainty_np > np.percentile(uncertainty_np, threshold_percentile))
        ),
        "total_voxels": int(uncertainty_np.size)
    }
    
    return metrics


