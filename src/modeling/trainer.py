"""
Custom Training Pipeline
Alternative to Auto3DSeg for manual training control
"""

import torch
import torch.nn as nn
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR, SegResNet
from monai.data import DataLoader
from typing import Dict, Optional, Callable
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


def create_model(
    model_type: str = "SwinUNETR",
    in_channels: int = 1,
    out_channels: int = 2,
    img_size: tuple = (96, 96, 96),
    feature_size: int = 48,
    use_checkpoint: bool = True,
    **kwargs
) -> nn.Module:
    """
    Creates a 3D segmentation model.
    
    Args:
        model_type: "SwinUNETR" or "SegResNet"
        in_channels: Number of input channels
        out_channels: Number of output classes
        img_size: Input image size (used for SegResNet, kept for config compatibility)
        feature_size: Feature size for SwinUNETR (default 48)
        use_checkpoint: Use gradient checkpointing to save memory (default True)
        **kwargs: Additional model parameters
        
    Returns:
        PyTorch model
    """
    if model_type == "SwinUNETR":
        # SwinUNETR in MONAI 1.3+ doesn't require img_size - it works with any input size
        # The img_size from config is used for sliding window inference patch size
        model = SwinUNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            use_checkpoint=use_checkpoint,
            **kwargs
        )
    elif model_type == "SegResNet":
        model = SegResNet(
            input_image_size=img_size,
            init_filters=16,
            in_channels=in_channels,
            out_channels=out_channels,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    logger.info(f"Created {model_type} model")
    return model


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int
) -> float:
    """
    Trains the model for one epoch.
    
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    logger.info(f"Starting training epoch {epoch+1}, total batches: {len(dataloader)}")
    
    for batch_idx, batch_data in enumerate(dataloader):
        inputs = batch_data["image"].to(device)
        labels = batch_data["label"].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Log progress more frequently for debugging
        if batch_idx % 5 == 0 or batch_idx == 0:
            logger.info(
                f"Epoch {epoch+1}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}"
            )
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    dice_metric: DiceMetric,
    device: torch.device
) -> Dict[str, float]:
    """
    Validates the model.
    
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_data in dataloader:
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)
            
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            
            # Calculate Dice score
            dice_metric(y_pred=outputs, y=labels)
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    dice_scores = dice_metric.aggregate().item()
    dice_metric.reset()
    
    return {
        "loss": avg_loss,
        "dice": dice_scores
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 100,
    learning_rate: float = 1e-4,
    device: Optional[torch.device] = None,
    save_dir: Optional[str] = None
) -> Dict:
    """
    Full training loop.
    
    Returns:
        Training history dictionary
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    
    best_dice = 0.0
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_dice": []
    }
    
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch
        )
        
        # Validate
        val_metrics = validate(model, val_loader, loss_fn, dice_metric, device)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_dice"].append(val_metrics["dice"])
        
        logger.info(
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"Val Dice: {val_metrics['dice']:.4f}"
        )
        
        # Save best model
        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            if save_dir:
                save_path = Path(save_dir) / "best_model.pth"
                torch.save(model.state_dict(), save_path)
                logger.info(f"Saved best model (Dice: {best_dice:.4f})")
    
    return history


def inference_sliding_window(
    model: nn.Module,
    image: torch.Tensor,
    roi_size: tuple = (96, 96, 96),
    sw_batch_size: int = 4,
    overlap: float = 0.25,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Performs sliding window inference on a full 3D volume.
    
    Args:
        model: Trained model
        image: Input image tensor (1, C, H, W, D)
        roi_size: Patch size for inference
        sw_batch_size: Batch size for sliding window
        overlap: Overlap ratio between patches
        device: Device to run inference on
        
    Returns:
        Prediction tensor
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

