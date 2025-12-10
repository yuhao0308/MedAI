#!/usr/bin/env python3
"""
Training Script for Medical Imaging Segmentation
Loads config, prepares data, and trains a segmentation model
"""

import argparse
import yaml
import torch
import logging
from pathlib import Path
import json
from datetime import datetime
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.modeling.trainer import create_model, train_model
from src.preprocessing.dataloader import (
    read_bids_dataset,
    create_dataset,
    create_dataloader,
    split_dataset
)
from src.preprocessing.transforms import get_train_transforms, get_val_transforms

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_device() -> torch.device:
    """Detect and setup compute device"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
    else:
        device = torch.device("cpu")
        logger.warning("CUDA not available, using CPU (training will be slow)")
    return device


def prepare_data(config: dict):
    """Prepare datasets and dataloaders"""
    bids_root = config['dataset']['bids_root']
    
    logger.info(f"Reading BIDS dataset from {bids_root}")
    data_dicts = read_bids_dataset(bids_root)
    
    if len(data_dicts) == 0:
        raise ValueError(f"No data found in {bids_root}. Please ensure BIDS dataset is properly organized.")
    
    # Filter to only include entries with labels (for training)
    data_dicts_with_labels = [d for d in data_dicts if 'label' in d]
    logger.info(f"Found {len(data_dicts_with_labels)} image-label pairs for training")
    
    if len(data_dicts_with_labels) == 0:
        raise ValueError("No image-label pairs found. Cannot train without ground truth labels.")
    
    # Split dataset
    split_config = config['training']['split']
    splits = split_dataset(
        data_dicts_with_labels,
        train_ratio=split_config['train_ratio'],
        val_ratio=split_config['val_ratio'],
        test_ratio=split_config['test_ratio'],
        random_seed=split_config['random_seed']
    )
    
    # Get transform parameters
    preprocess_config = config['preprocessing']
    train_config = config['training']
    
    # Create transforms
    train_transforms = get_train_transforms(
        spatial_size=tuple(train_config['spatial_size']),
        spacing=tuple(preprocess_config['spacing']),
        intensity_range=tuple(preprocess_config['intensity_range']),
        target_range=tuple(preprocess_config['target_range']),
        orientation=preprocess_config['orientation'],
        pos_neg_ratio=train_config['augmentation']['pos_neg_ratio'],
        num_samples=train_config['augmentation']['num_samples'],
        augmentation_prob=train_config['augmentation']['prob']
    )
    
    val_transforms = get_val_transforms(
        spacing=tuple(preprocess_config['spacing']),
        intensity_range=tuple(preprocess_config['intensity_range']),
        target_range=tuple(preprocess_config['target_range']),
        orientation=preprocess_config['orientation']
    )
    
    # Create datasets
    cache_type = train_config.get('cache_type', 'memory')
    cache_dir = train_config.get('cache_dir', None)
    
    train_dataset = create_dataset(
        splits['train'],
        transform=train_transforms,
        cache_dir=cache_dir,
        cache_type=cache_type,
        num_workers=train_config.get('num_workers', 4)
    )
    
    val_dataset = create_dataset(
        splits['val'],
        transform=val_transforms,
        cache_dir=cache_dir,
        cache_type=cache_type,
        num_workers=train_config.get('num_workers', 4)
    )
    
    # Create dataloaders
    train_loader = create_dataloader(
        train_dataset,
        batch_size=train_config['batch_size'],
        shuffle=True,
        num_workers=train_config.get('num_workers', 4)
    )
    
    val_loader = create_dataloader(
        val_dataset,
        batch_size=1,  # Validation typically uses batch_size=1
        shuffle=False,
        num_workers=train_config.get('num_workers', 4)
    )
    
    return train_loader, val_loader, splits


def main():
    parser = argparse.ArgumentParser(description='Train medical imaging segmentation model')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/pipeline_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Override output directory for models'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Override number of epochs'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Override learning rate'
    )
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    
    config = load_config(str(config_path))
    
    # Override config with command-line arguments
    if args.output_dir:
        config['output']['model_save_dir'] = args.output_dir
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.lr:
        config['training']['learning_rate'] = args.lr
    
    # Setup device
    device = setup_device()
    
    # Create output directory
    model_save_dir = Path(config['output']['model_save_dir'])
    model_save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config to output directory
    config_save_path = model_save_dir / 'training_config.yaml'
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"Saved training config to {config_save_path}")
    
    # Prepare data
    logger.info("Preparing datasets...")
    train_loader, val_loader, splits = prepare_data(config)
    
    # Create model
    model_config = config['model']
    logger.info(f"Creating {model_config['type']} model...")
    model = create_model(
        model_type=model_config['type'],
        in_channels=model_config['in_channels'],
        out_channels=model_config['out_channels'],
        img_size=tuple(model_config['img_size'])
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        logger.info(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        logger.info(f"Resuming from epoch {start_epoch}")
    
    # Train model
    logger.info("Starting training...")
    training_config = config['training']
    
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=training_config['num_epochs'],
        learning_rate=training_config['learning_rate'],
        device=device,
        save_dir=str(model_save_dir)
    )
    
    # Save training history
    history_path = model_save_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f"Saved training history to {history_path}")
    
    # Save split information
    split_info = {
        'train': [d['subject_id'] for d in splits['train']],
        'val': [d['subject_id'] for d in splits['val']],
        'test': [d['subject_id'] for d in splits['test']]
    }
    split_path = model_save_dir / 'data_splits.json'
    with open(split_path, 'w') as f:
        json.dump(split_info, f, indent=2)
    logger.info(f"Saved data splits to {split_path}")
    
    logger.info("Training complete!")
    logger.info(f"Best model saved to: {model_save_dir / 'best_model.pth'}")
    logger.info(f"Final validation Dice: {max(history['val_dice']):.4f}")


if __name__ == '__main__':
    main()


