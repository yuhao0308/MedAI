#!/usr/bin/env python3
"""
Inference Script for Medical Imaging Segmentation
Loads trained model and runs inference on test data
"""

import argparse
import yaml
import torch
import logging
import json
from pathlib import Path
from typing import Optional, List
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.modeling.inference import (
    load_trained_model,
    batch_inference,
    save_predictions,
    compute_basic_metrics
)
from src.preprocessing.transforms import get_inference_transforms
from src.preprocessing.dataloader import read_bids_dataset

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
    else:
        device = torch.device("cpu")
        logger.warning("CUDA not available, using CPU (inference will be slow)")
    return device


def load_test_subjects(config: dict, test_subjects: Optional[List[str]] = None):
    """Load test subjects from BIDS dataset"""
    bids_root = config['dataset']['bids_root']
    
    # Read all data
    data_dicts = read_bids_dataset(bids_root)
    
    # Filter to only images (for inference, labels optional)
    image_dicts = [d for d in data_dicts if 'image' in d]
    
    # Filter by subject if specified
    if test_subjects:
        image_dicts = [d for d in image_dicts if d['subject_id'] in test_subjects]
    
    # Load split information if available
    model_dir = Path(config['output']['model_save_dir'])
    split_file = model_dir / 'data_splits.json'
    
    if split_file.exists():
        with open(split_file, 'r') as f:
            splits = json.load(f)
        # Use test set if available
        test_subject_ids = splits.get('test', [])
        if test_subject_ids:
            image_dicts = [d for d in image_dicts if d['subject_id'] in test_subject_ids]
            logger.info(f"Using test set with {len(image_dicts)} subjects")
    
    if len(image_dicts) == 0:
        raise ValueError("No test subjects found")
    
    image_paths = [d['image'] for d in image_dicts]
    return image_paths, image_dicts


def main():
    parser = argparse.ArgumentParser(description='Run inference on medical imaging data')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/pipeline_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint (.pth file)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for predictions (overrides config)'
    )
    parser.add_argument(
        '--subjects',
        type=str,
        nargs='+',
        default=None,
        help='Specific subjects to process (e.g., sub-001 sub-002)'
    )
    parser.add_argument(
        '--uncertainty',
        action='store_true',
        help='Enable uncertainty quantification (Monte Carlo Dropout)'
    )
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Evaluate predictions against ground truth (if available)'
    )
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    
    config = load_config(str(config_path))
    
    # Override output directory if specified
    if args.output_dir:
        config['output']['predictions_dir'] = args.output_dir
    
    # Setup device
    device = setup_device()
    
    # Load model
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    # Try to load model config from checkpoint directory
    model_config = config['model']
    checkpoint_dir = checkpoint_path.parent
    config_file = checkpoint_dir / 'training_config.yaml'
    
    if config_file.exists():
        train_config = load_config(str(config_file))
        model_config = train_config.get('model', model_config)
        logger.info(f"Loaded model config from {config_file}")
    
    logger.info(f"Loading model from {checkpoint_path}")
    model = load_trained_model(
        checkpoint_path=str(checkpoint_path),
        model_type=model_config['type'],
        in_channels=model_config['in_channels'],
        out_channels=model_config['out_channels'],
        img_size=tuple(model_config['img_size']),
        device=device
    )
    
    # Enable dropout for uncertainty if requested
    if args.uncertainty:
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.train()  # Keep dropout active during inference
        logger.info("Uncertainty quantification enabled (Monte Carlo Dropout)")
    
    # Load test data
    logger.info("Loading test data...")
    image_paths, image_dicts = load_test_subjects(config, args.subjects)
    logger.info(f"Found {len(image_paths)} images to process")
    
    # Create inference transforms
    preprocess_config = config['preprocessing']
    inference_transforms = get_inference_transforms(
        spacing=tuple(preprocess_config['spacing']),
        intensity_range=tuple(preprocess_config['intensity_range']),
        target_range=tuple(preprocess_config['target_range']),
        orientation=preprocess_config['orientation']
    )
    
    # Run inference
    logger.info("Running inference...")
    inference_config = config['inference']
    results = batch_inference(
        model=model,
        image_paths=image_paths,
        transform=inference_transforms,
        roi_size=tuple(inference_config['roi_size']),
        sw_batch_size=inference_config['sw_batch_size'],
        overlap=inference_config['overlap'],
        device=device
    )
    
    # Save predictions
    output_dir = config['output']['predictions_dir']
    logger.info(f"Saving predictions to {output_dir}")
    saved_paths = save_predictions(
        predictions=results,
        output_dir=output_dir,
        reference_images=image_paths
    )
    
    logger.info(f"Saved {len(saved_paths)} predictions")
    
    # Evaluate if requested and ground truth available
    if args.evaluate:
        logger.info("Evaluating predictions...")
        evaluation_results = []
        
        for i, result in enumerate(results):
            if "error" in result:
                continue
            
            image_dict = image_dicts[i]
            if 'label' not in image_dict:
                logger.warning(f"No ground truth for {image_dict['subject_id']}, skipping evaluation")
                continue
            
            # Load ground truth
            import nibabel as nib
            gt_img = nib.load(image_dict['label'])
            gt_data = gt_img.get_fdata().astype(np.uint8)
            
            # Compute metrics
            pred_data = result['prediction']
            metrics = compute_basic_metrics(pred_data, gt_data)
            
            evaluation_results.append({
                "subject_id": image_dict['subject_id'],
                "image_path": result['image_path'],
                "metrics": metrics
            })
            
            logger.info(
                f"{image_dict['subject_id']}: Dice={metrics['dice']:.4f}, "
                f"IoU={metrics['iou']:.4f}"
            )
        
        # Save evaluation results
        eval_output = Path(output_dir) / 'evaluation_results.json'
        with open(eval_output, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        logger.info(f"Saved evaluation results to {eval_output}")
        
        # Compute average metrics
        if evaluation_results:
            avg_dice = sum(r['metrics']['dice'] for r in evaluation_results) / len(evaluation_results)
            avg_iou = sum(r['metrics']['iou'] for r in evaluation_results) / len(evaluation_results)
            logger.info(f"Average Dice: {avg_dice:.4f}, Average IoU: {avg_iou:.4f}")
    
    logger.info("Inference complete!")


if __name__ == '__main__':
    import numpy as np
    main()

