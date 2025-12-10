#!/usr/bin/env python3
"""
Model Evaluation Script
Evaluates trained model on test dataset and generates metrics report
"""

import argparse
import json
import logging
import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.modeling.evaluation import (
    evaluate_dataset,
    save_evaluation_report
)
from src.modeling.inference import batch_inference, load_trained_model
from src.preprocessing.transforms import get_inference_transforms
from src.preprocessing.dataloader import read_bids_dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/pipeline_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for evaluation report'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['json', 'csv'],
        default='json',
        help='Output format'
    )
    parser.add_argument(
        '--include-hausdorff',
        action='store_true',
        help='Include Hausdorff distance (slow)'
    )
    parser.add_argument(
        '--subjects',
        type=str,
        nargs='+',
        default=None,
        help='Specific subjects to evaluate'
    )
    
    args = parser.parse_args()
    
    # Load config
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    # Try to load model config
    model_config = config['model']
    checkpoint_dir = checkpoint_path.parent
    config_file = checkpoint_dir / 'training_config.yaml'
    
    if config_file.exists():
        with open(config_file, 'r') as f:
            train_config = yaml.safe_load(f)
        model_config = train_config.get('model', model_config)
    
    logger.info("Loading model...")
    model = load_trained_model(
        checkpoint_path=str(checkpoint_path),
        model_type=model_config['type'],
        in_channels=model_config['in_channels'],
        out_channels=model_config['out_channels'],
        img_size=tuple(model_config['img_size']),
        device=device
    )
    
    # Load test data
    bids_root = config['dataset']['bids_root']
    logger.info(f"Loading test data from {bids_root}...")
    
    data_dicts = read_bids_dataset(bids_root)
    
    # Filter to only include entries with labels
    data_dicts_with_labels = [d for d in data_dicts if 'label' in d]
    
    # Filter by subjects if specified
    if args.subjects:
        data_dicts_with_labels = [
            d for d in data_dicts_with_labels
            if d['subject_id'] in args.subjects
        ]
    
    # Load split information if available
    split_file = checkpoint_dir / 'data_splits.json'
    if split_file.exists():
        with open(split_file, 'r') as f:
            splits = json.load(f)
        test_subject_ids = splits.get('test', [])
        if test_subject_ids:
            data_dicts_with_labels = [
                d for d in data_dicts_with_labels
                if d['subject_id'] in test_subject_ids
            ]
            logger.info(f"Using test set with {len(data_dicts_with_labels)} subjects")
    
    if len(data_dicts_with_labels) == 0:
        logger.error("No test data with labels found")
        sys.exit(1)
    
    # Prepare inference
    image_paths = [d['image'] for d in data_dicts_with_labels]
    label_paths = [d['label'] for d in data_dicts_with_labels]
    
    # Create inference transforms
    preprocess_config = config['preprocessing']
    inference_transforms = get_inference_transforms(
        spacing=tuple(preprocess_config['spacing']),
        intensity_range=tuple(preprocess_config['intensity_range']),
        target_range=tuple(preprocess_config['target_range']),
        orientation=preprocess_config['orientation']
    )
    
    # Run inference
    logger.info("Running inference on test set...")
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
    
    # Save predictions temporarily for evaluation
    predictions_dir = Path(config['output']['predictions_dir'])
    predictions_dir.mkdir(parents=True, exist_ok=True)
    
    from src.modeling.inference import save_predictions
    pred_paths = save_predictions(
        predictions=results,
        output_dir=str(predictions_dir),
        reference_images=image_paths
    )
    
    # Filter out failed predictions
    valid_pred_paths = []
    valid_label_paths = []
    for i, result in enumerate(results):
        if "error" not in result:
            valid_pred_paths.append(pred_paths[i])
            valid_label_paths.append(label_paths[i])
    
    if len(valid_pred_paths) == 0:
        logger.error("No valid predictions generated")
        sys.exit(1)
    
    # Evaluate
    logger.info("Computing evaluation metrics...")
    
    # Get spacing from first image
    import nibabel as nib
    first_img = nib.load(label_paths[0])
    spacing = first_img.header.get_zooms()[:3]
    
    evaluation_results = evaluate_dataset(
        predictions=valid_pred_paths,
        ground_truths=valid_label_paths,
        num_classes=model_config['out_channels'],
        include_hausdorff=args.include_hausdorff,
        spacing=spacing
    )
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("Evaluation Summary")
    logger.info("="*80)
    logger.info(f"Number of samples: {evaluation_results['num_samples']}")
    
    aggregate = evaluation_results['aggregate']
    if 'overall_mean_dice' in aggregate:
        logger.info(f"Overall Mean Dice: {aggregate['overall_mean_dice']:.4f} ± {aggregate['overall_std_dice']:.4f}")
    
    # Print per-class metrics
    for key, value in aggregate.items():
        if 'dice_mean' in key:
            class_name = key.replace('_dice_mean', '')
            std_key = f"{class_name}_dice_std"
            std_value = aggregate.get(std_key, 0.0)
            logger.info(f"{class_name} Dice: {value:.4f} ± {std_value:.4f}")
    
    # Save report
    if args.output:
        output_path = args.output
    else:
        output_path = predictions_dir / f'evaluation_report.{args.format}'
    
    save_evaluation_report(
        evaluation_results,
        str(output_path),
        format=args.format
    )
    
    logger.info(f"\nEvaluation complete! Report saved to {output_path}")


if __name__ == '__main__':
    main()


