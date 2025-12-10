#!/usr/bin/env python3
"""
End-to-End Pipeline Script
Orchestrates validation, training, and inference
"""

import argparse
import subprocess
import sys
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status"""
    logger.info(f"Running: {description}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False
        )
        logger.info(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {description} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        logger.error(f"✗ Command not found: {cmd[0]}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Run end-to-end medical imaging AI pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python scripts/run_pipeline.py full --config configs/pipeline_config.yaml
  
  # Validate only
  python scripts/run_pipeline.py validate --bids-root data/bids_dataset
  
  # Train only
  python scripts/run_pipeline.py train --config configs/pipeline_config.yaml
  
  # Inference only
  python scripts/run_pipeline.py infer --checkpoint models/best_model.pth
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Pipeline command')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate BIDS dataset')
    validate_parser.add_argument(
        '--bids-root',
        type=str,
        default='data/bids_dataset',
        help='Path to BIDS dataset root'
    )
    validate_parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save validation report JSON'
    )
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train model')
    train_parser.add_argument(
        '--config',
        type=str,
        default='configs/pipeline_config.yaml',
        help='Path to configuration file'
    )
    train_parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    train_parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Override number of epochs'
    )
    
    # Infer command
    infer_parser = subparsers.add_parser('infer', help='Run inference')
    infer_parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    infer_parser.add_argument(
        '--config',
        type=str,
        default='configs/pipeline_config.yaml',
        help='Path to configuration file'
    )
    infer_parser.add_argument(
        '--subjects',
        type=str,
        nargs='+',
        default=None,
        help='Specific subjects to process'
    )
    infer_parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Evaluate predictions against ground truth'
    )
    
    # Full pipeline command
    full_parser = subparsers.add_parser('full', help='Run full pipeline (validate + train + infer)')
    full_parser.add_argument(
        '--config',
        type=str,
        default='configs/pipeline_config.yaml',
        help='Path to configuration file'
    )
    full_parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip validation step'
    )
    full_parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip training step (use existing model)'
    )
    full_parser.add_argument(
        '--skip-inference',
        action='store_true',
        help='Skip inference step'
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    scripts_dir = Path(__file__).parent
    
    # Execute based on command
    if args.command == 'validate':
        cmd = [
            sys.executable,
            str(scripts_dir / 'validate_bids.py'),
            args.bids_root
        ]
        if args.output:
            cmd.extend(['--output', args.output])
        
        success = run_command(cmd, "BIDS Validation")
        sys.exit(0 if success else 1)
    
    elif args.command == 'train':
        cmd = [
            sys.executable,
            str(scripts_dir / 'train_model.py'),
            '--config', args.config
        ]
        if args.resume:
            cmd.extend(['--resume', args.resume])
        if args.epochs:
            cmd.extend(['--epochs', str(args.epochs)])
        
        success = run_command(cmd, "Model Training")
        sys.exit(0 if success else 1)
    
    elif args.command == 'infer':
        cmd = [
            sys.executable,
            str(scripts_dir / 'run_inference.py'),
            '--checkpoint', args.checkpoint,
            '--config', args.config
        ]
        if args.subjects:
            cmd.extend(['--subjects'] + args.subjects)
        if args.evaluate:
            cmd.append('--evaluate')
        
        success = run_command(cmd, "Inference")
        sys.exit(0 if success else 1)
    
    elif args.command == 'full':
        logger.info("="*80)
        logger.info("Running Full Pipeline")
        logger.info("="*80)
        
        # Import config to get paths
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        bids_root = config['dataset']['bids_root']
        model_save_dir = config['output']['model_save_dir']
        
        all_success = True
        
        # Step 1: Validation
        if not args.skip_validation:
            logger.info("\n" + "="*80)
            logger.info("STEP 1: Validation")
            logger.info("="*80)
            cmd = [
                sys.executable,
                str(scripts_dir / 'validate_bids.py'),
                bids_root,
                '--output', str(Path(model_save_dir) / 'validation_report.json')
            ]
            if not run_command(cmd, "BIDS Validation"):
                logger.error("Validation failed. Fix issues before continuing.")
                sys.exit(1)
        else:
            logger.info("Skipping validation step")
        
        # Step 2: Training
        if not args.skip_training:
            logger.info("\n" + "="*80)
            logger.info("STEP 2: Training")
            logger.info("="*80)
            cmd = [
                sys.executable,
                str(scripts_dir / 'train_model.py'),
                '--config', args.config
            ]
            if not run_command(cmd, "Model Training"):
                logger.error("Training failed.")
                all_success = False
        else:
            logger.info("Skipping training step")
            # Check if model exists
            checkpoint_path = Path(model_save_dir) / 'best_model.pth'
            if not checkpoint_path.exists():
                logger.error(f"Model checkpoint not found: {checkpoint_path}")
                logger.error("Cannot run inference without a trained model.")
                sys.exit(1)
        
        # Step 3: Inference
        if not args.skip_inference:
            logger.info("\n" + "="*80)
            logger.info("STEP 3: Inference")
            logger.info("="*80)
            checkpoint_path = Path(model_save_dir) / 'best_model.pth'
            if not checkpoint_path.exists():
                logger.error(f"Model checkpoint not found: {checkpoint_path}")
                all_success = False
            else:
                cmd = [
                    sys.executable,
                    str(scripts_dir / 'run_inference.py'),
                    '--checkpoint', str(checkpoint_path),
                    '--config', args.config,
                    '--evaluate'
                ]
                if not run_command(cmd, "Inference"):
                    all_success = False
        else:
            logger.info("Skipping inference step")
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("Pipeline Summary")
        logger.info("="*80)
        if all_success:
            logger.info("✓ Full pipeline completed successfully!")
        else:
            logger.error("✗ Pipeline completed with errors")
        
        sys.exit(0 if all_success else 1)


if __name__ == '__main__':
    main()


