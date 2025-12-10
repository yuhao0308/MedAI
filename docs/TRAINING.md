# Training Guide

Complete guide for training medical imaging segmentation models.

## Prerequisites

1. **Processed BIDS Dataset**: Your data should be in BIDS format with images and labels
2. **GPU Recommended**: Training on CPU is possible but very slow
3. **Sufficient RAM**: At least 16GB recommended for caching

## Quick Start

```bash
python scripts/train_model.py --config configs/pipeline_config.yaml
```

## Configuration

Edit `configs/pipeline_config.yaml` to customize training:

### Key Parameters

**Training Settings:**
- `training.num_epochs`: Number of training epochs (default: 100)
- `training.batch_size`: Batch size (default: 1, increase if you have GPU memory)
- `training.learning_rate`: Learning rate (default: 0.0001)
- `training.spatial_size`: Patch size for training [96, 96, 96]

**Data Split:**
- `training.split.train_ratio`: Training set proportion (default: 0.7)
- `training.split.val_ratio`: Validation set proportion (default: 0.15)
- `training.split.test_ratio`: Test set proportion (default: 0.15)

**Model:**
- `model.type`: "SwinUNETR" or "SegResNet"
- `model.out_channels`: Number of classes (background + foreground classes)

## Training Process

### 1. Validate Data First

Always validate your dataset before training:

```bash
python scripts/validate_bids.py data/bids_dataset
```

### 2. Start Training

```bash
python scripts/train_model.py --config configs/pipeline_config.yaml
```

The script will:
- Load and validate BIDS dataset
- Create train/val/test splits
- Initialize model
- Train for specified epochs
- Save best model checkpoint
- Log training history

### 3. Monitor Training

Training progress is logged to console. Checkpoints are saved to:
- `models/best_model.pth` - Best model (highest validation Dice)
- `models/training_history.json` - Training metrics
- `models/data_splits.json` - Train/val/test split information

### 4. Resume Training

To resume from a checkpoint:

```bash
python scripts/train_model.py --config configs/pipeline_config.yaml --resume models/best_model.pth
```

## Hyperparameter Tuning

### Learning Rate

Start with default (0.0001) and adjust:
- Too high: Loss may not decrease or become unstable
- Too low: Training will be very slow

### Batch Size

- Increase if you have GPU memory (faster training)
- Decrease if you run out of memory
- Note: With patch-based training, effective batch size = `batch_size * num_samples`

### Spatial Size

- Larger patches: More context, but requires more GPU memory
- Smaller patches: Less memory, but may miss important context
- Recommended: [96, 96, 96] or [128, 128, 128] for most cases

### Model Architecture

**SwinUNETR:**
- Better accuracy (state-of-the-art)
- More parameters, slower training
- Requires more GPU memory

**SegResNet:**
- Faster training
- Fewer parameters
- Good for quick experiments

## Troubleshooting

### Out of Memory (OOM) Errors

1. Reduce `spatial_size` in config (e.g., [64, 64, 64])
2. Reduce `batch_size` to 1
3. Reduce `num_samples` in augmentation config
4. Use `cache_type: "disk"` instead of `"memory"`

### Training Loss Not Decreasing

1. Check learning rate (try 1e-5 or 1e-3)
2. Verify data quality (run validation)
3. Check that labels are correct
4. Ensure sufficient training data

### Validation Dice Not Improving

1. Model may be overfitting:
   - Reduce model complexity
   - Increase data augmentation
   - Add regularization
2. Check data split (may need more training data)
3. Verify label quality

### Slow Training

1. Use GPU if available
2. Increase `num_workers` for data loading
3. Use `cache_type: "memory"` for faster data access
4. Reduce `spatial_size` or `num_samples`

## Best Practices

1. **Always validate data first** - Catch issues before training
2. **Start with small epochs** - Test with 5-10 epochs first
3. **Monitor validation metrics** - Stop if overfitting
4. **Save checkpoints regularly** - Don't lose progress
5. **Use appropriate model** - SwinUNETR for accuracy, SegResNet for speed
6. **Balance data splits** - Ensure sufficient validation data

## Example Training Session

```bash
# 1. Validate
python scripts/validate_bids.py data/bids_dataset

# 2. Train for 50 epochs
python scripts/train_model.py --config configs/pipeline_config.yaml --epochs 50

# 3. Check results
cat models/training_history.json

# 4. Run inference
python scripts/run_inference.py --checkpoint models/best_model.pth --config configs/pipeline_config.yaml
```

## Advanced: Custom Training

For more control, modify `src/modeling/trainer.py` or create your own training script based on the provided functions.


