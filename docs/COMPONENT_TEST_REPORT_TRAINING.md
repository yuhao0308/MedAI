# Training Component Test Report

**Date**: 2025-11-13  
**Component**: Model Training  
**Test Command**: `python3 scripts/train_model.py --config configs/pipeline_config.yaml --epochs 2`

## Executive Summary

| Metric | Value |
|--------|-------|
| Overall Status | ❌ **FAIL** |
| Execution Time | ~6 minutes (failed during first epoch) |
| Critical Errors | 1 |
| Data Loading | ✅ SUCCESS |

## Test Execution

### Basic Training Test

**Command**: `python3 scripts/train_model.py --config configs/pipeline_config.yaml`

**Status**: ❌ **FAIL**

**Failure Point**: First batch of first epoch, during loss computation

### Configuration Used

- **Model Type**: SwinUNETR
- **Input Channels**: 1
- **Output Channels**: 2 (background + foreground)
- **Image Size**: [96, 96, 96]
- **Batch Size**: 1
- **Learning Rate**: 0.0001
- **Epochs**: 100 (failed before completion)
- **Train/Val/Test Split**: 35/7/8

### Component Status

| Component | Status | Details |
|-----------|--------|---------|
| Config Loading | ✅ PASS | Config loaded successfully |
| Data Loading | ✅ PASS | 35 train, 7 val samples loaded |
| Dataset Creation | ✅ PASS | CacheDataset created successfully |
| Model Creation | ✅ PASS | SwinUNETR model created (15,703,004 params) |
| Training Loop | ❌ FAIL | Failed on first batch |
| Checkpoint Saving | ❌ FAIL | No checkpoint saved |

### Data Loading Details

- **BIDS Dataset**: `data/bids_dataset`
- **Image-Label Pairs Found**: 50
- **Training Samples**: 35
- **Validation Samples**: 7
- **Test Samples**: 8
- **Data Loading Time**: ~13 seconds for training set, ~3 seconds for validation set

### Model Details

- **Model Architecture**: SwinUNETR
- **Total Parameters**: 15,703,004
- **Trainable Parameters**: 15,703,004
- **Model Creation**: Successful
- **Device**: CPU (CUDA not available)

## Critical Error

### Error: Label Value Out of Bounds

**Error Message**: 
```
RuntimeError: index 255 is out of bounds for dimension 1 with size 2
```

**Location**: 
- `src/modeling/trainer.py:94` (train_epoch function)
- `monai/losses/dice.py:171` (one_hot function)

**Stack Trace**:
```
File "/Users/yuhaowang/Documents/Itlize/MedAI/src/modeling/trainer.py", line 94, in train_epoch
    loss = loss_fn(outputs, labels)
File ".../monai/losses/dice.py", line 171, in forward
    target = one_hot(target, num_classes=n_pred_ch)
RuntimeError: index 255 is out of bounds for dimension 1 with size 2
```

**Root Cause Analysis**:

1. **Label Encoding Issue**: Labels contain pixel values up to 255 (8-bit encoding)
2. **Model Configuration**: Model expects 2 classes (background=0, foreground=1)
3. **Loss Function**: DiceCELoss tries to one-hot encode labels but fails when encountering value 255
4. **Missing Preprocessing**: No label normalization/clamping transform in preprocessing pipeline

**Impact**: 
- **Completely blocks training** - Cannot proceed past first batch
- Training cannot complete even a single epoch
- No model checkpoint can be saved

**Suggested Fixes**:

1. **Add Label Normalization Transform**:
   - Add transform to clamp label values to [0, 1] range
   - Or convert labels to binary (0/1) if multi-class not needed
   - Example: `AsDiscreted(threshold=0.5)` or `LabelToMaskd`

2. **Verify Label Encoding**:
   - Check if labels should be binary (0/1) or multi-class
   - If multi-class, update `out_channels` in model config
   - If binary, ensure labels are properly encoded

3. **Add Label Validation**:
   - Add check in dataloader to verify label value ranges
   - Add informative error message if mismatch detected

4. **Update Preprocessing Pipeline**:
   - Add label preprocessing step in `get_train_transforms()`
   - Ensure labels match model's expected class range

## Warnings

1. **CUDA Not Available**
   - Warning: "CUDA not available, using CPU (training will be slow)"
   - Impact: Low - Training runs on CPU but is very slow
   - Note: Expected on systems without GPU

2. **MONAI Deprecation Warning**
   - Warning: `FutureWarning` about `Orientationd` transform
   - Impact: Low - Future compatibility issue
   - Note: Should update to new API

3. **urllib3 OpenSSL Warning**
   - Warning: OpenSSL version mismatch
   - Impact: None - Cosmetic only

## Unit Tests Created

See `tests/test_training.py` for unit tests:
- `test_model_creation` - Tests model creation (✅ PASS)
- `test_training_execution` - Placeholder for full training test
- `test_data_loading` - Placeholder for data loading test
- `test_checkpoint_saving_loading` - Placeholder for checkpoint test

## Issues Found

1. **CRITICAL: Label value out of bounds** (see above)
   - Severity: Critical
   - Blocks: All training
   - Priority: Must fix before training can proceed

## Recommendations

1. **IMMEDIATE: Fix Label Preprocessing**
   - Add label normalization/clamping transform
   - Verify label encoding matches model configuration
   - Test with minimal epochs after fix

2. **Add Label Validation**
   - Check label value ranges in dataloader
   - Provide informative error messages

3. **Update MONAI Transforms**
   - Update `Orientationd` to use new API
   - Review other transforms for compatibility

4. **Test with Minimal Config**
   - Create test config with minimal epochs (1-2)
   - Test after label fix is applied

## Next Steps

1. Fix label preprocessing issue (CRITICAL)
2. Re-run training test with minimal epochs
3. Verify checkpoint saving works
4. Add more comprehensive unit tests


