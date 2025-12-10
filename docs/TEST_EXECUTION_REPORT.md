# Full Pipeline Test Execution Report

**Date**: 2025-11-13  
**Executor**: Automated Test Script  
**Pipeline Version**: Phase 0 Week 1  
**Config File**: `configs/pipeline_config.yaml`

## Executive Summary

| Metric | Value |
|--------|-------|
| Overall Status | **FAIL** |
| Execution Time | 517.18 seconds (8.6 minutes) |
| Steps Completed | 1/3 (Validation passed, Training failed, Inference skipped) |
| Critical Errors | 1 |
| Warnings | 118 (all related to Z-dimension differences in organ masks - expected) |

## Test Environment

- **Python Version**: 3.9
- **Operating System**: macOS (darwin 24.3.0)
- **GPU Available**: No (CPU only)
- **CUDA Version**: N/A
- **Dataset Location**: `data/bids_dataset`
- **Dataset Size**: 50 subjects

## Step-by-Step Execution

### Step 1: Validation

**Command**: `python3 scripts/validate_bids.py data/bids_dataset`

**Status**: ✅ **PASS**

**Execution Time**: ~2 minutes 40 seconds

**Output Summary**:
- Number of subjects validated: 50
- Number of images: 50
- Number of labels: 650
- Validation errors: 0
- Validation warnings: 118 (all related to Z-dimension differences - expected for organ-specific masks)

**Dataset Details**:
- Common Image Shape: (512, 512, 55)
- Common Spacing: (0.703125, 0.703125, 5.0) mm
- Intensity Range: (0.0, 4095.0)
- Valid Subjects: 50
- Subjects with Images: 50
- Subjects with Labels: 50

**Issues Found**:
None - Validation passed successfully. All warnings are expected for organ-specific masks where Z-dimension differs from full image.

**Log Output**:
```
Status: ✓ VALID
--- ERRORS ---
None ✓
--- WARNINGS (118) ---
[All warnings are about Z-dimension differences in organ masks - this is normal]
```

---

### Step 2: Training

**Command**: `python3 scripts/train_model.py --config configs/pipeline_config.yaml`

**Status**: ❌ **FAIL**

**Execution Time**: ~6 minutes (failed during first epoch)

**Configuration Used**:
- Model type: SwinUNETR
- Epochs: 100 (failed before completion)
- Batch size: 1
- Learning rate: 0.0001
- Train/Val/Test split: 35/7/8

**Training Progress**:
- Data loading: ✅ SUCCESS (35 train, 7 val samples loaded)
- Model creation: ✅ SUCCESS (15,703,004 parameters)
- Training loop: ❌ FAIL (failed during first epoch, first batch)
- Checkpoint saving: ❌ FAIL (no checkpoint saved)

**Final Metrics**: N/A (training failed before any metrics)

**Issues Found**:
1. **CRITICAL: Label value out of bounds in loss function**
   - Severity: **Critical**
   - Component: `src/modeling/trainer.py` - Loss computation
   - Error message: `RuntimeError: index 255 is out of bounds for dimension 1 with size 2`
   - Location: `monai/losses/dice.py` line 171, in `one_hot` function
   - Impact: **Blocks all training** - Cannot train model with current label format
   - Root Cause: Labels contain values up to 255, but model expects only 2 classes (background=0, foreground=1)
   - Suggested Fix: 
     - Preprocess labels to ensure values are in range [0, 1] or [0, num_classes-1]
     - Add label normalization/clamping in preprocessing transforms
     - Verify label encoding matches model's `out_channels` configuration

**Log Output**:
```
2025-11-13 00:18:56,052 - __main__ - INFO - Model parameters: 15,703,004 total, 15,703,004 trainable
2025-11-13 00:18:56,052 - __main__ - INFO - Starting training...
2025-11-13 00:18:56,054 - src.modeling.trainer - INFO - Epoch 1/100
...
RuntimeError: index 255 is out of bounds for dimension 1 with size 2
  File "/Users/yuhaowang/Documents/Itlize/MedAI/src/modeling/trainer.py", line 94, in train_epoch
    loss = loss_fn(outputs, labels)
  File ".../monai/losses/dice.py", line 171, in forward
    target = one_hot(target, num_classes=n_pred_ch)
```

---

### Step 3: Inference

**Command**: `python3 scripts/run_inference.py --checkpoint models/best_model.pth --config configs/pipeline_config.yaml --evaluate`

**Status**: ⏭️ **SKIP** (Training failed, no model checkpoint available)

**Execution Time**: N/A

**Inference Progress**:
- Model loading: ❌ SKIP (checkpoint not found)
- Data loading: ❌ SKIP
- Inference execution: ❌ SKIP
- Prediction saving: ❌ SKIP
- Evaluation: ❌ SKIP

**Results**: N/A

**Issues Found**:
1. **Model checkpoint not found**
   - Severity: **High** (but expected due to training failure)
   - Component: `scripts/run_pipeline.py`
   - Error message: `Model checkpoint not found: models/best_model.pth`
   - Impact: Inference cannot run without trained model
   - Note: This is expected since training failed. Will be resolved once training issue is fixed.

**Log Output**:
```
2025-11-13 00:24:32,353 - __main__ - ERROR - Model checkpoint not found: models/best_model.pth
```

---

## Full Pipeline Command

**Command**: `python3 scripts/run_pipeline.py full --config configs/pipeline_config.yaml`

**Status**: [PASS/FAIL/PARTIAL]

**Total Execution Time**: [Duration]

**Complete Log**: See `test_execution_output_[timestamp].json` and `test_execution_output_[timestamp]_readable.txt`

---

## Error Analysis

### Critical Errors (Blocking)

1. **Label Value Out of Bounds in Loss Function**
   - **Location**: `src/modeling/trainer.py:94` → `monai/losses/dice.py:171`
   - **Message**: `RuntimeError: index 255 is out of bounds for dimension 1 with size 2`
   - **Stack Trace**: 
     ```
     File "/Users/yuhaowang/Documents/Itlize/MedAI/src/modeling/trainer.py", line 94, in train_epoch
       loss = loss_fn(outputs, labels)
     File ".../monai/losses/dice.py", line 171, in forward
       target = one_hot(target, num_classes=n_pred_ch)
     RuntimeError: index 255 is out of bounds for dimension 1 with size 2
     ```
   - **Impact**: **Completely blocks training** - Cannot proceed past first batch of first epoch
   - **Root Cause**: 
     - Labels contain pixel values up to 255 (likely 8-bit encoded masks)
     - Model configured for 2 classes (background=0, foreground=1)
     - Loss function tries to one-hot encode labels but fails when encountering value 255
   - **Suggested Fix**: 
     1. Add label preprocessing transform to clamp/normalize values to [0, 1] range
     2. Or convert multi-class labels to binary (0/1) if appropriate
     3. Verify label encoding matches `out_channels=2` configuration
     4. Add validation in dataloader to check label value ranges

### High Priority Errors

1. **Model Checkpoint Not Found (Expected)**
   - **Location**: `scripts/run_pipeline.py:262`
   - **Message**: `Model checkpoint not found: models/best_model.pth`
   - **Impact**: Inference step cannot execute
   - **Note**: This is expected since training failed. Will resolve automatically once training issue is fixed.

### Warnings

1. **Z-Dimension Differences in Organ Masks (118 warnings)**
   - **Location**: `src/ingestion/validation.py` - Data consistency checks
   - **Message**: Multiple warnings about Z-dimension differences between images and organ-specific masks
   - **Impact**: None - This is expected behavior for organ-specific masks that don't span the full image volume
   - **Example**: `sub-021: Z-dimension differs - Image 60 vs Label 13 slices (sub-021_CT_seg-Stmch_mask.nii.gz)`
   - **Note**: These warnings are informational and don't indicate errors. Organ masks are typically smaller than full images.

2. **urllib3 OpenSSL Warning**
   - **Location**: Multiple (urllib3 library)
   - **Message**: `NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'LibreSSL 2.8.3'`
   - **Impact**: Low - Cosmetic warning, doesn't affect functionality
   - **Note**: Can be suppressed or resolved by updating OpenSSL/urllib3

3. **MONAI Deprecation Warning**
   - **Location**: `monai/utils/deprecate_utils.py:321`
   - **Message**: `FutureWarning: monai.transforms.spatial.dictionary Orientationd.__init__:labels: Current default value...`
   - **Impact**: Low - Future compatibility warning
   - **Note**: Should update to use new API in future MONAI versions

---

## Data Flow Verification

| Component | Input | Output | Status |
|-----------|-------|--------|--------|
| Validation | BIDS dataset | Validation report | [PASS/FAIL] |
| Training | BIDS dataset | Model checkpoint | [PASS/FAIL] |
| Inference | Model + BIDS dataset | Predictions | [PASS/FAIL] |

---

## Resource Usage

- **Peak Memory Usage**: [If available]
- **GPU Memory Usage**: [If available]
- **Disk Space Used**: [If available]
- **CPU Usage**: [If available]

---

## Recommendations

1. **CRITICAL: Fix Label Preprocessing**
   - Add label normalization/clamping transform to ensure label values are in [0, 1] or [0, num_classes-1] range
   - Investigate label encoding: Are labels 8-bit (0-255) or should they be binary (0/1)?
   - Update preprocessing transforms to handle label value ranges correctly
   - Add validation to check label value ranges before training

2. **Update MONAI Transform Usage**
   - Update `Orientationd` transform to use new API to avoid deprecation warnings
   - Review other MONAI transforms for compatibility

3. **Suppress/Resolve urllib3 Warning**
   - Update OpenSSL or suppress warning if not critical
   - Consider pinning urllib3 version if needed

4. **Add Better Error Handling**
   - Add validation in dataloader to catch label value issues early
   - Add informative error messages when label/model configuration mismatch occurs
   - Add progress indicators for long-running operations

5. **Test with GPU**
   - Re-run tests with GPU available to verify performance and identify GPU-specific issues

---

## Next Steps

1. Document all issues in `ISSUE_TRACKING.md`
2. Test individual components separately (Tickets 3-5)
3. Prioritize bug fixes based on impact
4. Re-run pipeline after fixes

---

## Appendix

### Complete Error Messages

[Full error messages and stack traces]

### Configuration Used

[Full config file contents if relevant]

### System Information

[Detailed system information if relevant]

