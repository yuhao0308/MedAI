# Issue Tracking - Phase 0 Week 1

**Date Created**: 2025-11-13  
**Phase**: Phase 0 - Complete & Stabilize Current Research Pipeline  
**Week**: Week 1 - Initial Testing & Issue Documentation

## Summary

| Priority | Count | Status |
|----------|-------|--------|
| Critical | 1 | Open |
| High | 1 | Open (Expected) |
| Medium | 0 | - |
| Low | 3 | Open |

## Critical Issues (Blocking)

### Issue #1: Label Value Out of Bounds in Loss Function

**Priority**: 🔴 **CRITICAL**  
**Status**: ✅ **RESOLVED**  
**Component**: Training  
**Affected Files**: 
- `src/modeling/trainer.py:94`
- `src/preprocessing/transforms.py` (missing label preprocessing)

**Description**:
Training fails immediately on the first batch of the first epoch with a runtime error when computing the loss function. The error occurs because labels contain pixel values up to 255 (8-bit encoding), but the model expects only 2 classes (background=0, foreground=1).

**Error Message**:
```
RuntimeError: index 255 is out of bounds for dimension 1 with size 2
```

**Stack Trace**:
```
File "/Users/yuhaowang/Documents/Itlize/MedAI/src/modeling/trainer.py", line 94, in train_epoch
    loss = loss_fn(outputs, labels)
File ".../monai/losses/dice.py", line 171, in forward
    target = one_hot(target, num_classes=n_pred_ch)
RuntimeError: index 255 is out of bounds for dimension 1 with size 2
```

**Steps to Reproduce**:
1. Run: `python3 scripts/train_model.py --config configs/pipeline_config.yaml`
2. Training starts, data loads successfully
3. First batch of first epoch fails immediately

**Expected Behavior**:
Training should proceed normally, processing batches and computing loss without errors.

**Actual Behavior**:
Training fails on first batch with index out of bounds error.

**Root Cause**:
- Labels are encoded as 8-bit values (0-255)
- Model is configured for 2 classes (out_channels=2)
- Loss function (DiceCELoss) tries to one-hot encode labels but fails when encountering value 255
- No label preprocessing/normalization in transform pipeline

**Impact**:
- **Completely blocks all training**
- Cannot train any models
- Cannot generate model checkpoints
- Blocks inference testing (no models available)

**Suggested Fix**:
1. Add label normalization transform to preprocessing pipeline:
   - Option A: Clamp labels to [0, 1] range: `AsDiscreted(threshold=0.5)` or custom transform
   - Option B: Convert to binary: `LabelToMaskd` or threshold-based conversion
   - Option C: If multi-class needed, update `out_channels` in model config

2. Add label validation in dataloader:
   - Check label value ranges before training
   - Provide informative error message if mismatch detected

3. Update preprocessing transforms:
   - Add label preprocessing step in `get_train_transforms()`
   - Ensure labels match model's expected class range

**Files Modified**:
- ✅ `src/preprocessing/transforms.py` - Added `BinaryLabelNormalizationd` transform class
- ✅ `src/preprocessing/transforms.py` - Added label normalization to `get_pre_transforms()`
- ✅ `src/preprocessing/transforms.py` - Fixed MONAI `Orientationd` deprecation warning

**Resolution**:
- Created `BinaryLabelNormalizationd` transform that converts any non-zero label value to 1, keeping 0 as 0
- Added transform to preprocessing pipeline in `get_pre_transforms()` 
- Transform handles both numpy arrays and torch tensors
- Tested: Successfully converts labels from [0, 1, 255, 128, 0] to [0., 1., 1., 1., 0.]

**Estimated Effort**: 2 hours (completed)

**Related Issues**: Issue #2 (will resolve automatically once training succeeds)

**Test Plan After Fix**:
1. ✅ Transform tested and working
2. ⏭️ Re-run training with minimal epochs (1-2) - Pending
3. ⏭️ Verify loss computation works - Pending
4. ⏭️ Verify training completes at least one epoch - Pending
5. ⏭️ Verify checkpoint is saved - Pending
6. ⏭️ Run full pipeline test - Pending

---

## High Priority Issues

### Issue #2: Model Checkpoint Not Found (Expected)

**Priority**: 🟠 **HIGH** (but expected)  
**Status**: Open (Expected - will resolve after Issue #1 is fixed)  
**Component**: Inference  
**Affected Files**: 
- `scripts/run_pipeline.py:262`

**Description**:
Inference step cannot execute because model checkpoint does not exist. This is expected since training failed (Issue #1).

**Error Message**:
```
Model checkpoint not found: models/best_model.pth
```

**Steps to Reproduce**:
1. Run: `python3 scripts/run_pipeline.py full --config configs/pipeline_config.yaml`
2. Training fails (Issue #1)
3. Inference step attempts to run but finds no checkpoint

**Expected Behavior**:
After successful training, checkpoint should exist and inference should run.

**Actual Behavior**:
Checkpoint does not exist because training failed.

**Root Cause**:
Training failed before any checkpoint could be saved.

**Impact**:
- Cannot test inference component
- Blocks full pipeline execution
- Prevents evaluation of model performance

**Suggested Fix**:
- Resolve Issue #1 first (label preprocessing)
- Training will then succeed and generate checkpoint
- Inference will then work automatically

**Files to Modify**: None (will resolve automatically)

**Estimated Effort**: 0 hours (depends on Issue #1)

**Related Issues**: Issue #1

**Test Plan After Fix**:
1. Verify checkpoint exists after training
2. Run inference with checkpoint
3. Verify predictions are generated
4. Verify evaluation works (if ground truth available)

---

## Low Priority Issues (Warnings)

### Issue #3: Z-Dimension Differences in Organ Masks (118 warnings)

**Priority**: 🟡 **LOW**  
**Status**: Informational (Expected behavior)  
**Component**: Validation  
**Affected Files**: 
- `src/ingestion/validation.py`

**Description**:
Validation generates 118 warnings about Z-dimension differences between full images and organ-specific masks. This is expected behavior because organ masks only cover the region where the organ exists, not the full image volume.

**Warning Message**:
```
sub-021: Z-dimension differs - Image 60 vs Label 13 slices (sub-021_CT_seg-Stmch_mask.nii.gz) - This is normal for organ-specific masks
```

**Steps to Reproduce**:
1. Run: `python3 scripts/validate_bids.py data/bids_dataset`
2. See 118 warnings about Z-dimension differences

**Expected Behavior**:
Warnings are informational and expected for organ-specific masks.

**Actual Behavior**:
Warnings are generated as expected.

**Root Cause**:
Organ masks are smaller in Z-dimension than full images, which is normal.

**Impact**:
- None - These are informational warnings
- Does not affect functionality
- May clutter output

**Suggested Fix**:
1. Option A: Suppress these warnings (they're expected)
2. Option B: Categorize as "info" instead of "warning"
3. Option C: Add flag to validation to skip Z-dimension checks for organ masks

**Files to Modify**:
- `src/ingestion/validation.py` - Update warning categorization

**Estimated Effort**: 1 hour

**Related Issues**: None

**Test Plan After Fix**:
1. Re-run validation
2. Verify warnings are suppressed or recategorized
3. Verify validation still catches real issues

---

### Issue #4: urllib3 OpenSSL Warning

**Priority**: 🟡 **LOW**  
**Status**: Open  
**Component**: Dependencies  
**Affected Files**: Multiple (urllib3 library)

**Description**:
Multiple warnings about urllib3 v2 requiring OpenSSL 1.1.1+, but system has LibreSSL 2.8.3.

**Warning Message**:
```
NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'LibreSSL 2.8.3'. See: https://github.com/urllib3/urllib3/issues/3020
```

**Steps to Reproduce**:
1. Run any Python script that uses urllib3
2. See warning in output

**Expected Behavior**:
No warnings, or warnings suppressed.

**Actual Behavior**:
Warning appears multiple times during execution.

**Root Cause**:
urllib3 v2 requires OpenSSL 1.1.1+, but macOS uses LibreSSL.

**Impact**:
- Low - Cosmetic only
- Does not affect functionality
- Clutters output

**Suggested Fix**:
1. Option A: Suppress warning in code
2. Option B: Pin urllib3 to compatible version
3. Option C: Update OpenSSL (may require system changes)

**Files to Modify**:
- Add warning filter in main scripts or config
- Or update `requirements.txt` to pin urllib3 version

**Estimated Effort**: 30 minutes

**Related Issues**: None

**Test Plan After Fix**:
1. Run pipeline
2. Verify warnings are suppressed
3. Verify functionality still works

---

### Issue #5: MONAI Deprecation Warning

**Priority**: 🟡 **LOW**  
**Status**: ✅ **RESOLVED**  
**Component**: Preprocessing  
**Affected Files**: 
- `src/preprocessing/transforms.py`

**Description**:
FutureWarning about `Orientationd` transform default argument change in future MONAI versions.

**Warning Message**:
```
FutureWarning: monai.transforms.spatial.dictionary Orientationd.__init__:labels: Current default value of argument `labels=(('L', 'R'), ('P', 'A'), ('I', 'S'))` was changed in version None from `labels=(('L', 'R'), ('P', 'A'), ('I', 'S'))` to `labels=None`.
```

**Steps to Reproduce**:
1. Run training or any script using transforms
2. See deprecation warning

**Expected Behavior**:
No deprecation warnings, or use updated API.

**Actual Behavior**:
Warning appears during transform initialization.

**Root Cause**:
MONAI API change - `Orientationd` now uses `labels=None` by default.

**Impact**:
- Low - Future compatibility issue
- Does not affect current functionality
- May break in future MONAI versions

**Suggested Fix**:
1. Update `Orientationd` usage to explicitly set `labels=None` or use new API
2. Review other MONAI transforms for similar issues

**Files Modified**:
- ✅ `src/preprocessing/transforms.py` - Updated `Orientationd` to use `labels=None` parameter

**Resolution**:
- Updated all `Orientationd` calls to include `labels=None` parameter
- This uses the new MONAI API that uses meta-tensor space information
- Applied to both training/validation transforms and inference transforms

**Estimated Effort**: 15 minutes (completed)

**Related Issues**: None

**Test Plan After Fix**:
1. ✅ Transform import tested successfully
2. ⏭️ Run training to verify no deprecation warnings - Pending
3. ✅ Transforms still work correctly

---

## Issue Resolution Status

### Open Issues: 3
- Critical: 0
- High: 1 (Expected - will resolve after training test)
- Low: 2

### Resolved Issues: 3
- ✅ Issue #1: Label preprocessing (CRITICAL)
- ✅ Issue #5: MONAI deprecation warning (LOW)
- ✅ Training hanging issue (macOS multiprocessing) - NEW

### Blocked Issues: 1
- Issue #2 (blocked by Issue #1 - should resolve after training test)

---

## Next Steps

1. ✅ **COMPLETED**: Fixed Issue #1 (Label preprocessing) - CRITICAL
2. ✅ **COMPLETED**: Fixed Issue #5 (MONAI deprecation warning) - LOW
3. ✅ **COMPLETED**: Fixed training hanging issue (macOS multiprocessing) - NEW
4. ⏭️ **PENDING**: Test training with 1 epoch to verify all fixes work
5. ⏭️ **PENDING**: After training succeeds, Issue #2 will resolve automatically
6. ⏭️ **PENDING**: Re-run full pipeline test after fixes verified
7. ⏭️ Address remaining low-priority issues (Issues #3, #4) as time permits

---

## Issue Tracking Format

Each issue follows this format:
- **Priority**: Critical/High/Medium/Low
- **Status**: Open/In Progress/Resolved/Blocked
- **Component**: Which part of the system
- **Description**: What the issue is
- **Steps to Reproduce**: How to trigger it
- **Expected vs Actual**: What should happen vs what does
- **Root Cause**: Why it's happening
- **Impact**: What it affects
- **Suggested Fix**: How to fix it
- **Files to Modify**: Which files need changes
- **Estimated Effort**: Time estimate
- **Related Issues**: Links to other issues
- **Test Plan**: How to verify the fix

