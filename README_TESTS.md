# Test Suite Documentation

## Overview

This directory contains the test suite for the Medical Imaging AI Pipeline. The tests are organized by component and use pytest as the testing framework.

## Test Structure

```
tests/
├── __init__.py              # Test package initialization
├── conftest.py              # Pytest configuration and shared fixtures
├── test_pipeline.py         # End-to-end pipeline tests
├── test_validation.py       # Validation component tests
├── test_training.py         # Training component tests
├── test_inference.py        # Inference component tests
└── fixtures/                # Test data fixtures (future)
```

## Running Tests

### Run All Tests

```bash
pytest
```

### Run Specific Test File

```bash
pytest tests/test_validation.py
```

### Run Tests with Markers

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"

# Skip tests requiring GPU
pytest -m "not requires_gpu"

# Skip tests requiring data
pytest -m "not requires_data"
```

### Run Tests with Coverage

```bash
pytest --cov=src --cov-report=html
```

### Run Tests Verbosely

```bash
pytest -v
```

## Test Markers

Tests are marked with pytest markers to categorize them:

- `@pytest.mark.unit` - Unit tests (fast, isolated)
- `@pytest.mark.integration` - Integration tests (slower, require components)
- `@pytest.mark.slow` - Slow tests (may take minutes)
- `@pytest.mark.requires_gpu` - Tests that require GPU
- `@pytest.mark.requires_data` - Tests that require data files

## Fixtures

Shared fixtures are defined in `conftest.py`:

- `project_root_path` - Path to project root
- `config_path` - Path to main pipeline config
- `load_config` - Loaded configuration dictionary
- `bids_dataset_path` - Path to BIDS dataset
- `models_dir` - Path to models directory
- `temp_dir` - Temporary directory for test outputs
- `temp_config` - Temporary config file for testing
- `minimal_training_config` - Minimal config for quick training tests

## Test Categories

### Unit Tests

Fast, isolated tests that test individual functions or classes:
- Model creation
- Transform application
- Utility functions

### Integration Tests

Tests that verify components work together:
- Full pipeline execution
- Training with real data
- Inference with trained models

## Current Test Status

### Implemented Tests

- ✅ `test_model_creation` - Tests model creation (SwinUNETR, SegResNet)
- ✅ `test_validate_bids_dataset` - Tests BIDS validation

### Placeholder Tests (To Be Implemented)

- ⏭️ `test_full_pipeline_execution` - Full pipeline test
- ⏭️ `test_training_execution` - Training test (blocked by label issue)
- ⏭️ `test_inference_execution` - Inference test (requires trained model)
- ⏭️ `test_validate_empty_dataset` - Edge case test
- ⏭️ `test_validate_malformed_structure` - Edge case test
- ⏭️ `test_data_loading` - Data loading test
- ⏭️ `test_checkpoint_saving_loading` - Checkpoint test
- ⏭️ `test_model_loading` - Model loading test
- ⏭️ `test_uncertainty_quantification` - Uncertainty test

## Known Issues

See `docs/ISSUE_TRACKING.md` for known issues that affect testing:

1. **CRITICAL**: Label preprocessing issue blocks training tests
2. **HIGH**: No model checkpoint available for inference tests

## Writing New Tests

### Test Naming

- Test files: `test_*.py`
- Test classes: `Test*`
- Test functions: `test_*`

### Example Test

```python
import pytest
from src.modeling.trainer import create_model

class TestModelCreation:
    @pytest.mark.unit
    def test_create_swinunetr(self):
        model = create_model(
            model_type="SwinUNETR",
            in_channels=1,
            out_channels=2,
            img_size=(64, 64, 64)
        )
        assert model is not None
        assert isinstance(model, torch.nn.Module)
```

### Using Fixtures

```python
def test_with_config(load_config):
    assert 'dataset' in load_config
    assert 'training' in load_config
```

### Skipping Tests

```python
@pytest.mark.skip(reason="Requires GPU")
def test_gpu_only():
    pass

def test_conditional_skip(bids_dataset_path):
    if not bids_dataset_path.exists():
        pytest.skip("BIDS dataset not found")
```

## Continuous Integration

Tests should be run in CI/CD pipeline:

1. Run unit tests on every commit
2. Run integration tests on pull requests
3. Run full test suite before releases

## Test Coverage Goals

- Target: >80% code coverage
- Current: TBD (to be measured after fixes)

## Troubleshooting

### Tests Fail with Import Errors

Make sure project root is in Python path:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pytest
```

### Tests Require Data Files

Some tests require the BIDS dataset. Skip them if data is not available:
```bash
pytest -m "not requires_data"
```

### Tests Are Slow

Use markers to skip slow tests:
```bash
pytest -m "not slow"
```

## Future Improvements

1. Add more unit tests for all components
2. Add edge case tests
3. Add performance benchmarks
4. Add mock fixtures for testing without data
5. Add CI/CD integration
6. Add test coverage reporting


