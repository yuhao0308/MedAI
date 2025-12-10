"""
Pytest configuration and shared fixtures for test suite
"""

import pytest
import sys
import yaml
from pathlib import Path
import tempfile
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def project_root_path():
    """Return the project root directory path"""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def config_path(project_root_path):
    """Return path to main pipeline config"""
    return project_root_path / "configs" / "pipeline_config.yaml"


@pytest.fixture(scope="session")
def load_config(config_path):
    """Load and return the main pipeline configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


@pytest.fixture(scope="session")
def bids_dataset_path(project_root_path):
    """Return path to BIDS dataset"""
    return project_root_path / "data" / "bids_dataset"


@pytest.fixture(scope="session")
def models_dir(project_root_path):
    """Return path to models directory"""
    return project_root_path / "models"


@pytest.fixture(scope="function")
def temp_dir():
    """Create a temporary directory for test outputs"""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture(scope="function")
def temp_config(temp_dir, load_config):
    """Create a temporary config file for testing"""
    test_config = load_config.copy()
    # Override paths to use temp directory
    test_config['output']['model_save_dir'] = str(temp_dir / "models")
    test_config['output']['predictions_dir'] = str(temp_dir / "predictions")
    test_config['output']['reports_dir'] = str(temp_dir / "reports")
    test_config['output']['visualizations_dir'] = str(temp_dir / "visualizations")
    test_config['training']['cache_dir'] = str(temp_dir / "cache")
    
    config_path = temp_dir / "test_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(test_config, f, default_flow_style=False)
    
    return config_path


@pytest.fixture(scope="function")
def minimal_training_config(temp_dir):
    """Create a minimal config for quick training tests"""
    config = {
        'dataset': {
            'bids_root': str(temp_dir / "bids_dataset")
        },
        'preprocessing': {
            'spacing': [1.5, 1.5, 1.5],
            'orientation': 'RAS',
            'intensity_range': [-1000.0, 1000.0],
            'target_range': [0.0, 1.0]
        },
        'training': {
            'spatial_size': [64, 64, 64],  # Smaller for testing
            'batch_size': 1,
            'num_epochs': 1,  # Minimal for testing
            'learning_rate': 0.0001,
            'num_workers': 0,  # Avoid multiprocessing issues in tests
            'cache_type': 'none',
            'split': {
                'train_ratio': 0.7,
                'val_ratio': 0.15,
                'test_ratio': 0.15,
                'random_seed': 42
            }
        },
        'model': {
            'type': 'SwinUNETR',
            'in_channels': 1,
            'out_channels': 2,
            'img_size': [64, 64, 64]
        },
        'loss': {
            'type': 'DiceCELoss',
            'to_onehot_y': True,
            'softmax': True
        },
        'inference': {
            'roi_size': [64, 64, 64],
            'sw_batch_size': 2,
            'overlap': 0.25
        },
        'output': {
            'model_save_dir': str(temp_dir / "models"),
            'predictions_dir': str(temp_dir / "predictions"),
            'reports_dir': str(temp_dir / "reports"),
            'visualizations_dir': str(temp_dir / "visualizations")
        },
        'logging': {
            'level': 'WARNING'  # Reduce verbosity in tests
        }
    }
    
    config_path = temp_dir / "minimal_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return config_path


