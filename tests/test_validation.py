"""
Validation component tests
Tests for BIDS dataset validation functionality
"""

import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.validation import run_all_validations


class TestBIDSValidation:
    """Tests for BIDS dataset validation"""
    
    @pytest.mark.unit
    @pytest.mark.requires_data
    def test_validate_bids_dataset(self, bids_dataset_path):
        """
        Test validation of actual BIDS dataset
        """
        if not bids_dataset_path.exists():
            pytest.skip(f"BIDS dataset not found at {bids_dataset_path}")
        
        # Run validation
        report = run_all_validations(str(bids_dataset_path))
        
        # Check that validation completed
        assert 'valid' in report
        assert 'summary' in report
        assert 'all_errors' in report
        assert 'all_warnings' in report
        
        # Check that dataset is valid
        assert report['valid'] == True
        
        # Check summary structure
        summary = report['summary']
        assert 'num_subjects' in summary
        assert 'num_images' in summary
        assert 'num_labels' in summary
        assert summary['num_subjects'] > 0
        assert summary['num_images'] > 0
        
        # Check that no critical errors
        assert report['num_errors'] == 0
    
    @pytest.mark.unit
    def test_validate_empty_dataset(self, temp_dir):
        """
        Test validation with empty dataset
        """
        # TODO: Implement edge case test
        pass
    
    @pytest.mark.unit
    def test_validate_malformed_structure(self, temp_dir):
        """
        Test validation with malformed BIDS structure
        """
        # TODO: Implement edge case test
        pass
    
    @pytest.mark.unit
    def test_validation_report_structure(self):
        """
        Test that validation report has expected structure
        """
        # TODO: Implement validation report structure test
        pass

