#!/usr/bin/env python3
"""
BIDS Dataset Validation Script
Validates structure, files, and data consistency
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.validation import run_all_validations

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_validation_report(report: dict):
    """Print human-readable validation report"""
    print("\n" + "="*80)
    print("BIDS DATASET VALIDATION REPORT")
    print("="*80)
    print(f"\nDataset: {report['bids_root']}")
    print(f"Status: {'✓ VALID' if report['valid'] else '✗ INVALID'}")
    
    # Summary
    summary = report['summary']
    print(f"\n--- Dataset Summary ---")
    print(f"Subjects: {summary['num_subjects']}")
    print(f"Images: {summary['num_images']}")
    print(f"Labels: {summary['num_labels']}")
    
    if 'common_shape' in summary:
        print(f"Common Image Shape: {summary['common_shape']}")
    if 'common_spacing' in summary:
        print(f"Common Spacing: {summary['common_spacing']} mm")
    if 'overall_intensity_range' in summary:
        print(f"Intensity Range: {summary['overall_intensity_range']}")
    
    # Errors
    if report['num_errors'] > 0:
        print(f"\n--- ERRORS ({report['num_errors']}) ---")
        for i, error in enumerate(report['all_errors'], 1):
            print(f"{i}. {error}")
    else:
        print("\n--- ERRORS ---")
        print("None ✓")
    
    # Warnings
    if report['num_warnings'] > 0:
        print(f"\n--- WARNINGS ({report['num_warnings']}) ---")
        for i, warning in enumerate(report['all_warnings'], 1):
            print(f"{i}. {warning}")
    else:
        print("\n--- WARNINGS ---")
        print("None ✓")
    
    # Structure validation details
    struct = report['structure_validation']
    if struct['summary']:
        print(f"\n--- Structure Details ---")
        print(f"Valid Subjects: {struct['summary'].get('valid_subjects', 0)}")
        print(f"Subjects with Images: {struct['summary'].get('subjects_with_images', 0)}")
        print(f"Subjects with Labels: {struct['summary'].get('subjects_with_labels', 0)}")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description='Validate BIDS dataset')
    parser.add_argument(
        'bids_root',
        type=str,
        help='Path to BIDS dataset root directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save JSON validation report'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Only print errors and warnings'
    )
    
    args = parser.parse_args()
    
    bids_root = Path(args.bids_root)
    if not bids_root.exists():
        logger.error(f"BIDS root directory does not exist: {bids_root}")
        sys.exit(1)
    
    # Run validations
    logger.info(f"Validating BIDS dataset: {bids_root}")
    report = run_all_validations(str(bids_root))
    
    # Print report
    if not args.quiet:
        print_validation_report(report)
    else:
        # Quiet mode: only print summary
        status = "VALID" if report['valid'] else "INVALID"
        print(f"Status: {status}")
        if report['num_errors'] > 0:
            print(f"Errors: {report['num_errors']}")
        if report['num_warnings'] > 0:
            print(f"Warnings: {report['num_warnings']}")
    
    # Save JSON report
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Saved validation report to {output_path}")
    
    # Exit with error code if validation failed
    if not report['valid']:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()


