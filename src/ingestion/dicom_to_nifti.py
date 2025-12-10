"""
DICOM-to-NIfTI Conversion
2025 Standard: Uses dcm2niix command-line tool for robust conversion
"""

import subprocess
import os
from pathlib import Path
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


def check_dcm2niix_installed() -> bool:
    """
    Checks if dcm2niix is installed and available in PATH.
    
    Returns:
        True if dcm2niix is available, False otherwise
    """
    try:
        result = subprocess.run(
            ["dcm2niix", "-h"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def convert_dicom_to_nifti(
    input_dir: str,
    output_dir: str,
    output_filename: Optional[str] = None,
    bids_compliant: bool = True,
    compress: bool = True
) -> str:
    """
    Converts a directory of DICOM files to a single NIfTI file using dcm2niix.
    
    Args:
        input_dir: Directory containing DICOM files for a single series
        output_dir: Directory where NIfTI file will be saved
        output_filename: Base name for output file (without extension)
                        If None, dcm2niix will auto-generate
        bids_compliant: Use BIDS-compliant naming conventions
        compress: Output compressed .nii.gz format
        
    Returns:
        Path to the created NIfTI file
        
    Raises:
        RuntimeError: If dcm2niix is not installed or conversion fails
    """
    if not check_dcm2niix_installed():
        raise RuntimeError(
            "dcm2niix is not installed. Please install it from:\n"
            "https://github.com/rordenlab/dcm2niix\n"
            "Or via: brew install dcm2niix (macOS) or apt-get install dcm2niix (Linux)"
        )
    
    # Ensure output directory exists and use absolute path
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Use absolute path for input directory
    input_dir = os.path.abspath(input_dir)
    
    # Build dcm2niix command
    cmd = ["dcm2niix"]
    
    # Output directory (must be absolute)
    cmd.extend(["-o", output_dir])
    
    # Compression
    if compress:
        cmd.extend(["-z", "y"])  # Compress output (y=pigz compression, creates .nii.gz)
    
    # BIDS-compliant naming
    if bids_compliant and output_filename:
        # dcm2niix format: %p=participant, %t=time, %s=series, %d=description
        # For BIDS: sub-001_T1w
        cmd.extend(["-f", output_filename])
    elif output_filename:
        cmd.extend(["-f", output_filename])
    
    # Additional flags for better compatibility
    cmd.extend(["-b", "y"])  # Save BIDS sidecar JSON (y=yes)
    # Note: -y flag removed as it may not be supported in all dcm2niix versions
    
    # Input directory (must be last argument)
    cmd.append(input_dir)
    
    try:
        logger.info(f"Running dcm2niix: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,  # Don't raise on non-zero exit, check output instead
            timeout=300  # 5 minute timeout
        )
        
        # Log dcm2niix output for debugging
        if result.stdout:
            logger.debug(f"dcm2niix stdout: {result.stdout}")
        if result.stderr:
            logger.debug(f"dcm2niix stderr: {result.stderr}")
        
        # Find the created NIfTI file
        output_path = Path(output_dir)
        nifti_files = list(output_path.glob("*.nii.gz")) + list(output_path.glob("*.nii"))
        
        if nifti_files:
            # Return the most recently created file
            nifti_file = max(nifti_files, key=lambda p: p.stat().st_mtime)
            logger.info(f"Successfully converted to: {nifti_file}")
            return str(nifti_file)
        else:
            # Check if dcm2niix reported an error
            error_msg = "dcm2niix completed but no NIfTI file was created"
            if result.stderr:
                error_msg += f"\ndcm2niix stderr: {result.stderr}"
            if result.stdout:
                error_msg += f"\ndcm2niix stdout: {result.stdout}"
            raise RuntimeError(error_msg)
            
    except subprocess.CalledProcessError as e:
        error_msg = f"dcm2niix conversion failed: {e.stderr}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    except subprocess.TimeoutExpired:
        error_msg = "dcm2niix conversion timed out (>5 minutes)"
        logger.error(error_msg)
        raise RuntimeError(error_msg)


def batch_convert_dicom_to_nifti(
    dicom_directories: List[str],
    output_base_dir: str,
    naming_function: Optional[callable] = None
) -> List[str]:
    """
    Batch converts multiple DICOM series directories to NIfTI.
    
    Args:
        dicom_directories: List of directories, each containing a DICOM series
        output_base_dir: Base directory for output NIfTI files
        naming_function: Optional function(dicom_dir, index) -> output_filename
        
    Returns:
        List of paths to created NIfTI files
    """
    nifti_files = []
    
    for idx, dicom_dir in enumerate(dicom_directories):
        if naming_function:
            output_filename = naming_function(dicom_dir, idx)
        else:
            # Default: use directory name
            output_filename = os.path.basename(os.path.normpath(dicom_dir))
        
        try:
            nifti_path = convert_dicom_to_nifti(
                input_dir=dicom_dir,
                output_dir=output_base_dir,
                output_filename=output_filename,
                bids_compliant=True
            )
            nifti_files.append(nifti_path)
        except Exception as e:
            logger.error(f"Failed to convert {dicom_dir}: {e}")
            continue
    
    return nifti_files

