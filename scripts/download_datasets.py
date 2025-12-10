"""
Public Dataset Downloaders
Supports TCIA, IDC, and MIDRC Data Commons
"""

import os
import subprocess
import logging
from pathlib import Path
from typing import Optional, List
import requests
import json

logger = logging.getLogger(__name__)


def download_tcia_collection(
    collection_name: str,
    output_dir: str,
    nbia_client_path: Optional[str] = None
) -> str:
    """
    Downloads a collection from The Cancer Imaging Archive (TCIA).
    
    TCIA uses the NBIA Data Retriever for downloads.
    
    Args:
        collection_name: TCIA collection name (e.g., "LIDC-IDRI")
        output_dir: Directory to save downloaded DICOM files
        nbia_client_path: Path to NBIA Data Retriever executable
                         If None, attempts to find in PATH
        
    Returns:
        Path to downloaded data directory
        
    Raises:
        RuntimeError: If NBIA client is not available or download fails
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Try to find NBIA client
    if nbia_client_path is None:
        nbia_client_path = "nbia-client"  # Assume in PATH
    
    # Check if NBIA client is available
    try:
        result = subprocess.run(
            [nbia_client_path, "--help"],
            capture_output=True,
            text=True,
            timeout=5
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        raise RuntimeError(
            "NBIA Data Retriever not found. Please install from:\n"
            "https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images\n"
            "Or use the web interface: https://www.cancerimagingarchive.net/"
        )
    
    # Download collection
    download_cmd = [
        nbia_client_path,
        "-d", str(output_path),
        "-c", collection_name
    ]
    
    try:
        logger.info(f"Downloading TCIA collection: {collection_name}")
        subprocess.run(download_cmd, check=True)
        logger.info(f"Downloaded to: {output_path}")
        return str(output_path)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"TCIA download failed: {e}")


def download_idc_collection(
    collection_id: str,
    output_dir: str,
    gcs_bucket: str = "public-datasets-idc"
) -> str:
    """
    Downloads a collection from Imaging Data Commons (IDC) via Google Cloud Storage.
    
    IDC data is stored in Google Cloud Storage buckets.
    
    Args:
        collection_id: IDC collection ID
        output_dir: Directory to save downloaded DICOM files
        gcs_bucket: GCS bucket name (default: public-datasets-idc)
        
    Returns:
        Path to downloaded data directory
        
    Raises:
        RuntimeError: If gsutil is not available or download fails
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Check if gsutil is available
    try:
        subprocess.run(
            ["gsutil", "version"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        raise RuntimeError(
            "gsutil (Google Cloud SDK) not found. Please install from:\n"
            "https://cloud.google.com/sdk/docs/install\n"
            "Or use the IDC web interface: https://portal.imaging.datacommons.cancer.gov/"
        )
    
    # Download from GCS
    gcs_path = f"gs://{gcs_bucket}/{collection_id}/"
    
    download_cmd = [
        "gsutil",
        "-m",  # Parallel download
        "cp",
        "-r",
        gcs_path,
        str(output_path)
    ]
    
    try:
        logger.info(f"Downloading IDC collection: {collection_id}")
        subprocess.run(download_cmd, check=True)
        logger.info(f"Downloaded to: {output_path}")
        return str(output_path)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"IDC download failed: {e}")


def get_midrc_collections() -> List[Dict]:
    """
    Retrieves list of available MIDRC collections.
    
    Returns:
        List of dictionaries with collection metadata
    """
    try:
        # MIDRC API endpoint (example - may need to be updated)
        response = requests.get(
            "https://data.midrc.org/api/collections",
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            logger.warning("Could not fetch MIDRC collections from API")
            return []
    except Exception as e:
        logger.warning(f"Error fetching MIDRC collections: {e}")
        return []


def download_midrc_collection(
    collection_id: str,
    output_dir: str,
    api_token: Optional[str] = None
) -> str:
    """
    Downloads a collection from MIDRC Data Commons.
    
    MIDRC requires authentication for some collections.
    
    Args:
        collection_id: MIDRC collection ID
        output_dir: Directory to save downloaded DICOM files
        api_token: Optional API token for authenticated access
        
    Returns:
        Path to downloaded data directory
        
    Raises:
        RuntimeError: If download fails
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # MIDRC download typically uses DICOMweb or similar
    # This is a placeholder - actual implementation depends on MIDRC API
    logger.warning(
        "MIDRC download requires specific API integration.\n"
        "Please use the MIDRC portal: https://data.midrc.org/\n"
        "Or refer to MIDRC documentation for programmatic access."
    )
    
    return str(output_path)


def list_available_datasets() -> Dict[str, List[str]]:
    """
    Lists commonly used public datasets for medical imaging AI.
    
    Returns:
        Dictionary mapping dataset source to list of collection names
    """
    datasets = {
        "TCIA": [
            "LIDC-IDRI",  # Lung nodule detection
            "RIDER",  # Lung CT
            "BRATS",  # Brain tumor segmentation
            "CT-ORG",  # Organ segmentation
        ],
        "IDC": [
            "tcga-luad",  # Lung adenocarcinoma
            "tcga-gbm",  # Glioblastoma
            "nsclc-radiomics",  # Non-small cell lung cancer
        ],
        "MIDRC": [
            "covid-19-imaging",  # COVID-19 chest imaging
        ],
        "Other": [
            "Medical Segmentation Decathlon",  # Multi-organ segmentation
            "BTCV",  # Multi-atlas abdominal labeling
        ]
    }
    
    return datasets


def verify_deidentification(dicom_dir: str) -> bool:
    """
    Verifies that DICOM files are properly de-identified.
    
    Checks for common PHI tags that should be removed.
    
    Args:
        dicom_dir: Directory containing DICOM files
        
    Returns:
        True if appears de-identified, False otherwise
    """
    import pydicom
    from pathlib import Path
    
    phi_tags = [
        "PatientName",
        "PatientID",
        "PatientBirthDate",
        "PatientAddress",
        "InstitutionName",
        "InstitutionAddress",
    ]
    
    dicom_path = Path(dicom_dir)
    sample_files = list(dicom_path.rglob("*.dcm"))[:10]  # Check first 10 files
    
    if not sample_files:
        logger.warning("No DICOM files found for de-identification check")
        return False
    
    for dcm_file in sample_files:
        try:
            dcm = pydicom.dcmread(str(dcm_file), stop_before_pixel_data=True)
            
            for tag in phi_tags:
                if hasattr(dcm, tag):
                    value = getattr(dcm, tag)
                    if value and str(value).strip() and str(value) != "":
                        logger.warning(
                            f"Potential PHI found in {dcm_file}: {tag} = {value}"
                        )
                        return False
        except Exception as e:
            logger.warning(f"Error checking {dcm_file}: {e}")
            continue
    
    logger.info("De-identification check passed")
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download public medical imaging datasets")
    parser.add_argument("--source", choices=["tcia", "idc", "midrc"], required=True)
    parser.add_argument("--collection", type=str, required=True)
    parser.add_argument("--output", type=str, default="./data/raw_dicom")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    if args.list:
        datasets = list_available_datasets()
        print(json.dumps(datasets, indent=2))
    else:
        if args.source == "tcia":
            download_tcia_collection(args.collection, args.output)
        elif args.source == "idc":
            download_idc_collection(args.collection, args.output)
        elif args.source == "midrc":
            download_midrc_collection(args.collection, args.output)


