"""
Download Small DICOM Datasets (< 20GB)
Helper script for downloading recommended small datasets
"""

import os
import requests
import zipfile
import tarfile
from pathlib import Path
import argparse
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_file(url: str, output_path: str, description: str = "Downloading"):
    """
    Downloads a file with progress bar.
    
    Args:
        url: URL to download from
        output_path: Path to save file
        description: Description for progress bar
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'wb') as f, tqdm(
        desc=description,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))


def download_3dicom_sample(output_dir: str, sample_name: str = "covid_lung"):
    """
    Downloads a sample from 3DICOM library.
    
    Note: This is a placeholder. Actual URLs need to be obtained from:
    https://3dicomviewer.com/dicom-library/
    
    Args:
        output_dir: Directory to save files
        sample_name: Name of sample to download
    """
    # These are example URLs - you'll need to get actual download links
    samples = {
        "covid_lung": {
            "url": "https://example.com/covid_lung.zip",  # Replace with actual URL
            "size": "135MB",
            "description": "CT Scan of COVID-19 Lung"
        },
        "skull_base": {
            "url": "https://example.com/skull_base.zip",  # Replace with actual URL
            "size": "50MB",
            "description": "CT Scan of Skull Base"
        }
    }
    
    if sample_name not in samples:
        logger.error(f"Unknown sample: {sample_name}")
        logger.info(f"Available samples: {list(samples.keys())}")
        return
    
    sample = samples[sample_name]
    logger.info(f"Note: 3DICOM samples require manual download from:")
    logger.info("https://3dicomviewer.com/dicom-library/")
    logger.info(f"Please download '{sample['description']}' manually and extract to {output_dir}")


def download_mipg_dataset(output_dir: str, dataset_name: str = "neck"):
    """
    Downloads MIPG dataset.
    
    Note: MIPG datasets require registration and manual download from:
    https://www.mipg.upenn.edu/Vnews/mipg_data_sharing.html
    
    Args:
        output_dir: Directory to save files
        dataset_name: Name of dataset (neck, thorax, abdomen, pelvis, torso)
    """
    datasets = {
        "neck": {"size": "426MB", "description": "Neck Dataset"},
        "thorax": {"size": "836MB", "description": "Thorax Dataset"},
        "abdomen": {"size": "~1GB", "description": "Abdomen Dataset"},
        "pelvis": {"size": "~1GB", "description": "Pelvis Dataset"},
        "torso": {"size": "~2GB", "description": "Torso Dataset"}
    }
    
    if dataset_name not in datasets:
        logger.error(f"Unknown dataset: {dataset_name}")
        logger.info(f"Available datasets: {list(datasets.keys())}")
        return
    
    dataset = datasets[dataset_name]
    logger.info(f"MIPG datasets require manual download:")
    logger.info("1. Go to: https://www.mipg.upenn.edu/Vnews/mipg_data_sharing.html")
    logger.info(f"2. Download '{dataset['description']}' ({dataset['size']})")
    logger.info(f"3. Extract to: {output_dir}")


def extract_archive(archive_path: str, output_dir: str):
    """
    Extracts a zip or tar archive.
    
    Args:
        archive_path: Path to archive file
        output_dir: Directory to extract to
    """
    logger.info(f"Extracting {archive_path} to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    if archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
    elif archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
        with tarfile.open(archive_path, 'r:gz') as tar_ref:
            tar_ref.extractall(output_dir)
    elif archive_path.endswith('.tar'):
        with tarfile.open(archive_path, 'r') as tar_ref:
            tar_ref.extractall(output_dir)
    else:
        logger.error(f"Unknown archive format: {archive_path}")
        return
    
    logger.info(f"Extracted to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Download small DICOM datasets (< 20GB)"
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["3dicom", "mipg", "manual"],
        default="manual",
        help="Dataset source"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset name (e.g., 'covid_lung', 'neck', 'thorax')"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/raw_dicom",
        help="Output directory"
    )
    parser.add_argument(
        "--extract",
        type=str,
        help="Path to archive file to extract"
    )
    
    args = parser.parse_args()
    
    if args.extract:
        # Extract an already-downloaded archive
        extract_archive(args.extract, args.output)
        logger.info(f"Extraction complete. DICOM files are in: {args.output}")
        return
    
    if args.source == "3dicom":
        if not args.dataset:
            args.dataset = "covid_lung"
        download_3dicom_sample(args.output, args.dataset)
        
    elif args.source == "mipg":
        if not args.dataset:
            args.dataset = "neck"
        download_mipg_dataset(args.output, args.dataset)
        
    else:
        # Manual download instructions
        logger.info("=" * 60)
        logger.info("MANUAL DOWNLOAD INSTRUCTIONS")
        logger.info("=" * 60)
        logger.info("")
        logger.info("Recommended datasets under 20GB:")
        logger.info("")
        logger.info("1. 3DICOM Samples (135MB - 50MB):")
        logger.info("   https://3dicomviewer.com/dicom-library/")
        logger.info("   - CT Scan of COVID-19 Lung: 135MB")
        logger.info("   - CT Scan of Skull Base: 50MB")
        logger.info("")
        logger.info("2. MIPG Datasets (426MB - 2GB):")
        logger.info("   https://www.mipg.upenn.edu/Vnews/mipg_data_sharing.html")
        logger.info("   - Neck Dataset: 426MB")
        logger.info("   - Thorax Dataset: 836MB")
        logger.info("")
        logger.info("3. TCIA LIDC-IDRI (Subset 5-10GB):")
        logger.info("   https://www.cancerimagingarchive.net/collection/lidc-idri/")
        logger.info("   Download first 20-30 patients only")
        logger.info("")
        logger.info(f"After downloading, extract to: {args.output}")
        logger.info("")
        logger.info("Then run:")
        logger.info(f"  python scripts/ingest_data.py --input-dir {args.output}")


if __name__ == "__main__":
    main()


