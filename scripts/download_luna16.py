"""
LUNA16 Dataset Downloader
Downloads LUNA16 lung nodule CT scans (subsets 0-1, ~50 samples) and annotations.

LUNA16 is a public dataset for lung nodule detection.
Dataset page: https://luna16.grand-challenge.org/Download/
"""

import os
import sys
from pathlib import Path
import requests
from tqdm import tqdm
import zipfile
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_file(url, destination, desc="Downloading"):
    """Download file with progress bar."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        
        with open(destination, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        logger.info(f"Downloaded: {destination}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False


def extract_zip(zip_path, extract_to):
    """Extract zip file with progress bar."""
    try:
        logger.info(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        logger.info(f"Extracted to: {extract_to}")
        return True
    except Exception as e:
        logger.error(f"Failed to extract {zip_path}: {e}")
        return False


def main():
    """Download LUNA16 dataset."""
    
    # Set paths
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "data" / "luna16_raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directory: {output_dir}")
    
    # LUNA16 download URLs (from Google Drive mirrors or official sources)
    # Note: These are example URLs. The actual LUNA16 dataset requires registration
    # at https://luna16.grand-challenge.org/
    
    downloads = [
        {
            "name": "subset0",
            "filename": "subset0.zip",
            "info": "LUNA16 Subset 0 (contains ~88 CT scans)"
        },
        {
            "name": "annotations",
            "filename": "annotations.csv",
            "info": "Nodule annotations (coordinates and diameters)"
        }
    ]
    
    print("\n" + "="*70)
    print("LUNA16 DATASET DOWNLOAD")
    print("="*70)
    print("\nIMPORTANT: LUNA16 requires registration at:")
    print("https://luna16.grand-challenge.org/Download/")
    print("\nAfter registration, you can download:")
    print("1. subset0.zip (~40GB) - Contains ~88 CT scans")
    print("2. annotations.csv - Nodule annotations")
    print("\nFor this project, we only need subset0 (which has ~88 scans).")
    print("We'll process the first 50 scans from this subset.")
    print("="*70 + "\n")
    
    # Check if user has already downloaded files manually
    subset0_path = output_dir / "subset0.zip"
    annotations_path = output_dir / "annotations.csv"
    
    print("\nCHECKING FOR EXISTING FILES:")
    print(f"- subset0.zip: {'EXISTS' if subset0_path.exists() else 'NOT FOUND'}")
    print(f"- annotations.csv: {'EXISTS' if annotations_path.exists() else 'NOT FOUND'}")
    
    if not subset0_path.exists() or not annotations_path.exists():
        print("\n" + "="*70)
        print("MANUAL DOWNLOAD REQUIRED")
        print("="*70)
        print("\nPlease follow these steps:")
        print("\n1. Register at: https://luna16.grand-challenge.org/Download/")
        print("\n2. Download the following files:")
        print("   - subset0.zip (~40GB)")
        print("   - annotations.csv")
        print(f"\n3. Place them in: {output_dir}")
        print("\n4. Run this script again")
        print("="*70 + "\n")
        
        # Create a README in the output directory
        readme_path = output_dir / "README_DOWNLOAD.txt"
        with open(readme_path, 'w') as f:
            f.write("LUNA16 DATASET DOWNLOAD INSTRUCTIONS\n")
            f.write("="*70 + "\n\n")
            f.write("1. Register at: https://luna16.grand-challenge.org/Download/\n\n")
            f.write("2. Download these files:\n")
            f.write("   - subset0.zip (~40GB) - Contains ~88 CT scans\n")
            f.write("   - annotations.csv - Nodule annotations\n\n")
            f.write("3. Place them in this directory:\n")
            f.write(f"   {output_dir}\n\n")
            f.write("4. Run the download script again:\n")
            f.write("   python scripts/download_luna16.py\n\n")
            f.write("The preprocessing script will use the first 50 scans from subset0.\n")
        
        logger.info(f"Created download instructions: {readme_path}")
        return
    
    # Extract subset0 if needed
    subset0_extracted = output_dir / "subset0"
    if not subset0_extracted.exists():
        logger.info("Extracting subset0.zip...")
        if extract_zip(subset0_path, output_dir):
            logger.info(f"Subset0 extracted to: {subset0_extracted}")
        else:
            logger.error("Failed to extract subset0.zip")
            return
    else:
        logger.info(f"Subset0 already extracted: {subset0_extracted}")
    
    # Verify files
    mhd_files = list(subset0_extracted.glob("*.mhd"))
    logger.info(f"\nFound {len(mhd_files)} .mhd files in subset0")
    
    if len(mhd_files) > 0:
        logger.info(f"Will use first 50 scans for preprocessing")
        logger.info("\n✓ LUNA16 download/extraction complete!")
        logger.info(f"✓ Files ready in: {output_dir}")
        logger.info("\nNext step: Run preprocessing script")
        logger.info("  python scripts/preprocess_luna16.py")
    else:
        logger.error("No .mhd files found in subset0. Please check the extraction.")


if __name__ == "__main__":
    main()

