"""
MONAI Data Loading Infrastructure
2025 Standard: CacheDataset/PersistentDataset with BIDS reader
"""

import os
import platform
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
from monai.data import (
    Dataset,
    CacheDataset,
    PersistentDataset,
    DataLoader,
    ThreadDataLoader,
    decollate_batch
)
from monai.transforms import Compose
import logging

logger = logging.getLogger(__name__)


def read_bids_dataset(
    bids_root: str,
    participants_tsv: Optional[str] = None
) -> List[Dict[str, str]]:
    """
    Reads a BIDS dataset and creates a list of data dictionaries.
    
    Each dictionary contains paths to image and label files.
    
    Args:
        bids_root: Root directory of BIDS dataset
        participants_tsv: Optional path to participants.tsv for filtering
        
    Returns:
        List of dictionaries with 'image' and 'label' keys
    """
    bids_path = Path(bids_root)
    data_dicts = []
    
    # Find all subject directories
    subject_dirs = [d for d in bids_path.iterdir() if d.is_dir() and d.name.startswith("sub-")]
    
    for sub_dir in subject_dirs:
        # Look for anatomical images
        anat_dir = sub_dir / "anat"
        if not anat_dir.exists():
            continue
        
        # Find image files
        image_files = list(anat_dir.glob("*.nii.gz")) + list(anat_dir.glob("*.nii"))
        
        for image_file in image_files:
            # Skip if it's already a label/segmentation
            if "seg" in image_file.name or "label" in image_file.name:
                continue
            
            # Look for corresponding label in derivatives
            sub_id = sub_dir.name
            label_dir = bids_path / "derivatives" / "labels" / sub_id / "anat"
            
            # Try to find any mask file for this subject
            # First try the original naming pattern
            label_file = label_dir / image_file.name.replace(".nii.gz", "_seg-tumor_mask.nii.gz")
            
            if not label_file.exists():
                # Try to find any mask file (for multi-organ datasets)
                mask_files = list(label_dir.glob("*_mask.nii.gz")) if label_dir.exists() else []
                if mask_files:
                    # For now, use the first mask found (could be extended to handle multiple organs)
                    label_file = mask_files[0]
                else:
                    label_file = None
            
            # If label doesn't exist, still include image (for inference)
            if label_file and not label_file.exists():
                label_file = None
            
            data_dict = {
                "image": str(image_file),
                "subject_id": sub_id
            }
            
            if label_file:
                data_dict["label"] = str(label_file)
            
            data_dicts.append(data_dict)
    
    logger.info(f"Found {len(data_dicts)} image-label pairs in BIDS dataset")
    return data_dicts


def create_dataset(
    data_dicts: List[Dict[str, str]],
    transform: Compose,
    cache_dir: Optional[str] = None,
    cache_type: str = "memory",
    num_workers: int = 4
) -> Dataset:
    """
    Creates a MONAI Dataset with caching support.
    
    Args:
        data_dicts: List of data dictionaries
        transform: MONAI Compose transform pipeline
        cache_dir: Directory for persistent caching (if cache_type="disk")
        cache_type: "memory" for CacheDataset, "disk" for PersistentDataset, "none" for Dataset
        num_workers: Number of workers for data loading (0 for single-threaded)
        
    Returns:
        MONAI Dataset object
    """
    # On macOS, multiprocessing can cause hanging issues, so reduce workers
    if platform.system() == "Darwin" and num_workers > 0:
        logger.warning(f"macOS detected: Reducing num_workers from {num_workers} to 0 for CacheDataset to avoid hanging")
        num_workers = 0
    
    if cache_type == "memory":
        dataset = CacheDataset(
            data=data_dicts,
            transform=transform,
            num_workers=num_workers,
            progress=True  # Show progress during caching
        )
        logger.info(f"Created CacheDataset (RAM caching, num_workers={num_workers})")
        
    elif cache_type == "disk":
        if cache_dir is None:
            raise ValueError("cache_dir must be provided for PersistentDataset")
        
        os.makedirs(cache_dir, exist_ok=True)
        dataset = PersistentDataset(
            data=data_dicts,
            transform=transform,
            cache_dir=cache_dir
        )
        logger.info(f"Created PersistentDataset (disk caching at {cache_dir})")
        
    else:  # cache_type == "none"
        dataset = Dataset(
            data=data_dicts,
            transform=transform
        )
        logger.info("Created Dataset (no caching)")
    
    return dataset


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 4,
    use_threads: bool = True
) -> DataLoader:
    """
    Creates a MONAI DataLoader.
    
    Args:
        dataset: MONAI Dataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes (0 for single-threaded, useful for debugging)
        use_threads: Use ThreadDataLoader instead of DataLoader
        
    Returns:
        MONAI DataLoader or ThreadDataLoader
    """
    # On macOS, multiprocessing can cause issues, so use 0 workers if num_workers > 0
    # This is a workaround for potential hanging issues
    import platform
    if platform.system() == "Darwin" and num_workers > 0:
        logger.warning(f"macOS detected: Reducing num_workers from {num_workers} to 0 to avoid multiprocessing issues")
        num_workers = 0
    
    if use_threads and num_workers > 0:
        dataloader = ThreadDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )
    else:
        # Use standard DataLoader with num_workers=0 for single-threaded loading
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0  # Single-threaded to avoid hanging
        )
    
    return dataloader


def split_dataset(
    data_dicts: List[Dict[str, str]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Dict[str, List[Dict[str, str]]]:
    """
    Splits dataset into train/val/test sets.
    
    Args:
        data_dicts: List of data dictionaries
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary with 'train', 'val', 'test' keys
    """
    import random
    
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")
    
    random.seed(random_seed)
    shuffled = data_dicts.copy()
    random.shuffle(shuffled)
    
    n_total = len(shuffled)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    splits = {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:]
    }
    
    logger.info(
        f"Dataset split: train={len(splits['train'])}, "
        f"val={len(splits['val'])}, test={len(splits['test'])}"
    )
    
    return splits

