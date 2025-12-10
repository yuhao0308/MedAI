"""
Nodule Post-Processing Pipeline
Connected component analysis, size filtering, and feature extraction for lung nodules
"""

import numpy as np
from scipy import ndimage
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class LungRADSCategory(Enum):
    """Lung-RADS assessment categories based on nodule size"""
    CATEGORY_1 = "1"  # Negative: No nodules or definitely benign
    CATEGORY_2 = "2"  # Benign appearance: <6mm solid, <6mm part-solid
    CATEGORY_3 = "3"  # Probably benign: 6-8mm solid, 6mm+ part-solid
    CATEGORY_4A = "4A"  # Suspicious: 8-15mm solid
    CATEGORY_4B = "4B"  # Very suspicious: >15mm solid
    CATEGORY_4X = "4X"  # Additional features concerning for malignancy


@dataclass
class NoduleCandidate:
    """Data class representing a detected nodule candidate"""
    id: int
    centroid_voxel: Tuple[int, int, int]
    centroid_mm: Tuple[float, float, float]
    bounding_box: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]
    volume_voxels: int
    volume_mm3: float
    equivalent_diameter_mm: float
    max_diameter_mm: float
    mean_confidence: float
    min_confidence: float
    max_confidence: float
    sphericity: float
    lung_rads_category: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


def extract_nodule_candidates(
    segmentation: np.ndarray,
    spacing: Tuple[float, float, float],
    confidence_map: Optional[np.ndarray] = None,
    min_voxels: int = 10
) -> List[NoduleCandidate]:
    """
    Extract individual nodule candidates from segmentation using connected components.
    
    Args:
        segmentation: Binary or multi-class segmentation mask
        spacing: Voxel spacing in mm (x, y, z)
        confidence_map: Optional confidence/probability map
        min_voxels: Minimum number of voxels for a valid candidate
        
    Returns:
        List of NoduleCandidate objects
    """
    # Ensure binary mask (foreground vs background)
    binary_mask = (segmentation > 0).astype(np.uint8)
    
    # Connected component labeling
    labeled_array, num_features = ndimage.label(binary_mask)
    
    logger.info(f"Found {num_features} connected components")
    
    candidates = []
    
    for label_id in range(1, num_features + 1):
        # Extract component mask
        component_mask = (labeled_array == label_id)
        volume_voxels = int(np.sum(component_mask))
        
        # Skip very small components
        if volume_voxels < min_voxels:
            continue
        
        # Compute features
        features = compute_nodule_features(
            mask=component_mask,
            spacing=spacing,
            confidence_map=confidence_map
        )
        
        candidate = NoduleCandidate(
            id=len(candidates) + 1,
            **features
        )
        
        candidates.append(candidate)
    
    # Sort by volume (largest first)
    candidates.sort(key=lambda x: x.volume_mm3, reverse=True)
    
    # Re-assign IDs after sorting
    for i, candidate in enumerate(candidates):
        candidate.id = i + 1
    
    logger.info(f"Extracted {len(candidates)} nodule candidates after filtering")
    
    return candidates


def compute_nodule_features(
    mask: np.ndarray,
    spacing: Tuple[float, float, float],
    confidence_map: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Compute geometric and confidence features for a nodule.
    
    Args:
        mask: Binary mask for single nodule
        spacing: Voxel spacing in mm
        confidence_map: Optional confidence map
        
    Returns:
        Dictionary of computed features
    """
    # Get voxel coordinates
    coords = np.array(np.where(mask)).T  # Shape: (N, 3)
    
    # Volume
    volume_voxels = int(len(coords))
    voxel_volume_mm3 = float(np.prod(spacing))
    volume_mm3 = volume_voxels * voxel_volume_mm3
    
    # Centroid in voxel coordinates
    centroid_voxel = tuple(int(x) for x in np.mean(coords, axis=0))
    
    # Centroid in mm coordinates
    centroid_mm = tuple(float(c * s) for c, s in zip(centroid_voxel, spacing))
    
    # Bounding box
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)
    bounding_box = tuple(
        (int(min_c), int(max_c)) for min_c, max_c in zip(min_coords, max_coords)
    )
    
    # Equivalent spherical diameter: d = (6V/π)^(1/3)
    equivalent_diameter_mm = float((6 * volume_mm3 / np.pi) ** (1/3))
    
    # Max diameter (longest axis of bounding box)
    bbox_dims_mm = [
        (max_c - min_c + 1) * s 
        for (min_c, max_c), s in zip(bounding_box, spacing)
    ]
    max_diameter_mm = float(max(bbox_dims_mm))
    
    # Sphericity: ratio of equivalent sphere surface to actual surface
    # Approximation: sphericity = (π^(1/3) * (6V)^(2/3)) / A
    # For simplicity, we use ratio of equivalent diameter to max diameter
    sphericity = float(equivalent_diameter_mm / max_diameter_mm) if max_diameter_mm > 0 else 0.0
    
    # Confidence statistics
    if confidence_map is not None:
        confidence_values = confidence_map[mask]
        mean_confidence = float(np.mean(confidence_values))
        min_confidence = float(np.min(confidence_values))
        max_confidence = float(np.max(confidence_values))
    else:
        mean_confidence = 1.0
        min_confidence = 1.0
        max_confidence = 1.0
    
    # Lung-RADS category based on size
    lung_rads = get_lung_rads_category(equivalent_diameter_mm)
    
    return {
        "centroid_voxel": centroid_voxel,
        "centroid_mm": centroid_mm,
        "bounding_box": bounding_box,
        "volume_voxels": volume_voxels,
        "volume_mm3": volume_mm3,
        "equivalent_diameter_mm": equivalent_diameter_mm,
        "max_diameter_mm": max_diameter_mm,
        "mean_confidence": mean_confidence,
        "min_confidence": min_confidence,
        "max_confidence": max_confidence,
        "sphericity": sphericity,
        "lung_rads_category": lung_rads.value
    }


def get_lung_rads_category(diameter_mm: float) -> LungRADSCategory:
    """
    Determine Lung-RADS category based on nodule diameter.
    
    This is a simplified categorization based on solid nodule size only.
    Clinical Lung-RADS also considers nodule type (solid, part-solid, ground-glass),
    growth rate, and other features.
    
    Args:
        diameter_mm: Nodule equivalent diameter in mm
        
    Returns:
        LungRADSCategory enum value
    """
    if diameter_mm < 3:
        return LungRADSCategory.CATEGORY_1  # Likely artifact or very small
    elif diameter_mm < 6:
        return LungRADSCategory.CATEGORY_2  # Benign appearance
    elif diameter_mm < 8:
        return LungRADSCategory.CATEGORY_3  # Probably benign
    elif diameter_mm < 15:
        return LungRADSCategory.CATEGORY_4A  # Suspicious
    else:
        return LungRADSCategory.CATEGORY_4B  # Very suspicious


def filter_by_size(
    candidates: List[NoduleCandidate],
    min_diameter_mm: float = 3.0,
    max_diameter_mm: float = 100.0
) -> List[NoduleCandidate]:
    """
    Filter nodule candidates by diameter.
    
    Args:
        candidates: List of NoduleCandidate objects
        min_diameter_mm: Minimum diameter threshold
        max_diameter_mm: Maximum diameter threshold
        
    Returns:
        Filtered list of candidates
    """
    filtered = [
        c for c in candidates
        if min_diameter_mm <= c.equivalent_diameter_mm <= max_diameter_mm
    ]
    
    logger.info(f"Filtered {len(candidates)} -> {len(filtered)} candidates by size ({min_diameter_mm}-{max_diameter_mm}mm)")
    
    return filtered


def filter_by_confidence(
    candidates: List[NoduleCandidate],
    min_confidence: float = 0.5
) -> List[NoduleCandidate]:
    """
    Filter nodule candidates by confidence score.
    
    Args:
        candidates: List of NoduleCandidate objects
        min_confidence: Minimum mean confidence threshold
        
    Returns:
        Filtered list of candidates
    """
    filtered = [
        c for c in candidates
        if c.mean_confidence >= min_confidence
    ]
    
    logger.info(f"Filtered {len(candidates)} -> {len(filtered)} candidates by confidence (>={min_confidence})")
    
    return filtered


def filter_by_lung_rads(
    candidates: List[NoduleCandidate],
    min_category: str = "2"
) -> List[NoduleCandidate]:
    """
    Filter nodule candidates by Lung-RADS category.
    
    Args:
        candidates: List of NoduleCandidate objects
        min_category: Minimum Lung-RADS category to include
        
    Returns:
        Filtered list of candidates
    """
    category_order = ["1", "2", "3", "4A", "4B", "4X"]
    min_idx = category_order.index(min_category) if min_category in category_order else 0
    
    filtered = [
        c for c in candidates
        if category_order.index(c.lung_rads_category) >= min_idx
    ]
    
    logger.info(f"Filtered {len(candidates)} -> {len(filtered)} candidates by Lung-RADS (>={min_category})")
    
    return filtered


def get_nodule_mask(
    segmentation: np.ndarray,
    nodule_id: int
) -> np.ndarray:
    """
    Extract mask for a specific nodule by ID.
    
    Args:
        segmentation: Original segmentation mask
        nodule_id: ID of the nodule to extract
        
    Returns:
        Binary mask for the specified nodule
    """
    binary_mask = (segmentation > 0).astype(np.uint8)
    labeled_array, _ = ndimage.label(binary_mask)
    return (labeled_array == nodule_id).astype(np.uint8)


def create_labeled_nodule_map(
    segmentation: np.ndarray,
    candidates: List[NoduleCandidate]
) -> np.ndarray:
    """
    Create a labeled map where each nodule has a unique label.
    
    Args:
        segmentation: Original segmentation mask
        candidates: List of NoduleCandidate objects
        
    Returns:
        Labeled array where each voxel has the nodule ID (0 for background)
    """
    binary_mask = (segmentation > 0).astype(np.uint8)
    labeled_array, _ = ndimage.label(binary_mask)
    
    # Create output with candidates re-labeled by their ID
    output = np.zeros_like(labeled_array, dtype=np.uint8)
    
    for candidate in candidates:
        # Find which original label corresponds to this candidate's centroid
        original_label = labeled_array[candidate.centroid_voxel]
        if original_label > 0:
            output[labeled_array == original_label] = candidate.id
    
    return output


def compute_aggregate_statistics(
    candidates: List[NoduleCandidate]
) -> Dict[str, Any]:
    """
    Compute aggregate statistics across all nodule candidates.
    
    Args:
        candidates: List of NoduleCandidate objects
        
    Returns:
        Dictionary with aggregate statistics
    """
    if not candidates:
        return {
            "total_nodules": 0,
            "total_volume_mm3": 0.0,
            "mean_diameter_mm": 0.0,
            "max_diameter_mm": 0.0,
            "category_counts": {},
            "suspicious_count": 0
        }
    
    # Category counts
    category_counts = {}
    for c in candidates:
        cat = c.lung_rads_category
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    # Suspicious nodules (4A, 4B, 4X)
    suspicious_count = sum(
        1 for c in candidates 
        if c.lung_rads_category in ["4A", "4B", "4X"]
    )
    
    return {
        "total_nodules": len(candidates),
        "total_volume_mm3": sum(c.volume_mm3 for c in candidates),
        "mean_diameter_mm": np.mean([c.equivalent_diameter_mm for c in candidates]),
        "max_diameter_mm": max(c.equivalent_diameter_mm for c in candidates),
        "mean_confidence": np.mean([c.mean_confidence for c in candidates]),
        "category_counts": category_counts,
        "suspicious_count": suspicious_count
    }


def postprocess_segmentation(
    segmentation: np.ndarray,
    spacing: Tuple[float, float, float],
    confidence_map: Optional[np.ndarray] = None,
    min_diameter_mm: float = 3.0,
    max_diameter_mm: float = 100.0,
    min_confidence: float = 0.3,
    min_voxels: int = 10
) -> Tuple[List[NoduleCandidate], Dict[str, Any]]:
    """
    Full post-processing pipeline for nodule segmentation.
    
    Args:
        segmentation: Segmentation prediction
        spacing: Voxel spacing in mm
        confidence_map: Optional confidence map
        min_diameter_mm: Minimum nodule diameter
        max_diameter_mm: Maximum nodule diameter
        min_confidence: Minimum confidence threshold
        min_voxels: Minimum voxels for connected component
        
    Returns:
        Tuple of (filtered_candidates, aggregate_statistics)
    """
    # Extract all candidates
    candidates = extract_nodule_candidates(
        segmentation=segmentation,
        spacing=spacing,
        confidence_map=confidence_map,
        min_voxels=min_voxels
    )
    
    # Apply filters
    candidates = filter_by_size(candidates, min_diameter_mm, max_diameter_mm)
    
    if confidence_map is not None:
        candidates = filter_by_confidence(candidates, min_confidence)
    
    # Compute aggregate statistics
    stats = compute_aggregate_statistics(candidates)
    
    return candidates, stats


