"""
Visualization Utilities
NIfTI visualization, segmentation overlays, uncertainty maps
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple
import nibabel as nib
import logging

logger = logging.getLogger(__name__)


def visualize_nifti_slices(
    nifti_path: str,
    slice_indices: Optional[Tuple[int, int, int]] = None,
    output_path: Optional[str] = None,
    cmap: str = "gray"
) -> None:
    """
    Visualizes a NIfTI file as 2D slices in three orthogonal views.
    
    Args:
        nifti_path: Path to NIfTI file
        slice_indices: Tuple of (axial, coronal, sagittal) slice indices
                      If None, uses middle slices
        output_path: Optional path to save figure
        cmap: Colormap for visualization
    """
    img = nib.load(nifti_path)
    data = img.get_fdata()
    
    if slice_indices is None:
        # Use middle slices
        slice_indices = (
            data.shape[2] // 2,  # Axial
            data.shape[1] // 2,  # Coronal
            data.shape[0] // 2   # Sagittal
        )
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Axial
    axes[0].imshow(data[:, :, slice_indices[0]], cmap=cmap)
    axes[0].set_title(f"Axial Slice {slice_indices[0]}")
    axes[0].axis("off")
    
    # Coronal
    axes[1].imshow(data[:, slice_indices[1], :], cmap=cmap)
    axes[1].set_title(f"Coronal Slice {slice_indices[1]}")
    axes[1].axis("off")
    
    # Sagittal
    axes[2].imshow(data[slice_indices[2], :, :], cmap=cmap)
    axes[2].set_title(f"Sagittal Slice {slice_indices[2]}")
    axes[2].axis("off")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved visualization: {output_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_segmentation_overlay(
    image_path: str,
    segmentation_path: str,
    slice_index: Optional[int] = None,
    output_path: Optional[str] = None,
    alpha: float = 0.5
) -> None:
    """
    Visualizes segmentation mask overlaid on image.
    
    Args:
        image_path: Path to image NIfTI file
        segmentation_path: Path to segmentation NIfTI file
        slice_index: Slice index to visualize (axial)
        output_path: Optional path to save figure
        alpha: Transparency of overlay
    """
    img = nib.load(image_path)
    seg = nib.load(segmentation_path)
    
    img_data = img.get_fdata()
    seg_data = seg.get_fdata()
    
    if slice_index is None:
        slice_index = img_data.shape[2] // 2
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image
    axes[0].imshow(img_data[:, :, slice_index], cmap="gray")
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    # Overlay
    axes[1].imshow(img_data[:, :, slice_index], cmap="gray")
    axes[1].imshow(seg_data[:, :, slice_index], alpha=alpha, cmap="jet")
    axes[1].set_title("Segmentation Overlay")
    axes[1].axis("off")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved overlay visualization: {output_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_uncertainty_map(
    image_path: str,
    uncertainty_path: str,
    slice_index: Optional[int] = None,
    output_path: Optional[str] = None
) -> None:
    """
    Visualizes uncertainty map as heatmap overlay.
    
    Args:
        image_path: Path to image NIfTI file
        uncertainty_path: Path to uncertainty map NIfTI file
        slice_index: Slice index to visualize
        output_path: Optional path to save figure
    """
    img = nib.load(image_path)
    unc = nib.load(uncertainty_path)
    
    img_data = img.get_fdata()
    unc_data = unc.get_fdata()
    
    if slice_index is None:
        slice_index = img_data.shape[2] // 2
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image
    axes[0].imshow(img_data[:, :, slice_index], cmap="gray")
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    # Uncertainty heatmap
    im = axes[1].imshow(unc_data[:, :, slice_index], cmap="hot", alpha=0.7)
    axes[1].set_title("Uncertainty Map (Hot = High Uncertainty)")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], label="Uncertainty")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved uncertainty visualization: {output_path}")
    else:
        plt.show()
    
    plt.close()


def create_interactive_viewer(
    nifti_path: str,
    segmentation_path: Optional[str] = None
) -> None:
    """
    Creates an interactive 3D viewer using itkwidgets (for Jupyter).
    
    Args:
        nifti_path: Path to image NIfTI file
        segmentation_path: Optional path to segmentation NIfTI file
    """
    try:
        import itkwidgets as itk
        from itkwidgets import view
        
        img = nib.load(nifti_path)
        data = img.get_fdata()
        
        # Convert to itk image
        itk_image = itk.image_from_array(data)
        
        # Add segmentation if provided
        if segmentation_path:
            seg = nib.load(segmentation_path)
            seg_data = seg.get_fdata()
            itk_seg = itk.image_from_array(seg_data)
            
            # View with overlay
            view(itk_image, label_image=itk_seg)
        else:
            view(itk_image)
            
    except ImportError:
        logger.warning(
            "itkwidgets not installed. Install with: pip install itkwidgets"
        )
        logger.info("Falling back to static visualization")
        visualize_nifti_slices(nifti_path)


