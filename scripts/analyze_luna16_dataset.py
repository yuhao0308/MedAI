"""
LUNA16 Dataset Analysis Script

Analyzes the preprocessed LUNA16 dataset to understand:
1. How many subjects are positive (with nodules) vs negative (no nodules)
2. Distribution of number of nodules per subject
3. Distribution of nodule sizes (small / medium / large)

This analysis helps design a stratified subject selection and splitting strategy.

Usage:
    python scripts/analyze_luna16_dataset.py --bids-root /path/to/luna16_processed
    
    # For Kaggle (copy this into a notebook cell):
    # Just copy the functions and run analyze_full_dataset()
"""

import os
import sys
from pathlib import Path
import numpy as np
import json
from typing import List, Dict, Optional
from collections import Counter
import argparse

# Try to import required packages
try:
    import nibabel as nib
except ImportError:
    print("nibabel not installed. Install with: pip install nibabel")
    sys.exit(1)

try:
    from scipy import ndimage
except ImportError:
    print("scipy not installed. Install with: pip install scipy")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("matplotlib not installed. Plots will be skipped.")


# =============================================================================
# DATASET LOADING
# =============================================================================

def read_luna16_dataset(bids_root: Path, verbose: bool = True) -> List[Dict[str, str]]:
    """
    Read LUNA16 dataset in BIDS-like format with robust filename matching.
    """
    if verbose:
        print(f"\n📂 Scanning BIDS dataset: {bids_root}")

    bids_root = Path(bids_root)
    if not bids_root.exists():
        raise ValueError(f"❌ BIDS root does not exist: {bids_root}")

    data_dicts = []
    errors = []

    # Find all subject directories
    subject_dirs = sorted(
        [d for d in bids_root.iterdir() if d.is_dir() and d.name.startswith("sub-")]
    )
    if len(subject_dirs) == 0:
        raise ValueError(f"❌ No subject directories found in {bids_root}")

    if verbose:
        print(f"   Found {len(subject_dirs)} subjects")

    for sub_dir in subject_dirs:
        sub_id = sub_dir.name

        try:
            # ----- IMAGE -----
            anat_dir = sub_dir / "anat"
            if not anat_dir.exists():
                errors.append(f"{sub_id}: anat/ not found")
                continue

            img_candidates = (
                list(anat_dir.glob("*CT*.nii.gz"))
                + list(anat_dir.glob("*CT*.nii"))
                + list(anat_dir.glob("*.nii.gz"))
                + list(anat_dir.glob("*.nii"))
            )
            if not img_candidates:
                errors.append(f"{sub_id}: No NIfTI image found")
                continue
            image_path = img_candidates[0]

            # ----- LABEL -----
            label_anat_dir = bids_root / "derivatives" / "labels" / sub_id / "anat"
            if not label_anat_dir.exists():
                errors.append(f"{sub_id}: label anat/ not found")
                continue

            lbl_candidates = (
                list(label_anat_dir.glob("*seg-nodule_mask.nii.gz"))
                + list(label_anat_dir.glob("*seg-nodule_mask.nii"))
                + list(label_anat_dir.glob("*.nii.gz"))
                + list(label_anat_dir.glob("*.nii"))
            )
            if not lbl_candidates:
                errors.append(f"{sub_id}: No NIfTI label found")
                continue
            label_path = lbl_candidates[0]

            if image_path.stat().st_size == 0 or label_path.stat().st_size == 0:
                errors.append(f"{sub_id}: Empty file")
                continue

            data_dicts.append({
                "image": str(image_path),
                "label": str(label_path),
                "subject_id": sub_id,
            })

        except Exception as e:
            errors.append(f"{sub_id}: {str(e)}")
            continue

    if verbose:
        print(f"   ✅ Found {len(data_dicts)} valid image-label pairs")
        if errors:
            print(f"   ⚠️  {len(errors)} errors (showing first 5):")
            for error in errors[:5]:
                print(f"      - {error}")

    return data_dicts


# =============================================================================
# SINGLE SUBJECT ANALYSIS
# =============================================================================

def analyze_single_subject(data_dict: Dict, verbose: bool = False) -> Dict:
    """
    Analyze a single subject's label to determine nodule characteristics.
    
    Returns:
        Dict with:
        - has_nodules: bool
        - num_nodules: int
        - nodule_sizes_mm: List[float] - approximate diameter of each nodule
        - nodule_volumes_mm3: List[float] - volume of each nodule
        - size_category: str - 'negative', 'small', 'medium', 'large', 'mixed'
    """
    label_path = data_dict["label"]
    subject_id = data_dict.get("subject_id", Path(label_path).stem)
    
    try:
        # Load the label/mask
        nii = nib.load(label_path)
        mask = nii.get_fdata()
        
        # Get voxel spacing (mm)
        voxel_spacing = nii.header.get_zooms()[:3]
        if len(voxel_spacing) < 3:
            voxel_spacing = (1.0, 1.0, 1.0)  # Default if not available
        
        # Check if any nodules exist
        has_nodules = np.any(mask > 0)
        
        if not has_nodules:
            return {
                "subject_id": subject_id,
                "has_nodules": False,
                "num_nodules": 0,
                "nodule_sizes_mm": [],
                "nodule_volumes_mm3": [],
                "size_category": "negative",
                "total_nodule_volume_mm3": 0.0,
                "max_nodule_diameter_mm": 0.0,
                "voxel_spacing": list(voxel_spacing),
            }
        
        # Find connected components (individual nodules)
        binary_mask = (mask > 0).astype(np.int32)
        labeled_array, num_nodules = ndimage.label(binary_mask)
        
        # Calculate size of each nodule
        voxel_volume_mm3 = float(np.prod(voxel_spacing))
        nodule_sizes_mm = []
        nodule_volumes_mm3 = []
        
        for nodule_id in range(1, num_nodules + 1):
            nodule_mask = (labeled_array == nodule_id)
            nodule_volume_voxels = int(np.sum(nodule_mask))
            nodule_volume_mm3 = nodule_volume_voxels * voxel_volume_mm3
            nodule_volumes_mm3.append(nodule_volume_mm3)
            
            # Approximate diameter assuming spherical nodule
            # V = (4/3) * π * r³  →  d = 2 * (3V / 4π)^(1/3)
            if nodule_volume_mm3 > 0:
                diameter_mm = 2 * ((3 * nodule_volume_mm3) / (4 * np.pi)) ** (1/3)
            else:
                diameter_mm = 0.0
            nodule_sizes_mm.append(diameter_mm)
        
        # Determine size category based on largest nodule
        max_size = max(nodule_sizes_mm) if nodule_sizes_mm else 0
        
        # Standard lung nodule size categories:
        # - Small: < 6mm (typically benign, low risk)
        # - Medium: 6mm to <10mm (follow-up recommended)  
        # - Large: ≥ 10mm (high suspicion, biopsy often needed)
        if max_size < 6:
            size_category = "small"
        elif max_size < 10:
            size_category = "medium"
        else:
            size_category = "large"
        
        # Check if mixed sizes (nodules in different categories)
        if len(nodule_sizes_mm) > 1:
            size_cats = []
            for s in nodule_sizes_mm:
                if s < 6:
                    size_cats.append("small")
                elif s < 10:
                    size_cats.append("medium")
                else:
                    size_cats.append("large")
            if len(set(size_cats)) > 1:
                size_category = "mixed"
        
        if verbose:
            print(f"  {subject_id}: {num_nodules} nodule(s), max diameter: {max_size:.1f}mm, category: {size_category}")
        
        return {
            "subject_id": subject_id,
            "has_nodules": True,
            "num_nodules": num_nodules,
            "nodule_sizes_mm": nodule_sizes_mm,
            "nodule_volumes_mm3": nodule_volumes_mm3,
            "size_category": size_category,
            "total_nodule_volume_mm3": sum(nodule_volumes_mm3),
            "max_nodule_diameter_mm": max_size,
            "voxel_spacing": list(voxel_spacing),
        }
        
    except Exception as e:
        print(f"⚠️ Error analyzing {subject_id}: {e}")
        return {
            "subject_id": subject_id,
            "has_nodules": False,
            "num_nodules": 0,
            "nodule_sizes_mm": [],
            "nodule_volumes_mm3": [],
            "size_category": "error",
            "total_nodule_volume_mm3": 0.0,
            "max_nodule_diameter_mm": 0.0,
            "error": str(e),
        }


# =============================================================================
# FULL DATASET ANALYSIS
# =============================================================================

def analyze_full_dataset(data_dicts: List[Dict], verbose: bool = True) -> Dict:
    """
    Analyze entire dataset and return comprehensive statistics.
    
    Returns:
        Dict with:
        - subject_analyses: List of per-subject analysis dicts
        - summary: Overall statistics
    """
    print("\n" + "="*70)
    print("LUNA16 DATASET ANALYSIS")
    print("="*70)
    
    analyses = []
    for i, data_dict in enumerate(data_dicts):
        if verbose and (i + 1) % 10 == 0:
            print(f"  Analyzing subject {i+1}/{len(data_dicts)}...")
        analysis = analyze_single_subject(data_dict, verbose=False)
        analysis["data_dict"] = data_dict  # Keep reference to original
        analyses.append(analysis)
    
    # === SUMMARY STATISTICS ===
    
    # 1. Positive vs Negative
    positive_cases = [a for a in analyses if a["has_nodules"]]
    negative_cases = [a for a in analyses if not a["has_nodules"]]
    
    # 2. Nodule count distribution (among positive cases)
    nodule_counts = [a["num_nodules"] for a in positive_cases]
    count_distribution = Counter(nodule_counts)
    
    # Categorize by nodule count
    single_nodule = len([a for a in positive_cases if a["num_nodules"] == 1])
    multi_nodule = len([a for a in positive_cases if 2 <= a["num_nodules"] <= 3])
    many_nodule = len([a for a in positive_cases if a["num_nodules"] >= 4])
    
    # 3. Size category distribution
    size_categories = [a["size_category"] for a in analyses]
    size_distribution = Counter(size_categories)
    
    # 4. Nodule size statistics (all individual nodules)
    all_sizes = []
    for a in positive_cases:
        all_sizes.extend(a["nodule_sizes_mm"])
    
    # Size breakdown
    small_nodules = [s for s in all_sizes if s < 6]
    medium_nodules = [s for s in all_sizes if 6 <= s < 10]
    large_nodules = [s for s in all_sizes if s >= 10]
    
    summary = {
        "total_subjects": len(data_dicts),
        "positive_count": len(positive_cases),
        "negative_count": len(negative_cases),
        "positive_ratio": len(positive_cases) / len(data_dicts) if data_dicts else 0,
        
        "nodule_count_distribution": dict(count_distribution),
        "nodule_count_categories": {
            "single_nodule (1)": single_nodule,
            "multi_nodule (2-3)": multi_nodule,
            "many_nodule (4+)": many_nodule,
        },
        
        "size_category_distribution": dict(size_distribution),
        
        "total_nodules": sum(nodule_counts),
        "nodule_size_breakdown": {
            "small (<6mm)": len(small_nodules),
            "medium (6-<10mm)": len(medium_nodules),
            "large (≥10mm)": len(large_nodules),
        },
        
        "nodule_size_stats": {
            "min_mm": float(min(all_sizes)) if all_sizes else 0,
            "max_mm": float(max(all_sizes)) if all_sizes else 0,
            "mean_mm": float(np.mean(all_sizes)) if all_sizes else 0,
            "median_mm": float(np.median(all_sizes)) if all_sizes else 0,
            "std_mm": float(np.std(all_sizes)) if all_sizes else 0,
        },
    }
    
    # === PRINT DETAILED REPORT ===
    
    print("\n" + "="*70)
    print("📊 ANALYSIS RESULTS")
    print("="*70)
    
    print("\n1️⃣  POSITIVE vs NEGATIVE SUBJECTS")
    print("-"*50)
    print(f"   Total subjects:     {summary['total_subjects']}")
    print(f"   Positive (nodules): {summary['positive_count']} ({summary['positive_ratio']*100:.1f}%)")
    print(f"   Negative (none):    {summary['negative_count']} ({(1-summary['positive_ratio'])*100:.1f}%)")
    
    print("\n2️⃣  NODULE COUNT DISTRIBUTION (among positive subjects)")
    print("-"*50)
    for count in sorted(count_distribution.keys()):
        freq = count_distribution[count]
        pct = freq / len(positive_cases) * 100 if positive_cases else 0
        bar = "█" * int(pct / 5)
        print(f"   {count} nodule(s): {freq:3d} subjects ({pct:5.1f}%) {bar}")
    
    print("\n   Summary:")
    print(f"      Single-nodule (1):    {single_nodule} subjects")
    print(f"      Multi-nodule (2-3):   {multi_nodule} subjects")
    print(f"      Many-nodule (4+):     {many_nodule} subjects")
    
    print("\n3️⃣  NODULE SIZE DISTRIBUTION")
    print("-"*50)
    print(f"   Total nodules found: {summary['total_nodules']}")
    print(f"\n   By size category (individual nodules):")
    print(f"      Small (<6mm):     {len(small_nodules)} nodules ({len(small_nodules)/len(all_sizes)*100:.1f}%)" if all_sizes else "      Small (<6mm):     0 nodules")
    print(f"      Medium (6-<10mm): {len(medium_nodules)} nodules ({len(medium_nodules)/len(all_sizes)*100:.1f}%)" if all_sizes else "      Medium (6-<10mm): 0 nodules")
    print(f"      Large (≥10mm):    {len(large_nodules)} nodules ({len(large_nodules)/len(all_sizes)*100:.1f}%)" if all_sizes else "      Large (≥10mm):    0 nodules")
    
    if all_sizes:
        print(f"\n   Size statistics:")
        print(f"      Min diameter:    {summary['nodule_size_stats']['min_mm']:.2f} mm")
        print(f"      Max diameter:    {summary['nodule_size_stats']['max_mm']:.2f} mm")
        print(f"      Mean diameter:   {summary['nodule_size_stats']['mean_mm']:.2f} mm")
        print(f"      Median diameter: {summary['nodule_size_stats']['median_mm']:.2f} mm")
        print(f"      Std deviation:   {summary['nodule_size_stats']['std_mm']:.2f} mm")
    
    print("\n4️⃣  SUBJECT SIZE CATEGORY (based on largest nodule)")
    print("-"*50)
    for cat in ["negative", "small", "medium", "large", "mixed", "error"]:
        if cat in size_distribution:
            freq = size_distribution[cat]
            pct = freq / len(analyses) * 100
            bar = "█" * int(pct / 5)
            print(f"   {cat:12s}: {freq:3d} subjects ({pct:5.1f}%) {bar}")
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE")
    print("="*70 + "\n")
    
    return {
        "subject_analyses": analyses,
        "summary": summary,
        "all_nodule_sizes": all_sizes,
    }


def plot_analysis(analysis_result: Dict, output_path: Optional[Path] = None):
    """Generate visualization plots for the analysis."""
    
    if not HAS_MATPLOTLIB:
        print("⚠️ matplotlib not available, skipping plots")
        return
    
    summary = analysis_result["summary"]
    all_sizes = analysis_result["all_nodule_sizes"]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Positive vs Negative pie chart
    ax1 = axes[0, 0]
    labels = ['Positive\n(with nodules)', 'Negative\n(no nodules)']
    sizes = [summary['positive_count'], summary['negative_count']]
    colors = ['#ff6b6b', '#4ecdc4']
    explode = (0.05, 0)
    ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.set_title('1. Positive vs Negative Subjects', fontsize=12, fontweight='bold')
    
    # 2. Nodule count distribution bar chart
    ax2 = axes[0, 1]
    count_dist = summary['nodule_count_distribution']
    if count_dist:
        counts = sorted(count_dist.keys())
        freqs = [count_dist[c] for c in counts]
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(counts)))
        bars = ax2.bar([str(c) for c in counts], freqs, color=colors, edgecolor='black')
        ax2.set_xlabel('Number of Nodules per Subject')
        ax2.set_ylabel('Number of Subjects')
        ax2.set_title('2. Nodule Count Distribution\n(among positive subjects)', fontsize=12, fontweight='bold')
        # Add value labels
        for bar, freq in zip(bars, freqs):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    str(freq), ha='center', va='bottom', fontsize=10)
    else:
        ax2.text(0.5, 0.5, 'No positive cases', ha='center', va='center', fontsize=14)
        ax2.set_title('2. Nodule Count Distribution', fontsize=12, fontweight='bold')
    
    # 3. Nodule size histogram
    ax3 = axes[1, 0]
    if all_sizes:
        ax3.hist(all_sizes, bins=20, color='#95a5a6', edgecolor='black', alpha=0.7)
        # Add vertical lines for size categories
        ax3.axvline(x=6, color='orange', linestyle='--', linewidth=2, label='6mm (small/medium)')
        ax3.axvline(x=10, color='red', linestyle='--', linewidth=2, label='10mm (medium/large)')
        ax3.set_xlabel('Nodule Diameter (mm)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('3. Nodule Size Distribution', fontsize=12, fontweight='bold')
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, 'No nodules found', ha='center', va='center', fontsize=14)
        ax3.set_title('3. Nodule Size Distribution', fontsize=12, fontweight='bold')
    
    # 4. Size category distribution
    ax4 = axes[1, 1]
    size_dist = summary['size_category_distribution']
    categories = ['negative', 'small', 'medium', 'large', 'mixed']
    colors_map = {'negative': '#4ecdc4', 'small': '#45b7d1', 'medium': '#f9ca24', 'large': '#ff6b6b', 'mixed': '#a55eea'}
    cat_counts = [size_dist.get(c, 0) for c in categories]
    cat_colors = [colors_map[c] for c in categories]
    bars = ax4.bar(categories, cat_counts, color=cat_colors, edgecolor='black')
    ax4.set_xlabel('Size Category')
    ax4.set_ylabel('Number of Subjects')
    ax4.set_title('4. Subject Size Category\n(based on largest nodule)', fontsize=12, fontweight='bold')
    # Add value labels
    for bar, count in zip(bars, cat_counts):
        if count > 0:
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    str(count), ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"📊 Plot saved to: {output_path}")
    
    plt.show()


def save_analysis(analysis_result: Dict, output_path: Path):
    """Save analysis results to JSON file."""
    
    # Prepare serializable summary
    output = {
        "summary": analysis_result["summary"],
        "subjects": []
    }
    
    for a in analysis_result["subject_analyses"]:
        subject_info = {
            "subject_id": a["subject_id"],
            "has_nodules": a["has_nodules"],
            "num_nodules": a["num_nodules"],
            "size_category": a["size_category"],
            "max_nodule_diameter_mm": a.get("max_nodule_diameter_mm", 0),
            "nodule_sizes_mm": a.get("nodule_sizes_mm", []),
        }
        output["subjects"].append(subject_info)
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"💾 Analysis saved to: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze LUNA16 dataset for stratified selection")
    parser.add_argument(
        "--bids-root",
        type=str,
        default="./data/luna16_processed",
        help="Path to BIDS-formatted LUNA16 dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/luna16_processed",
        help="Directory to save analysis results"
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating plots"
    )
    
    args = parser.parse_args()
    
    bids_root = Path(args.bids_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    data_dicts = read_luna16_dataset(bids_root, verbose=True)
    
    if not data_dicts:
        print("❌ No data found!")
        return
    
    # Analyze
    analysis_result = analyze_full_dataset(data_dicts, verbose=True)
    
    # Save results
    save_analysis(analysis_result, output_dir / "dataset_analysis.json")
    
    # Plot
    if not args.no_plot and HAS_MATPLOTLIB:
        plot_analysis(analysis_result, output_dir / "dataset_analysis.png")
    
    print("\n✅ Analysis complete!")
    print(f"   Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

