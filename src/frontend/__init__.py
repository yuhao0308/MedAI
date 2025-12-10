"""Frontend shared utilities for Streamlit pages"""

import streamlit as st
from pathlib import Path
from typing import List, Optional
import json


def get_available_datasets() -> List[Path]:
    """
    Detect available BIDS datasets in the data directory.
    A valid dataset must have a dataset_description.json file.
    """
    data_dir = Path("./data")
    datasets = []
    
    if not data_dir.exists():
        return datasets
    
    for subdir in sorted(data_dir.iterdir()):
        if subdir.is_dir():
            if (subdir / "dataset_description.json").exists():
                datasets.append(subdir)
    
    return datasets


def get_dataset_info(dataset_path: Path) -> dict:
    """Get information about a dataset from its description file."""
    info = {
        "name": dataset_path.name,
        "path": dataset_path,
        "subjects": 0,
        "description": ""
    }
    
    subjects = [d for d in dataset_path.iterdir() 
                if d.is_dir() and d.name.startswith("sub-")]
    info["subjects"] = len(subjects)
    
    desc_file = dataset_path / "dataset_description.json"
    if desc_file.exists():
        try:
            with open(desc_file, 'r') as f:
                desc = json.load(f)
                info["description"] = desc.get("Description", desc.get("Name", ""))
        except Exception:
            pass
    
    return info


def dataset_selector(key: str = "dataset_selector") -> Optional[Path]:
    """
    Render a compact dataset selector in the sidebar.
    Used by pages that need to select a dataset.
    """
    datasets = get_available_datasets()
    
    if not datasets:
        return None
    
    # Initialize session state
    if "selected_dataset" not in st.session_state:
        st.session_state.selected_dataset = str(datasets[0])
    
    # Ensure stored dataset still exists
    stored_path = Path(st.session_state.selected_dataset)
    if stored_path not in datasets:
        st.session_state.selected_dataset = str(datasets[0])
    
    # Get current index
    current_idx = 0
    for i, ds in enumerate(datasets):
        if str(ds) == st.session_state.selected_dataset:
            current_idx = i
            break
    
    # Render selector in sidebar
    st.sidebar.markdown("**Dataset**")
    
    selected = st.sidebar.selectbox(
        "Dataset",
        datasets,
        index=current_idx,
        format_func=lambda x: x.name,
        key=key,
        label_visibility="collapsed"
    )
    
    # Update session state
    st.session_state.selected_dataset = str(selected)
    
    # Show subject count
    info = get_dataset_info(selected)
    st.sidebar.caption(f"{info['subjects']} subjects")
    
    return selected


def get_labels_dir(dataset_path: Path) -> Path:
    """Get the labels/derivatives directory for a dataset."""
    return dataset_path / "derivatives" / "labels"


def get_subjects(bids_dir: Path) -> List[Path]:
    """Get list of subject directories from a BIDS dataset."""
    return sorted([d for d in bids_dir.iterdir() 
            if d.is_dir() and d.name.startswith("sub-")])


def count_files(directory: Path, pattern: str, exclude: str = None) -> int:
    """Count files matching pattern in directory."""
    if not directory.exists():
        return 0
    
    files = list(directory.rglob(pattern))
    if exclude:
        files = [f for f in files if exclude not in str(f)]
    return len(files)


