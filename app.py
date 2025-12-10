"""
Medical Imaging AI Pipeline - Main Entry Point
Uses Streamlit's native multi-page app system.
Pages are auto-detected from the pages/ directory.
"""

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="MedAI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Landing page content
st.title("🏥 Medical Imaging AI")
st.markdown("### MONAI-Powered Pipeline for CT Analysis")

st.markdown("""
Welcome to the Medical Imaging AI Pipeline. Use the sidebar to navigate:

- **📁 Data** - Import, browse, and view medical images
- **🔬 Analyze** - Run AI models for nodule detection

---
""")

# Quick status
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Quick Start")
    st.markdown("""
    1. Go to **Data** → Import your DICOM files
    2. Browse and view your processed images
    3. Go to **Analyze** → Run inference
    """)

with col2:
    st.markdown("#### System Status")
    
    # Check datasets
    from pathlib import Path
    data_dir = Path("./data")
    datasets = []
    if data_dir.exists():
        for subdir in data_dir.iterdir():
            if subdir.is_dir() and (subdir / "dataset_description.json").exists():
                datasets.append(subdir.name)
    
    if datasets:
        st.success(f"✅ {len(datasets)} dataset(s) available")
    else:
        st.info("No datasets yet")
    
    # Check models
    models_dir = Path("./models")
    model_count = 0
    if models_dir.exists():
        for item in models_dir.iterdir():
            if item.is_dir():
                if (item / "config.json").exists() and (item / "best_model.pth").exists():
                    model_count += 1
                for subitem in item.iterdir():
                    if subitem.is_dir() and (subitem / "config.json").exists() and (subitem / "best_model.pth").exists():
                        model_count += 1
    
    if model_count > 0:
        st.success(f"✅ {model_count} model(s) available")
    else:
        st.warning("No trained models")

st.markdown("---")
st.caption("MONAI Pipeline • 2025")
