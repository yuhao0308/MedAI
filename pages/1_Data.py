"""
Data Page - Import, Browse, and View Medical Images
Tabs: Import | Browse | View
"""

import streamlit as st
import sys
from pathlib import Path
import os
import tempfile
import shutil
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.dicom_parser import get_series_metadata
from src.ingestion.dicom_to_nifti import convert_dicom_to_nifti
from src.ingestion.bids_organizer import (
    create_bids_structure,
    create_dataset_description,
    create_participants_tsv,
    organize_nifti_to_bids
)
from src.frontend import get_available_datasets, get_labels_dir, get_subjects, count_files


# ============================================================
# PROCESSING FUNCTIONS
# ============================================================

def process_luna16_data(max_scans: int = 50):
    """Process LUNA16 raw data"""
    import SimpleITK as sitk
    import nibabel as nib
    
    luna16_raw_dir = Path("./data/luna16_raw")
    luna16_processed_dir = Path("./data/luna16_processed")
    
    subset_dir = luna16_raw_dir / "subset0"
    annotations_path = luna16_raw_dir / "annotations.csv"
    
    if not subset_dir.exists():
        st.error(f"❌ Subset not found: {subset_dir}")
        return
    
    if not annotations_path.exists():
        st.error(f"❌ Annotations not found")
        return
    
    progress_bar = st.progress(0, text="Reading annotations...")
    annotations_df = pd.read_csv(annotations_path)
    
    mhd_files = sorted(list(subset_dir.glob("*.mhd")))[:max_scans]
    if not mhd_files:
        st.error("❌ No .mhd files found")
        return
    
    luna16_processed_dir.mkdir(parents=True, exist_ok=True)
    create_luna16_bids_structure(luna16_processed_dir)
    
    participants_info = []
    successful = 0
    
    for idx, mhd_path in enumerate(mhd_files):
        subject_id = f"{idx+1:03d}"
        seriesuid = mhd_path.stem
        
        progress = int(100 * (idx + 1) / len(mhd_files))
        progress_bar.progress(progress / 100, text=f"Processing {idx+1}/{len(mhd_files)}")
        
        try:
            sitk_image = sitk.ReadImage(str(mhd_path))
            resampled_image = resample_image(sitk_image, (1.5, 1.5, 1.5))
            
            image_dir = luna16_processed_dir / f"sub-{subject_id}" / "anat"
            label_dir = luna16_processed_dir / "derivatives" / "labels" / f"sub-{subject_id}" / "anat"
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            
            array, affine = sitk_to_nifti_data(resampled_image)
            nifti_img = nib.Nifti1Image(array, affine)
            nib.save(nifti_img, str(image_dir / f"sub-{subject_id}_CT.nii.gz"))
            
            scan_annotations = annotations_df[annotations_df['seriesuid'] == seriesuid]
            mask = create_nodule_mask(array.shape, resampled_image.GetSpacing(), 
                                     resampled_image.GetOrigin(), scan_annotations)
            
            mask_nifti = nib.Nifti1Image(mask, affine)
            nib.save(mask_nifti, str(label_dir / f"sub-{subject_id}_CT_seg-nodule_mask.nii.gz"))
            
            participants_info.append({
                "participant_id": f"sub-{subject_id}",
                "seriesuid": seriesuid,
                "nodule_count": len(scan_annotations)
            })
            successful += 1
        except Exception:
            continue
    
    if participants_info:
        pd.DataFrame(participants_info).to_csv(luna16_processed_dir / "participants.tsv", sep='\t', index=False)
    
    progress_bar.progress(1.0, text="Complete!")
    st.success(f"✅ Processed {successful}/{len(mhd_files)} scans")


def create_luna16_bids_structure(output_dir: Path):
    """Create BIDS structure for LUNA16"""
    dataset_desc = {
        "Name": "LUNA16 Lung Nodule Dataset",
        "BIDSVersion": "1.6.0",
        "DatasetType": "raw"
    }
    with open(output_dir / "dataset_description.json", 'w') as f:
        json.dump(dataset_desc, f, indent=2)
    
    derivatives_dir = output_dir / "derivatives" / "labels"
    derivatives_dir.mkdir(parents=True, exist_ok=True)
    with open(derivatives_dir / "dataset_description.json", 'w') as f:
        json.dump({"Name": "LUNA16 Labels", "BIDSVersion": "1.6.0", "DatasetType": "derivative"}, f, indent=2)


def resample_image(sitk_image, new_spacing):
    """Resample SimpleITK image"""
    import SimpleITK as sitk
    
    original_spacing = sitk_image.GetSpacing()
    original_size = sitk_image.GetSize()
    new_size = [int(round(osz * osp / nsp)) for osz, osp, nsp in zip(original_size, original_spacing, new_spacing)]
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(sitk_image.GetDirection())
    resampler.SetOutputOrigin(sitk_image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(sitk_image.GetPixelIDValue())
    resampler.SetInterpolator(sitk.sitkLinear)
    return resampler.Execute(sitk_image)


def sitk_to_nifti_data(sitk_image):
    """Convert SimpleITK to numpy + affine"""
    import SimpleITK as sitk
    
    array = sitk.GetArrayFromImage(sitk_image)
    spacing = sitk_image.GetSpacing()
    origin = sitk_image.GetOrigin()
    
    affine = np.eye(4)
    affine[0, 0], affine[1, 1], affine[2, 2] = spacing
    affine[0, 3], affine[1, 3], affine[2, 3] = origin
    return array, affine


def create_nodule_mask(image_shape, spacing, origin, nodule_annotations):
    """Create binary nodule mask"""
    mask = np.zeros(image_shape, dtype=np.uint8)
    
    for _, nodule in nodule_annotations.iterrows():
        world_x, world_y, world_z = nodule['coordX'], nodule['coordY'], nodule['coordZ']
        diameter = nodule['diameter_mm']
        
        voxel_x = int(round((world_x - origin[0]) / spacing[0]))
        voxel_y = int(round((world_y - origin[1]) / spacing[1]))
        voxel_z = int(round((world_z - origin[2]) / spacing[2]))
        
        radius_mm = diameter / 2.0
        radius_x = int(np.ceil(radius_mm / spacing[0]))
        radius_y = int(np.ceil(radius_mm / spacing[1]))
        radius_z = int(np.ceil(radius_mm / spacing[2]))
        
        z_min, z_max = max(0, voxel_z - radius_z), min(image_shape[0], voxel_z + radius_z + 1)
        y_min, y_max = max(0, voxel_y - radius_y), min(image_shape[1], voxel_y + radius_y + 1)
        x_min, x_max = max(0, voxel_x - radius_x), min(image_shape[2], voxel_x + radius_x + 1)
        
        for z in range(z_min, z_max):
            for y in range(y_min, y_max):
                for x in range(x_min, x_max):
                    dist_sq = ((x - voxel_x) * spacing[0])**2 + ((y - voxel_y) * spacing[1])**2 + ((z - voxel_z) * spacing[2])**2
                    if dist_sq <= radius_mm ** 2:
                        mask[z, y, x] = 1
    return mask


def process_dicom_directory(dicom_dir: str):
    """Process DICOM directory"""
    dicom_path = Path(dicom_dir)
    if not dicom_path.exists():
        raise ValueError(f"Directory not found: {dicom_dir}")
    
    patient_folders = [d for d in dicom_path.iterdir() if d.is_dir() and d.name.startswith("Thrx-CT")]
    
    if patient_folders:
        st.info(f"Detected thorax CT ({len(patient_folders)} patients)")
        return process_thorax_structure(dicom_dir)
    else:
        st.info("Using generic DICOM processing")
        return process_generic_dicom(dicom_dir)


def process_thorax_structure(dicom_dir: str):
    """Process thorax CT dataset"""
    from src.ingestion.bids_organizer import create_bids_readme
    
    bids_root = "./data/bids_dataset"
    dicom_path = Path(dicom_dir)
    
    patient_folders = sorted([d for d in dicom_path.iterdir() if d.is_dir() and d.name.startswith("Thrx-CT")])
    if not patient_folders:
        raise ValueError("No Thrx-CT folders found")
    
    create_bids_structure(bids_root)
    create_dataset_description(bids_root, dataset_name="Thorax CT Dataset")
    create_bids_readme(bids_root)
    
    temp_nifti_dir = Path(bids_root).parent / "temp_nifti"
    temp_nifti_dir.mkdir(exist_ok=True, parents=True)
    
    participants_data = []
    successful = 0
    progress_bar = st.progress(0)
    
    for idx, patient_folder in enumerate(patient_folders):
        patient_id = f"{idx+1:03d}"
        patient_name = patient_folder.name
        progress_bar.progress((idx + 1) / len(patient_folders), text=f"Processing {patient_name}")
        
        try:
            image_dir = patient_folder / "dicom" / patient_name / "image"
            if not image_dir.exists():
                continue
            
            dicom_files = list(image_dir.glob("*.dcm")) + list(image_dir.glob("*.DCM"))
            if not dicom_files:
                continue
            
            nifti_image = convert_dicom_to_nifti(str(image_dir), str(temp_nifti_dir), f"sub-{patient_id}_CT", True)
            organize_nifti_to_bids(nifti_image, bids_root, patient_id, "anat", "CT")
            
            mask_dir = patient_folder / "dicom" / patient_name / "mask"
            if mask_dir.exists():
                for mask_subdir in [d for d in mask_dir.iterdir() if d.is_dir()]:
                    mask_dicom_files = list(mask_subdir.glob("*.dcm")) + list(mask_subdir.glob("*.DCM"))
                    if mask_dicom_files:
                        try:
                            nifti_mask = convert_dicom_to_nifti(str(mask_subdir), str(temp_nifti_dir), 
                                                               f"sub-{patient_id}_CT_seg-{mask_subdir.name}_mask", True)
                            bids_mask_dir = Path(bids_root) / "derivatives" / "labels" / f"sub-{patient_id}" / "anat"
                            bids_mask_dir.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(nifti_mask, bids_mask_dir / f"sub-{patient_id}_CT_seg-{mask_subdir.name}_mask.nii.gz")
                        except Exception:
                            continue
            
            participants_data.append({"participant_id": f"sub-{patient_id}", "original_id": patient_name})
            successful += 1
        except Exception:
            continue
    
    if participants_data:
        create_participants_tsv(bids_root, participants_data)
    if temp_nifti_dir.exists():
        shutil.rmtree(temp_nifti_dir)
    
    progress_bar.empty()
    return successful


def process_generic_dicom(dicom_dir: str):
    """Process generic DICOM"""
    series_map = get_series_metadata(dicom_dir)
    
    if not series_map:
        dicom_path = Path(dicom_dir)
        dicom_files = list(dicom_path.rglob("*.dcm")) + list(dicom_path.rglob("*.DCM"))
        if not dicom_files:
            raise ValueError("No DICOM files found")
        series_map = get_series_metadata(os.path.dirname(dicom_files[0]))
        if not series_map:
            raise ValueError("Could not parse DICOM series")
    
    bids_root = "./data/bids_dataset"
    create_bids_structure(bids_root)
    create_dataset_description(bids_root)
    
    participants_data = []
    processed_count = 0
    
    for series_uid, metadata in series_map.items():
        if not metadata.get("files"):
            continue
        
        participant_id = f"{processed_count + 1:03d}"
        series_dir = os.path.dirname(metadata["files"][0])
        
        try:
            temp_nifti_dir = "./data/temp_nifti"
            os.makedirs(temp_nifti_dir, exist_ok=True)
            nifti_path = convert_dicom_to_nifti(series_dir, temp_nifti_dir, f"sub-{participant_id}_CT", True)
            organize_nifti_to_bids(nifti_path, bids_root, participant_id, "anat", "CT")
            participants_data.append({"participant_id": f"sub-{participant_id}", "modality": metadata.get("modality", "UNKNOWN")})
            processed_count += 1
        except Exception:
            continue
    
    if participants_data:
        create_participants_tsv(bids_root, participants_data)
    return processed_count


# ============================================================
# IMPORT TAB FUNCTIONS
# ============================================================

def render_import_tab():
    """Combined import interface"""
    
    import_type = st.radio(
        "Data Source",
        ["Upload Files", "Local Directory", "LUNA16 Dataset"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    if import_type == "Upload Files":
        render_upload_section()
    elif import_type == "Local Directory":
        render_directory_section()
    else:
        render_luna16_section()


def render_upload_section():
    """File upload interface"""
    st.markdown("### Upload DICOM Files")
    
    uploaded_files = st.file_uploader(
        "Select DICOM files",
        type=['dcm', 'DCM'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.info(f"📁 {len(uploaded_files)} files selected")
        
        if st.button("Process Uploaded Files", type="primary"):
            with st.spinner("Processing..."):
                try:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        temp_path = Path(temp_dir)
                        for uploaded_file in uploaded_files:
                            file_path = temp_path / uploaded_file.name
                            with open(file_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                        process_dicom_directory(str(temp_path))
                        st.success("✅ Files processed!")
                        st.balloons()
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")


def render_directory_section():
    """Directory processing interface"""
    st.markdown("### Process Local DICOM Directory")
    
    common_dirs = ["./New_thorax_ct_dicom", "./data/raw_dicom", "./MyHead"]
    existing_dirs = [d for d in common_dirs if Path(d).exists()]
    
    if existing_dirs:
        selected_dir = st.selectbox("Found directories", existing_dirs)
        
        if st.button("Process Selected Directory", type="primary"):
            with st.spinner(f"Processing {selected_dir}..."):
                try:
                    process_dicom_directory(selected_dir)
                    st.success("✅ Processed!")
                    st.balloons()
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    with st.expander("Enter custom path"):
        custom_path = st.text_input("DICOM Directory Path", placeholder="./path/to/dicom/files")
        if custom_path and Path(custom_path).exists():
            if st.button("Process Custom Directory", type="primary"):
                with st.spinner(f"Processing {custom_path}..."):
                    try:
                        process_dicom_directory(custom_path)
                        st.success("✅ Processed!")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")


def render_luna16_section():
    """LUNA16 dataset processing"""
    st.markdown("### LUNA16 Lung Nodule Dataset")
    
    luna16_raw_dir = Path("./data/luna16_raw")
    luna16_processed_dir = Path("./data/luna16_processed")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if luna16_raw_dir.exists():
            mhd_files = list(luna16_raw_dir.rglob("*.mhd"))
            st.success(f"✅ Raw data found ({len(mhd_files)} scans)")
        else:
            st.warning("⚠️ Raw data not found")
    
    with col2:
        if luna16_processed_dir.exists():
            subjects = [d for d in luna16_processed_dir.iterdir() 
                       if d.is_dir() and d.name.startswith("sub-")]
            st.success(f"✅ Processed: {len(subjects)} subjects")
        else:
            st.info("Not yet processed")
    
    st.markdown("---")
    
    if luna16_raw_dir.exists() and not luna16_processed_dir.exists():
        max_scans = st.slider("Maximum scans to process", 10, 100, 50)
        if st.button("🚀 Process LUNA16 Data", type="primary", use_container_width=True):
            process_luna16_data(max_scans=max_scans)
    elif luna16_processed_dir.exists():
        st.success("✅ LUNA16 data is ready. Use **Browse** or **View** tabs.")
        with st.expander("Reprocess"):
            max_scans = st.slider("Max scans", 10, 100, 50, key="reprocess_max")
            if st.button("Reprocess"):
                shutil.rmtree(luna16_processed_dir)
                process_luna16_data(max_scans=max_scans)
    else:
        st.info("Download LUNA16 from [luna16.grand-challenge.org](https://luna16.grand-challenge.org/)")


# ============================================================
# BROWSE TAB FUNCTIONS
# ============================================================

def render_browse_tab():
    """Dataset browser interface"""
    
    datasets = get_available_datasets()
    
    if not datasets:
        st.info("No datasets found. Use the **Import** tab to add data.")
        return
    
    selected_dataset = st.selectbox("Select Dataset", datasets, format_func=lambda x: x.name, key="browse_ds")
    
    if not selected_dataset or not selected_dataset.exists():
        return
    
    # Metrics
    subjects = get_subjects(selected_dataset)
    images = count_files(selected_dataset, "*.nii.gz", exclude="*_mask*")
    labels_dir = get_labels_dir(selected_dataset)
    masks = count_files(labels_dir, "*_mask.nii.gz")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Subjects", len(subjects))
    col2.metric("Images", images)
    col3.metric("Masks", masks)
    
    st.markdown("---")
    
    if subjects:
        selected_subject = st.selectbox("Select Subject", subjects, format_func=lambda x: x.name, key="browse_subj")
        if selected_subject:
            display_subject_details(selected_subject, selected_dataset)
    else:
        st.info("No subjects found")


def display_subject_details(subject_dir: Path, bids_dir: Path):
    """Display subject details"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        anat_dir = subject_dir / "anat"
        if anat_dir.exists():
            st.markdown("#### Images")
            images = list(anat_dir.glob("*.nii.gz"))
            for img in images:
                size_mb = img.stat().st_size / (1024 * 1024)
                st.text(f"• {img.name} ({size_mb:.1f} MB)")
        else:
            st.info("No images found")
    
    with col2:
        labels_dir = get_labels_dir(bids_dir) / subject_dir.name / "anat"
        if labels_dir.exists():
            masks = list(labels_dir.glob("*_mask.nii.gz"))
            if masks:
                st.markdown("#### Masks")
                for mask in masks:
                    size_mb = mask.stat().st_size / (1024 * 1024)
                    organ = mask.name.split("seg-")[1].split("_mask")[0] if "seg-" in mask.name else "Unknown"
                    st.text(f"• {organ} ({size_mb:.2f} MB)")
            else:
                st.info("No masks found")
        else:
            st.info("No masks found")
    
    with st.expander("Metadata"):
        participants_file = bids_dir / "participants.tsv"
        if participants_file.exists():
            try:
                df = pd.read_csv(participants_file, sep="\t")
                if subject_dir.name in df["participant_id"].values:
                    subject_data = df[df["participant_id"] == subject_dir.name].iloc[0]
                    st.json(subject_data.to_dict())
            except Exception as e:
                st.caption(f"Could not load: {e}")


# ============================================================
# VIEW TAB FUNCTIONS
# ============================================================

def render_view_tab():
    """Slice viewer with windowing controls"""
    import nibabel as nib
    
    datasets = get_available_datasets()
    
    if not datasets:
        st.info("No datasets found. Use the **Import** tab to add data.")
        return
    
    # Dataset selector
    col1, col2 = st.columns(2)
    
    with col1:
        selected_dataset = st.selectbox(
            "Dataset", 
            datasets, 
            format_func=lambda x: x.name,
            key="view_dataset"
        )
    
    if not selected_dataset:
        return
    
    subjects = get_subjects(selected_dataset)
    if not subjects:
        st.info("No subjects in this dataset")
        return
    
    with col2:
        selected_subject = st.selectbox(
            "Subject",
            subjects,
            format_func=lambda x: x.name,
            key="view_subject"
        )
    
    if not selected_subject:
        return
    
    # Find images
    anat_dir = selected_subject / "anat"
    if not anat_dir.exists():
        st.warning("No images found")
        return
    
    images = list(anat_dir.glob("*.nii.gz"))
    if not images:
        st.warning("No NIfTI images found")
        return
    
    selected_image = st.selectbox(
        "Image",
        images,
        format_func=lambda x: x.name,
        key="view_image"
    )
    
    if not selected_image:
        return
    
    # Load image
    try:
        img = nib.load(str(selected_image))
        img_data = img.get_fdata()
        
        # Image info
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Shape", f"{img_data.shape[0]}×{img_data.shape[1]}×{img_data.shape[2]}")
        col2.metric("Spacing", f"{img.header.get_zooms()[:3]}")
        col3.metric("Min", f"{img_data.min():.0f}")
        col4.metric("Max", f"{img_data.max():.0f}")
        
        st.markdown("---")
        
        # View controls
        col1, col2 = st.columns([1, 2])
        
        with col1:
            view_type = st.radio("View", ["Axial", "Coronal", "Sagittal"], key="view_type")
            
            # Window presets
            st.markdown("**CT Window**")
            preset = st.selectbox(
                "Preset",
                ["Lung", "Mediastinum", "Bone", "Soft Tissue", "Custom"],
                key="window_preset"
            )
            
            presets = {
                "Lung": (1500, -600),
                "Mediastinum": (400, 40),
                "Bone": (2000, 300),
                "Soft Tissue": (400, 50),
            }
            
            if preset == "Custom":
                window = st.number_input("Window (W)", 1, 4000, 1500, key="window_w")
                level = st.number_input("Level (L)", -2000, 2000, -600, key="window_l")
            else:
                window, level = presets.get(preset, (1500, -600))
                st.caption(f"W: {window}, L: {level}")
        
        with col2:
            # Slice navigation
            if view_type == "Axial":
                max_slice = img_data.shape[2] - 1
                axis = 2
            elif view_type == "Coronal":
                max_slice = img_data.shape[1] - 1
                axis = 1
            else:
                max_slice = img_data.shape[0] - 1
                axis = 0
            
            slice_idx = st.slider("Slice", 0, max_slice, max_slice // 2, key="slice_idx")
            
            # Extract slice
            if axis == 2:
                img_slice = img_data[:, :, slice_idx]
            elif axis == 1:
                img_slice = img_data[:, slice_idx, :]
            else:
                img_slice = img_data[slice_idx, :, :]
            
            # Apply windowing
            min_val = level - window / 2
            max_val = level + window / 2
            img_windowed = np.clip(img_slice, min_val, max_val)
            img_windowed = (img_windowed - min_val) / (max_val - min_val)
            
            # Display
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(img_windowed.T, cmap="gray", origin="lower")
            ax.set_title(f"{view_type} - Slice {slice_idx}")
            ax.axis("off")
            st.pyplot(fig)
            plt.close(fig)
        
        # Mask overlay option
        labels_dir = get_labels_dir(selected_dataset) / selected_subject.name / "anat"
        if labels_dir.exists():
            masks = list(labels_dir.glob("*_mask.nii.gz"))
            if masks:
                st.markdown("---")
                st.markdown("#### Segmentation Overlay")
                
                selected_mask = st.selectbox(
                    "Mask",
                    [None] + masks,
                    format_func=lambda x: "None" if x is None else (x.name.split("seg-")[1].split("_mask")[0] if "seg-" in x.name else x.name),
                    key="view_mask"
                )
                
                if selected_mask:
                    mask_img = nib.load(str(selected_mask))
                    mask_data = mask_img.get_fdata()
                    
                    if axis == 2:
                        mask_slice = mask_data[:, :, slice_idx]
                    elif axis == 1:
                        mask_slice = mask_data[:, slice_idx, :]
                    else:
                        mask_slice = mask_data[slice_idx, :, :]
                    
                    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
                    
                    axes[0].imshow(img_windowed.T, cmap="gray", origin="lower")
                    axes[0].set_title("Image")
                    axes[0].axis("off")
                    
                    axes[1].imshow(img_windowed.T, cmap="gray", origin="lower")
                    mask_overlay = np.ma.masked_where(mask_slice == 0, mask_slice)
                    axes[1].imshow(mask_overlay.T, cmap="jet", alpha=0.5, origin="lower")
                    axes[1].set_title("With Mask Overlay")
                    axes[1].axis("off")
                    
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    # Mask stats
                    mask_voxels = int(np.sum(mask_slice > 0))
                    st.caption(f"Mask voxels in slice: {mask_voxels:,}")
        
    except Exception as e:
        st.error(f"Error loading image: {e}")


# ============================================================
# PAGE ENTRY POINT
# ============================================================

# Page config
st.set_page_config(page_title="Data - MedAI", page_icon="📁", layout="wide")

st.title("📁 Data")

# Three main tabs: Import, Browse, View
tab1, tab2, tab3 = st.tabs(["Import", "Browse", "View"])

with tab1:
    render_import_tab()

with tab2:
    render_browse_tab()

with tab3:
    render_view_tab()
