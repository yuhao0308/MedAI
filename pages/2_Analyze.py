"""
Analyze Page - Run AI Models on Medical Images
Tabs: Inference | History
"""

import streamlit as st
import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import time

# Page config - must be first Streamlit command
st.set_page_config(page_title="Analyze - MedAI", page_icon="🔬", layout="wide")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
    Orientationd, ScaleIntensityRanged, CropForegroundd,
    DivisiblePadd, ToTensord
)

from src.frontend import dataset_selector

# Import postprocessing module
try:
    from src.modeling.postprocessing import (
        postprocess_segmentation,
        NoduleCandidate,
        compute_aggregate_statistics
    )
    POSTPROCESSING_AVAILABLE = True
except ImportError:
    POSTPROCESSING_AVAILABLE = False

# Import uncertainty module
try:
    from src.modeling.uncertainty import (
        monte_carlo_dropout_inference,
        get_final_segmentation,
        compute_uncertainty_metrics
    )
    UNCERTAINTY_AVAILABLE = True
except ImportError:
    UNCERTAINTY_AVAILABLE = False

# Import report generator
try:
    from src.visualization.nodule_report import (
        generate_nodule_report,
        save_report
    )
    REPORT_AVAILABLE = True
except ImportError:
    REPORT_AVAILABLE = False

# Import inference history
try:
    from src.modeling.inference_history import get_history_manager
    HISTORY_AVAILABLE = True
except ImportError:
    HISTORY_AVAILABLE = False

# Import cloud inference service
try:
    from src.modeling.inference_service import (
        InferenceService,
        InferenceMode,
        get_inference_service
    )
    from src.cloud.base import InferenceSource
    CLOUD_AVAILABLE = True
except ImportError:
    CLOUD_AVAILABLE = False
    InferenceMode = None
    InferenceSource = None


def get_cloud_status() -> Dict[str, Any]:
    """Get current cloud inference status"""
    if not CLOUD_AVAILABLE:
        return {"available": False, "reason": "Cloud module not installed"}
    
    try:
        service = get_inference_service()
        return service.get_status()
    except Exception as e:
        return {"available": False, "reason": str(e)}


def render():
    st.title("Analyze Images")
    
    # Single dataset selector for all tabs
    bids_dir = dataset_selector(key="analyze_dataset")
    
    # Tab selection
    tab1, tab2 = st.tabs(["Run Inference", "History"])
    
    with tab1:
        render_inference_tab(bids_dir)
    
    with tab2:
        render_history_tab(bids_dir)


def render_inference_tab(bids_dir):
    """Render the main inference tab"""
    if bids_dir is None:
        st.warning("⚠️ No BIDS datasets found. Please process data first.")
        return
    
    if not bids_dir.exists():
        st.warning("⚠️ Selected dataset not found. Please process DICOM files first.")
        return
    
    # ============ CLOUD STATUS INDICATOR ============
    if CLOUD_AVAILABLE:
        cloud_status = get_cloud_status()
        
        with st.expander("☁️ Cloud Inference Status", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if cloud_status.get("cloud_available"):
                    st.success("✅ Cloud Ready")
                else:
                    st.warning("⚠️ Cloud Unavailable")
            
            with col2:
                local_device = cloud_status.get("local_device", "cpu")
                cuda_available = cloud_status.get("cuda_available", False)
                if cuda_available:
                    st.info(f"🖥️ Local: {local_device}")
                else:
                    st.info(f"🖥️ Local: CPU")
            
            with col3:
                stats = cloud_status.get("statistics", {})
                total = stats.get("total_inferences", 0)
                cloud_count = stats.get("cloud_inferences", 0)
                st.metric("Session Stats", f"{cloud_count}/{total} cloud")
            
            # Detailed status
            if cloud_status.get("cloud_available"):
                provider_status = cloud_status.get("cloud_provider", {})
                health = provider_status.get("health", {})
                if health.get("cuda_available"):
                    st.caption(f"☁️ Modal GPU: {health.get('cuda_device', 'T4')}")
            else:
                reason = cloud_status.get("reason", "Unknown")
                st.caption(f"Reason: {reason}")
                st.info("💡 To enable cloud inference:\n1. `pip install modal`\n2. `modal token new`")
    
    # Check for trained models
    models_dir = Path("./models")
    has_models = models_dir.exists() and any(models_dir.iterdir())
    
    if not has_models:
        st.warning("⚠️ No trained models found.")
        st.info("""
        To train a model:
        1. Use the Kaggle notebook: `notebooks/04_kaggle_luna16_training.ipynb`
        2. Download trained model and place in `models/` directory
        3. Ensure model directory contains `best_model.pth` and `config.json`
        """)
        return
    
    # Model selector
    models = list_available_models()
    
    if not models:
        st.warning("⚠️ No compatible models found. Models must have `config.json` and `best_model.pth`.")
        return
    
    selected_model = st.selectbox(
        "🧠 Select Model",
        models,
        format_func=lambda x: f"{x['name']} ({x['type']})"
    )
    
    # Show model info
    if selected_model:
        with st.expander("📋 Model Information", expanded=False):
            display_model_info(selected_model)
    
    st.markdown("---")
    
    # Subject selector
    subjects = sorted([d for d in bids_dir.iterdir() 
                if d.is_dir() and d.name.startswith("sub-")])
    
    if not subjects:
        st.info("No subjects found in selected dataset.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_subject = st.selectbox(
            "👤 Select Subject",
            subjects,
            format_func=lambda x: x.name
        )
    
    # Image selector
    anat_dir = selected_subject / "anat"
    if not anat_dir.exists():
        st.warning("No images found for this subject")
        return
    
    images = list(anat_dir.glob("*.nii.gz"))
    
    with col2:
        selected_image = st.selectbox(
            "🖼️ Select Image",
            images,
            format_func=lambda x: x.name
        )
    
    if selected_image and selected_model:
        st.markdown("---")
        
        # Default inference options (simplified)
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            save_prediction = st.checkbox("💾 Save prediction", value=True)
        with col2:
            overlap = st.select_slider("Overlap", options=[0.25, 0.5, 0.75], value=0.5)
        with col3:
            # Cloud inference toggle
            if CLOUD_AVAILABLE:
                cloud_status = get_cloud_status()
                use_cloud = st.checkbox(
                    "☁️ Use Cloud GPU",
                    value=cloud_status.get("cloud_available", False),
                    disabled=not cloud_status.get("cloud_available", False),
                    help="Run inference on Modal's serverless GPU for faster processing"
                )
            else:
                use_cloud = False
        
        # Advanced options in expander
        with st.expander("Advanced Options"):
            col1, col2 = st.columns(2)
            
            with col1:
                sw_batch_size = st.selectbox("SW batch size", [1, 2, 4], index=1)
                min_nodule_size = st.slider("Min nodule size (mm)", 1.0, 10.0, 3.0, 0.5)
            
            with col2:
                confidence_threshold = st.slider("Confidence threshold", 0.1, 0.9, 0.3, 0.1)
                enable_uncertainty = st.checkbox(
                    "Uncertainty estimation",
                    value=False,
                    disabled=not UNCERTAINTY_AVAILABLE or use_cloud  # Uncertainty not supported on cloud yet
                )
            
            num_mc_passes = 10
            if enable_uncertainty:
                num_mc_passes = st.slider("MC passes", 5, 30, 10, 5)
            
            # Cloud-specific options
            if CLOUD_AVAILABLE and use_cloud:
                st.info("☁️ Inference will run on Modal's T4 GPU")
        
        # Main action button
        if st.button("🚀 Run Inference", type="primary", use_container_width=True):
            run_inference_pipeline(
                image_path=selected_image,
                model_info=selected_model,
                subject_dir=selected_subject,
                bids_dir=bids_dir,
                overlap=overlap,
                sw_batch_size=sw_batch_size,
                save_prediction=save_prediction,
                enable_uncertainty=enable_uncertainty if UNCERTAINTY_AVAILABLE else False,
                num_mc_passes=num_mc_passes,
                min_nodule_size=min_nodule_size,
                confidence_threshold=confidence_threshold,
                use_cloud=use_cloud if CLOUD_AVAILABLE else False
            )
        
        # Batch processing in separate expander
        with st.expander("📦 Batch Processing"):
            selected_subjects_batch = st.multiselect(
                "Select multiple subjects",
                subjects,
                default=[],
                format_func=lambda x: x.name,
                key="batch_subjects"
            )
            
            if selected_subjects_batch:
                st.caption(f"{len(selected_subjects_batch)} subject(s) selected")
                
                if st.button("Run Batch Inference", use_container_width=True):
                    run_batch_inference(
                        subjects=selected_subjects_batch,
                        model_info=selected_model,
                        bids_dir=bids_dir,
                        overlap=overlap,
                        sw_batch_size=sw_batch_size,
                        save_prediction=save_prediction,
                        enable_uncertainty=enable_uncertainty if UNCERTAINTY_AVAILABLE else False,
                        num_mc_passes=num_mc_passes,
                        min_nodule_size=min_nodule_size,
                        confidence_threshold=confidence_threshold
                    )


def list_available_models() -> List[Dict[str, Any]]:
    """
    List available trained models by finding directories with config.json and best_model.pth.
    Supports nested versioned directories (e.g., model_name/v1_2025-12-03/)
    """
    models_dir = Path("./models")
    
    if not models_dir.exists():
        return []
    
    models = []
    
    def find_model_versions(base_dir: Path, model_name: str) -> List[Dict]:
        """Find all versions of a model"""
        found = []
        
        # Check if this directory itself is a valid model
        if (base_dir / "config.json").exists() and (base_dir / "best_model.pth").exists():
            config = load_config_safe(base_dir / "config.json")
            found.append({
                "name": model_name,
                "path": base_dir,
                "config_path": base_dir / "config.json",
                "checkpoint_path": base_dir / "best_model.pth",
                "type": config.get("model", {}).get("type", "Unknown"),
                "config": config
            })
        
        # Check subdirectories for versioned models
        for subdir in base_dir.iterdir():
            if subdir.is_dir() and (subdir / "config.json").exists() and (subdir / "best_model.pth").exists():
                config = load_config_safe(subdir / "config.json")
                version_name = f"{model_name}/{subdir.name}"
                found.append({
                    "name": version_name,
                    "path": subdir,
                    "config_path": subdir / "config.json",
                    "checkpoint_path": subdir / "best_model.pth",
                    "type": config.get("model", {}).get("type", "Unknown"),
                    "config": config
                })
        
        return found
    
    # Iterate through models directory
    for item in models_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            found_models = find_model_versions(item, item.name)
            models.extend(found_models)
    
    # Also look for .pth files directly in models dir (legacy support)
    for pth_file in models_dir.glob("*.pth"):
        # Check if there's a corresponding config
        config_file = pth_file.with_suffix('.json')
        if not config_file.exists():
            config_file = models_dir / "config.json"
        
        if config_file.exists():
            config = load_config_safe(config_file)
            models.append({
                "name": pth_file.stem,
                "path": pth_file.parent,
                "config_path": config_file,
                "checkpoint_path": pth_file,
                "type": config.get("model", {}).get("type", "PyTorch"),
                "config": config
            })
    
    return models


def load_config_safe(config_path: Path) -> Dict:
    """Safely load a config file"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


def display_model_info(model_info: Dict):
    """Display model information in the UI"""
    config = model_info.get("config", {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Model Configuration:**")
        model_config = config.get("model", {})
        st.json({
            "Type": model_config.get("type", "Unknown"),
            "Input Channels": model_config.get("in_channels", 1),
            "Output Channels": model_config.get("out_channels", 2),
            "Image Size": model_config.get("img_size", [96, 96, 96]),
            "Feature Size": model_config.get("feature_size", 48)
        })
    
    with col2:
        st.markdown("**Preprocessing Configuration:**")
        preproc_config = config.get("preprocessing", {})
        st.json({
            "Spacing": preproc_config.get("spacing", [1.5, 1.5, 1.5]),
            "Orientation": preproc_config.get("orientation", "RAS"),
            "Intensity Range": preproc_config.get("intensity_range", [-1000, 400]),
            "Target Range": preproc_config.get("target_range", [0, 1])
        })
    
    # Show training history summary if available
    history_path = model_info["path"] / "training_history.json"
    if history_path.exists():
        try:
            with open(history_path, 'r') as f:
                history = json.load(f)
            
            st.markdown("**Training Summary:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                epochs = len(history.get("train_loss", []))
                st.metric("Epochs Trained", epochs)
            with col2:
                val_dice = history.get("val_dice", [])
                if val_dice:
                    st.metric("Best Val Dice", f"{max(val_dice):.4f}")
            with col3:
                train_loss = history.get("train_loss", [])
                if train_loss:
                    st.metric("Final Train Loss", f"{train_loss[-1]:.4f}")
        except Exception:
            pass


def get_inference_transforms(config: Dict) -> Compose:
    """
    Build inference transforms from model config.
    Matches the preprocessing used during training.
    """
    preproc = config.get("preprocessing", {})
    
    spacing = preproc.get("spacing", [1.5, 1.5, 1.5])
    orientation = preproc.get("orientation", "RAS")
    intensity_range = preproc.get("intensity_range", [-1000.0, 400.0])
    target_range = preproc.get("target_range", [0.0, 1.0])
    divisible_k = preproc.get("divisible_k", 32)
    
    transforms = Compose([
        LoadImaged(keys=["image"], image_only=False),
        EnsureChannelFirstd(keys=["image"]),
        Spacingd(keys=["image"], pixdim=spacing, mode="bilinear"),
        Orientationd(keys=["image"], axcodes=orientation, labels=None),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=intensity_range[0],
            a_max=intensity_range[1],
            b_min=target_range[0],
            b_max=target_range[1],
            clip=True
        ),
        CropForegroundd(keys=["image"], source_key="image", margin=8, allow_smaller=True),
        DivisiblePadd(keys=["image"], k=divisible_k),
        ToTensord(keys=["image"])
    ])
    
    return transforms


def run_inference_pipeline(
    image_path: Path,
    model_info: Dict,
    subject_dir: Path,
    bids_dir: Path,
    overlap: float = 0.5,
    sw_batch_size: int = 2,
    save_prediction: bool = True,
    enable_uncertainty: bool = False,
    num_mc_passes: int = 10,
    min_nodule_size: float = 3.0,
    confidence_threshold: float = 0.3,
    use_cloud: bool = False
):
    """Run model inference on selected image with full pipeline and proper spatial alignment"""
    
    # Initialize timing dictionary
    timing_info = {
        "total": 0.0,
        "model_loading": 0.0,
        "preprocessing": 0.0,
        "inference": 0.0,
        "post_processing": 0.0,
        "resampling": 0.0,
        "nodule_extraction": 0.0,
        "saving": 0.0
    }
    
    start_total = time.time()
    progress_bar = st.progress(0, text="Initializing...")
    inference_source = "local"  # Track where inference ran
    
    try:
        # Step 1: Load configuration
        progress_bar.progress(10, text="Loading model configuration...")
        config = model_info["config"]
        model_config = config.get("model", {})
        
        # Step 2: Load original image metadata for spatial alignment
        progress_bar.progress(15, text="Loading image metadata...")
        original_img = nib.load(str(image_path))
        original_affine = original_img.affine
        original_header = original_img.header
        original_shape = original_img.shape
        original_spacing = original_img.header.get_zooms()[:3]
        
        # ============ CLOUD OR LOCAL INFERENCE ============
        if use_cloud and CLOUD_AVAILABLE:
            # Use cloud inference service
            progress_bar.progress(30, text="☁️ Running inference on cloud GPU...")
            
            try:
                service = get_inference_service()
                result = service.run_inference(
                    image_path=str(image_path),
                    model_path=str(model_info["path"]),
                    mode=InferenceMode.CLOUD_PREFERRED,
                    inference_params={
                        "overlap": overlap,
                        "sw_batch_size": sw_batch_size
                    },
                    enable_uncertainty=enable_uncertainty,
                    num_mc_passes=num_mc_passes
                )
                
                if result.success:
                    # Extract results from cloud inference
                    pred_resampled = result.prediction
                    confidence_resampled = result.confidence
                    uncertainty_resampled = result.uncertainty
                    timing_info.update(result.timing)
                    
                    # Determine source
                    if result.source == InferenceSource.CLOUD:
                        inference_source = "cloud"
                        st.success("☁️ Inference completed on Modal GPU")
                    elif result.source == InferenceSource.CLOUD_FALLBACK:
                        inference_source = "cloud_fallback"
                        st.warning("⚠️ Cloud failed, used local fallback")
                    else:
                        inference_source = "local"
                        st.info("🖥️ Inference completed locally")
                    
                    progress_bar.progress(75, text="Post-processing results...")
                else:
                    raise Exception(result.error or "Cloud inference failed")
                    
            except Exception as cloud_error:
                st.warning(f"☁️ Cloud inference failed: {cloud_error}. Falling back to local...")
                use_cloud = False  # Fall through to local inference
                inference_source = "cloud_fallback"
        
        if not use_cloud or not CLOUD_AVAILABLE or inference_source == "cloud_fallback":
            # Local inference path
            if inference_source != "cloud_fallback":
                # Step 3: Build transforms
                progress_bar.progress(20, text="Building preprocessing pipeline...")
                transforms = get_inference_transforms(config)
                
                # Step 4: Load model
                progress_bar.progress(30, text="Loading model (this may take a moment)...")
                start_model = time.time()
                
                from src.modeling.inference import load_model_from_config, predict_volume, resample_to_original
                
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model, _ = load_model_from_config(str(model_info["path"]), device=device)
                
                timing_info["model_loading"] = time.time() - start_model
                st.success(f"✅ Model loaded on {device}")
                inference_source = "local"
                
                # Step 5: Preprocess image
                progress_bar.progress(50, text="Preprocessing image...")
                start_preprocess = time.time()
                
                data_dict = {"image": str(image_path)}
                transformed = transforms(data_dict)
                image_tensor = transformed["image"].unsqueeze(0)  # Add batch dimension
                preprocessed_shape = image_tensor.shape[2:]  # Get spatial dims only
                
                timing_info["preprocessing"] = time.time() - start_preprocess
                st.info(f"📐 Original shape: {original_shape} | Preprocessed: {tuple(preprocessed_shape)}")
                
                # Step 6: Run inference (with or without uncertainty)
                roi_size = tuple(model_config.get("img_size", [96, 96, 96]))
                uncertainty_map = None
                uncertainty_metrics = None
                
                start_inference = time.time()
                
                if enable_uncertainty and UNCERTAINTY_AVAILABLE:
                    progress_bar.progress(55, text=f"Running Monte Carlo Dropout ({num_mc_passes} passes)...")
                    
                    # Use Monte Carlo Dropout for uncertainty estimation
                    mean_probs, uncertainty_map_tensor = monte_carlo_dropout_inference(
                        model=model,
                        image=image_tensor,
                        num_passes=num_mc_passes,
                        roi_size=roi_size,
                        sw_batch_size=sw_batch_size,
                        overlap=overlap,
                        device=device
                    )
                    
                    # Get segmentation from mean probabilities
                    pred_tensor = mean_probs
                    uncertainty_map = uncertainty_map_tensor.cpu().numpy()[0, 0]  # Remove batch and channel dims
                    
                    # Compute uncertainty metrics
                    uncertainty_metrics = compute_uncertainty_metrics(uncertainty_map_tensor)
                    
                    progress_bar.progress(70, text="Processing uncertainty results...")
                else:
                    progress_bar.progress(60, text="Running sliding window inference...")
                    
                    with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                        pred_tensor = predict_volume(
                            model=model,
                            image=image_tensor,
                            roi_size=roi_size,
                            sw_batch_size=sw_batch_size,
                            overlap=overlap,
                            device=device
                        )
                
                timing_info["inference"] = time.time() - start_inference
                
                # Step 7: Post-process predictions
                progress_bar.progress(75, text="Post-processing predictions...")
                start_postprocess = time.time()
                
                pred_np = pred_tensor.cpu().numpy()[0]  # Remove batch dimension
                
                # Apply softmax and argmax for multi-class
                if pred_np.shape[0] > 1:
                    pred_softmax = np.exp(pred_np) / np.sum(np.exp(pred_np), axis=0, keepdims=True)
                    pred_seg = np.argmax(pred_softmax, axis=0).astype(np.uint8)
                    confidence = np.max(pred_softmax, axis=0)
                else:
                    pred_seg = (pred_np[0] > 0.5).astype(np.uint8)
                    confidence = pred_np[0]
                
                timing_info["post_processing"] = time.time() - start_postprocess
                
                # Step 8: Resample prediction back to original space for proper spatial alignment
                progress_bar.progress(82, text="Resampling to original space...")
                start_resample = time.time()
                
                pred_resampled = resample_to_original(
                    prediction=pred_seg,
                    original_shape=original_shape,
                    preprocessed_shape=tuple(preprocessed_shape),
                    mode='nearest'
                )
                
                # Also resample confidence map
                confidence_resampled = resample_to_original(
                    prediction=confidence.astype(np.float32),
                    original_shape=original_shape,
                    preprocessed_shape=tuple(preprocessed_shape),
                    mode='linear'
                ) if confidence is not None else None
                
                # Resample uncertainty map if available
                uncertainty_resampled = None
                if uncertainty_map is not None:
                    uncertainty_resampled = resample_to_original(
                        prediction=uncertainty_map.astype(np.float32),
                        original_shape=original_shape,
                        preprocessed_shape=tuple(preprocessed_shape),
                        mode='linear'
                    )
                
                timing_info["resampling"] = time.time() - start_resample
        
        # Step 9: Post-processing - extract nodule candidates
        nodule_candidates = []
        nodule_stats = {}
        
        if POSTPROCESSING_AVAILABLE and np.sum(pred_resampled > 0) > 0:
            progress_bar.progress(88, text="Extracting nodule candidates...")
            start_nodule = time.time()
            
            nodule_candidates, nodule_stats = postprocess_segmentation(
                segmentation=pred_resampled,
                spacing=original_spacing,
                confidence_map=confidence_resampled,
                min_diameter_mm=min_nodule_size,
                max_diameter_mm=100.0,
                min_confidence=confidence_threshold,
                min_voxels=10
            )
            
            timing_info["nodule_extraction"] = time.time() - start_nodule
        
        # Step 10: Save prediction if requested (with proper spatial alignment)
        saved_path = None
        if save_prediction:
            progress_bar.progress(92, text="Saving prediction with spatial alignment...")
            start_saving = time.time()
            
            saved_path = save_prediction_nifti_aligned(
                prediction=pred_resampled,
                confidence=confidence_resampled,
                image_path=image_path,
                subject_dir=subject_dir,
                bids_dir=bids_dir,
                model_name=model_info["name"].replace("/", "_"),
                original_affine=original_affine,
                original_header=original_header
            )
            
            timing_info["saving"] = time.time() - start_saving
        
        timing_info["total"] = time.time() - start_total
        progress_bar.progress(100, text="Complete!")
        
        # Store results in session state for visualization
        st.session_state['last_inference_result'] = {
            'pred_seg': pred_resampled,
            'confidence': confidence_resampled,
            'uncertainty': uncertainty_resampled,
            'uncertainty_metrics': uncertainty_metrics if 'uncertainty_metrics' in dir() else None,
            'original_shape': original_shape,
            'original_spacing': original_spacing,
            'image_path': str(image_path),
            'model_name': model_info["name"],
            'timestamp': datetime.now().isoformat(),
            'nodule_candidates': nodule_candidates,
            'nodule_stats': nodule_stats,
            'enable_uncertainty': enable_uncertainty,
            'timing_info': timing_info,
            'inference_source': inference_source
        }
        
        # Display results
        st.markdown("---")
        
        # Show inference source indicator
        if inference_source == "cloud":
            st.markdown("### ☁️ Cloud Inference Results")
            st.caption("Inference ran on Modal's serverless T4 GPU")
        elif inference_source == "cloud_fallback":
            st.markdown("### 🔄 Fallback Inference Results")
            st.caption("Cloud was unavailable, inference ran locally")
        else:
            st.markdown("### 🖥️ Local Inference Results")
            device_name = "GPU" if torch.cuda.is_available() else "CPU"
            st.caption(f"Inference ran on local {device_name}")
        
        st.markdown("### ⏱️ Inference Timing")
        display_timing_info(timing_info, inference_source=inference_source)
        
        st.markdown("---")
        st.markdown("### 📊 Results")
        
        # Metrics with spacing-aware calculations
        display_prediction_metrics(pred_resampled, original_spacing)
        
        # Uncertainty results (if enabled)
        if enable_uncertainty and uncertainty_metrics is not None:
            st.markdown("### 🎲 Uncertainty Analysis")
            display_uncertainty_results(uncertainty_metrics, uncertainty_resampled, pred_resampled)
        
        # Nodule detection results
        if nodule_candidates:
            st.markdown("### 🔬 Nodule Detection")
            display_nodule_results(nodule_candidates, nodule_stats)
        
        # Visualization - use resampled prediction with original image
        st.markdown("### 🖼️ Visualization")
        display_prediction_overlay_aligned(image_path, pred_resampled, confidence_resampled, uncertainty_resampled)
        
        # Show saved path
        if saved_path:
            st.success(f"💾 Prediction saved to: `{saved_path}`")
            st.info("✅ Prediction saved with proper spatial alignment (affine preserved)")
        
        # Save to history with full result data
        if HISTORY_AVAILABLE:
            try:
                history_manager = get_history_manager(str(bids_dir))
                
                # Serialize nodule candidates for storage
                serialized_nodules = []
                for candidate in nodule_candidates:
                    serialized_nodules.append(candidate.to_dict())
                
                # Get uncertainty metrics if available
                stored_uncertainty_metrics = None
                if enable_uncertainty and 'uncertainty_metrics' in dir() and uncertainty_metrics is not None:
                    stored_uncertainty_metrics = uncertainty_metrics
                
                history_manager.add_entry(
                    image_path=str(image_path),
                    model_name=model_info["name"],
                    prediction_path=saved_path,
                    nodule_count=nodule_stats.get("total_nodules", 0),
                    suspicious_count=nodule_stats.get("suspicious_count", 0),
                    stats=nodule_stats,
                    parameters={
                        "overlap": overlap,
                        "sw_batch_size": sw_batch_size,
                        "enable_uncertainty": enable_uncertainty,
                        "min_nodule_size": min_nodule_size,
                        "confidence_threshold": confidence_threshold
                    },
                    metadata={
                        "original_shape": list(original_shape),
                        "original_spacing": list(original_spacing)
                    },
                    # New fields for full result preservation
                    timing_info=timing_info,
                    inference_source=inference_source,
                    nodule_candidates=serialized_nodules,
                    uncertainty_metrics=stored_uncertainty_metrics
                )
            except Exception as hist_e:
                st.warning(f"Could not save to history: {hist_e}")
        
        # Clean up GPU memory
        if inference_source == "local" or inference_source == "cloud_fallback":
            try:
                del model, pred_tensor, image_tensor
            except NameError:
                pass  # Variables may not exist if cloud inference was used
        torch.cuda.empty_cache()
        
    except Exception as e:
        progress_bar.empty()
        st.error(f"❌ Inference failed: {e}")
        st.exception(e)


def run_batch_inference(
    subjects: List[Path],
    model_info: Dict,
    bids_dir: Path,
    overlap: float = 0.5,
    sw_batch_size: int = 2,
    save_prediction: bool = True,
    enable_uncertainty: bool = False,
    num_mc_passes: int = 10,
    min_nodule_size: float = 3.0,
    confidence_threshold: float = 0.3
):
    """Run batch inference on multiple subjects with progress tracking"""
    
    start_batch_total = time.time()
    
    st.markdown("---")
    st.markdown("### 📊 Batch Processing Progress")
    
    # Initialize results storage
    batch_results = []
    total_subjects = len(subjects)
    
    # Progress tracking
    overall_progress = st.progress(0, text=f"Processing 0/{total_subjects} subjects...")
    status_container = st.empty()
    
    # Load model once for all subjects
    start_model_load = time.time()
    model_load_time = 0.0
    try:
        status_container.info("🔄 Loading model...")
        
        from src.modeling.inference import load_model_from_config, predict_volume, resample_to_original
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, _ = load_model_from_config(str(model_info["path"]), device=device)
        config = model_info["config"]
        model_config = config.get("model", {})
        transforms = get_inference_transforms(config)
        roi_size = tuple(model_config.get("img_size", [96, 96, 96]))
        
        model_load_time = time.time() - start_model_load
        status_container.success(f"✅ Model loaded on {device} ({model_load_time:.2f}s)")
        
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return
    
    # Process each subject
    subject_times = []
    for idx, subject_dir in enumerate(subjects):
        start_subject = time.time()
        subject_name = subject_dir.name
        status_container.info(f"🔄 Processing {subject_name} ({idx + 1}/{total_subjects})...")
        
        try:
            # Find images for this subject
            anat_dir = subject_dir / "anat"
            if not anat_dir.exists():
                batch_results.append({
                    "subject": subject_name,
                    "status": "❌ Error",
                    "message": "No anat directory found",
                    "nodules": 0
                })
                continue
            
            images = list(anat_dir.glob("*.nii.gz"))
            if not images:
                batch_results.append({
                    "subject": subject_name,
                    "status": "❌ Error",
                    "message": "No NIfTI images found",
                    "nodules": 0
                })
                continue
            
            # Process first image (or could process all)
            image_path = images[0]
            
            # Load image metadata
            original_img = nib.load(str(image_path))
            original_affine = original_img.affine
            original_header = original_img.header
            original_shape = original_img.shape
            original_spacing = original_img.header.get_zooms()[:3]
            
            # Preprocess
            data_dict = {"image": str(image_path)}
            transformed = transforms(data_dict)
            image_tensor = transformed["image"].unsqueeze(0)
            preprocessed_shape = image_tensor.shape[2:]
            
            # Run inference
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                pred_tensor = predict_volume(
                    model=model,
                    image=image_tensor,
                    roi_size=roi_size,
                    sw_batch_size=sw_batch_size,
                    overlap=overlap,
                    device=device
                )
            
            # Post-process
            pred_np = pred_tensor.cpu().numpy()[0]
            
            if pred_np.shape[0] > 1:
                pred_softmax = np.exp(pred_np) / np.sum(np.exp(pred_np), axis=0, keepdims=True)
                pred_seg = np.argmax(pred_softmax, axis=0).astype(np.uint8)
                confidence = np.max(pred_softmax, axis=0)
            else:
                pred_seg = (pred_np[0] > 0.5).astype(np.uint8)
                confidence = pred_np[0]
            
            # Resample to original space
            pred_resampled = resample_to_original(
                prediction=pred_seg,
                original_shape=original_shape,
                preprocessed_shape=tuple(preprocessed_shape),
                mode='nearest'
            )
            
            confidence_resampled = resample_to_original(
                prediction=confidence.astype(np.float32),
                original_shape=original_shape,
                preprocessed_shape=tuple(preprocessed_shape),
                mode='linear'
            )
            
            # Post-processing for nodules
            nodule_candidates = []
            nodule_stats = {}
            
            if POSTPROCESSING_AVAILABLE and np.sum(pred_resampled > 0) > 0:
                nodule_candidates, nodule_stats = postprocess_segmentation(
                    segmentation=pred_resampled,
                    spacing=original_spacing,
                    confidence_map=confidence_resampled,
                    min_diameter_mm=min_nodule_size,
                    max_diameter_mm=100.0,
                    min_confidence=confidence_threshold,
                    min_voxels=10
                )
            
            # Save prediction
            saved_path = None
            if save_prediction:
                saved_path = save_prediction_nifti_aligned(
                    prediction=pred_resampled,
                    confidence=confidence_resampled,
                    image_path=image_path,
                    subject_dir=subject_dir,
                    bids_dir=bids_dir,
                    model_name=model_info["name"].replace("/", "_"),
                    original_affine=original_affine,
                    original_header=original_header
                )
            
            # Record result
            subject_time = time.time() - start_subject
            subject_times.append(subject_time)
            
            suspicious_count = nodule_stats.get("suspicious_count", 0)
            batch_results.append({
                "subject": subject_name,
                "status": "✅ Success",
                "nodules": nodule_stats.get("total_nodules", 0),
                "suspicious": suspicious_count,
                "max_diameter": f"{nodule_stats.get('max_diameter_mm', 0):.1f}" if nodule_stats.get('max_diameter_mm', 0) > 0 else "N/A",
                "saved_path": saved_path,
                "alert": "⚠️" if suspicious_count > 0 else "",
                "processing_time": subject_time
            })
            
        except Exception as e:
            subject_time = time.time() - start_subject
            batch_results.append({
                "subject": subject_name,
                "status": "❌ Error",
                "message": str(e),
                "nodules": 0,
                "processing_time": subject_time
            })
        
        # Update progress
        progress_pct = (idx + 1) / total_subjects
        overall_progress.progress(progress_pct, text=f"Processing {idx + 1}/{total_subjects} subjects...")
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    
    # Calculate total time
    total_batch_time = time.time() - start_batch_total
    
    # Display results summary
    overall_progress.progress(1.0, text="✅ Batch processing complete!")
    status_container.empty()
    
    # Display timing summary
    st.markdown("### ⏱️ Batch Processing Timing")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Time", f"{total_batch_time:.2f} s")
    with col2:
        st.metric("Model Loading", f"{model_load_time:.2f} s")
    with col3:
        avg_time = np.mean(subject_times) if subject_times else 0
        st.metric("Avg per Subject", f"{avg_time:.2f} s")
    with col4:
        throughput = total_subjects / total_batch_time if total_batch_time > 0 else 0
        st.metric("Throughput", f"{throughput:.2f} subjects/s")
    
    st.markdown("### 📋 Batch Results Summary")
    
    # Summary metrics
    successful = sum(1 for r in batch_results if r["status"] == "✅ Success")
    total_nodules = sum(r.get("nodules", 0) for r in batch_results)
    total_suspicious = sum(r.get("suspicious", 0) for r in batch_results)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("✅ Processed", f"{successful}/{total_subjects}")
    with col2:
        st.metric("🎯 Total Nodules", total_nodules)
    with col3:
        st.metric("⚠️ Suspicious", total_suspicious)
    with col4:
        st.metric("❌ Errors", total_subjects - successful)
    
    # Results table
    st.markdown("**Detailed Results:**")
    results_data = []
    for r in batch_results:
        row = {
            "Subject": r["subject"],
            "Status": r["status"],
            "Nodules": r.get("nodules", 0),
            "Suspicious": r.get("suspicious", 0),
            "Max Diameter (mm)": r.get("max_diameter", "N/A"),
            "Time (s)": f"{r.get('processing_time', 0):.2f}",
            "Alert": r.get("alert", "")
        }
        if "message" in r:
            row["Notes"] = r["message"]
        results_data.append(row)
    
    st.dataframe(results_data, use_container_width=True)
    
    # Store batch results in session state
    st.session_state['batch_results'] = batch_results
    
    # Alert for suspicious findings
    if total_suspicious > 0:
        st.warning(f"⚠️ {total_suspicious} suspicious nodule(s) detected across {sum(1 for r in batch_results if r.get('suspicious', 0) > 0)} subject(s). Please review these cases.")


def save_prediction_nifti_aligned(
    prediction: np.ndarray,
    confidence: Optional[np.ndarray],
    image_path: Path,
    subject_dir: Path,
    bids_dir: Path,
    model_name: str,
    original_affine: np.ndarray,
    original_header: nib.Nifti1Header
) -> str:
    """
    Save prediction as NIfTI in BIDS derivatives format with proper spatial alignment.
    
    Args:
        prediction: Segmentation prediction resampled to original space
        confidence: Confidence map (optional)
        image_path: Original image path
        subject_dir: Subject directory path
        bids_dir: BIDS dataset root path
        model_name: Name of the model used
        original_affine: Original image affine matrix
        original_header: Original image NIfTI header
        
    Returns:
        Path to saved prediction file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create derivatives directory within the selected dataset
    derivatives_dir = bids_dir / "derivatives" / "predictions" / model_name / subject_dir.name / "anat"
    derivatives_dir.mkdir(parents=True, exist_ok=True)
    
    # Create header with proper data type for segmentation
    new_header = original_header.copy()
    new_header.set_data_dtype(np.uint8)
    
    # Create NIfTI image with original spatial information (proper alignment!)
    pred_img = nib.Nifti1Image(
        prediction.astype(np.uint8),
        affine=original_affine,
        header=new_header
    )
    
    # Generate filename with timestamp for versioning
    image_name = image_path.stem.replace(".nii", "")
    output_filename = f"{subject_dir.name}_{image_name}_pred-{model_name}_{timestamp}.nii.gz"
    output_path = derivatives_dir / output_filename
    
    nib.save(pred_img, str(output_path))
    
    # Also save confidence map if available
    if confidence is not None:
        conf_header = original_header.copy()
        conf_header.set_data_dtype(np.float32)
        conf_img = nib.Nifti1Image(
            confidence.astype(np.float32),
            affine=original_affine,
            header=conf_header
        )
        conf_filename = f"{subject_dir.name}_{image_name}_confidence-{model_name}_{timestamp}.nii.gz"
        conf_path = derivatives_dir / conf_filename
        nib.save(conf_img, str(conf_path))
    
    # Save inference metadata as JSON sidecar
    metadata = {
        "model_name": model_name,
        "timestamp": timestamp,
        "original_image": str(image_path),
        "prediction_file": str(output_path),
        "original_shape": list(prediction.shape),
        "unique_labels": [int(x) for x in np.unique(prediction)],
        "foreground_voxels": int(np.sum(prediction > 0)),
        "total_voxels": int(prediction.size)
    }
    
    metadata_path = output_path.with_suffix('.json')
    with open(str(metadata_path).replace('.nii.gz.json', '.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return str(output_path)


def display_timing_info(timing_info: Dict[str, float], inference_source: Optional[str] = None):
    """Display inference timing information in a clear format
    
    Args:
        timing_info: Dict with timing breakdown
        inference_source: Optional source indicator for historical data ("local", "cloud", "cloud_fallback")
    """
    
    def format_time(seconds: float) -> str:
        """Format time in seconds to human-readable format"""
        if seconds < 1:
            return f"{seconds * 1000:.1f} ms"
        elif seconds < 60:
            return f"{seconds:.2f} s"
        else:
            mins = int(seconds // 60)
            secs = seconds % 60
            return f"{mins}m {secs:.1f}s"
    
    def format_percentage(part: float, total: float) -> str:
        """Format as percentage"""
        if total == 0:
            return "0%"
        return f"{(part / total * 100):.1f}%"
    
    total_time = timing_info.get("total", 0)
    
    # Main timing metrics
    st.markdown("**⏱️ Total Inference Time:**")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.metric("Total Time", format_time(total_time))
    
    with col2:
        # Use provided inference_source for history, or detect current device
        if inference_source:
            if inference_source == "cloud":
                device_type = "☁️ Cloud GPU"
            elif inference_source == "cloud_fallback":
                device_type = "🔄 Fallback"
            else:
                device_type = "🖥️ Local"
        else:
            device_type = "GPU" if torch.cuda.is_available() else "CPU"
        st.metric("Device", device_type)
    
    # Breakdown by stage
    st.markdown("**📋 Time Breakdown by Stage:**")
    
    timing_data = []
    
    # Model loading
    model_time = timing_info.get("model_loading", 0)
    if model_time > 0:
        timing_data.append({
            "Stage": "🔄 Model Loading",
            "Time": format_time(model_time),
            "Percentage": format_percentage(model_time, total_time),
            "Description": "Loading model weights from disk into memory"
        })
    
    # Preprocessing
    preprocess_time = timing_info.get("preprocessing", 0)
    if preprocess_time > 0:
        timing_data.append({
            "Stage": "🔧 Image Preprocessing",
            "Time": format_time(preprocess_time),
            "Percentage": format_percentage(preprocess_time, total_time),
            "Description": "Resampling, normalization, and transform operations"
        })
    
    # Inference (most important!)
    inference_time = timing_info.get("inference", 0)
    if inference_time > 0:
        timing_data.append({
            "Stage": "🧠 Model Inference",
            "Time": format_time(inference_time),
            "Percentage": format_percentage(inference_time, total_time),
            "Description": "**Core AI prediction** - Your trained model processing the image"
        })
    
    # Post-processing
    postprocess_time = timing_info.get("post_processing", 0)
    if postprocess_time > 0:
        timing_data.append({
            "Stage": "⚙️ Post-Processing",
            "Time": format_time(postprocess_time),
            "Percentage": format_percentage(postprocess_time, total_time),
            "Description": "Converting probabilities to segmentation mask"
        })
    
    # Resampling
    resample_time = timing_info.get("resampling", 0)
    if resample_time > 0:
        timing_data.append({
            "Stage": "📐 Spatial Resampling",
            "Time": format_time(resample_time),
            "Percentage": format_percentage(resample_time, total_time),
            "Description": "Mapping predictions back to original image space"
        })
    
    # Nodule extraction
    nodule_time = timing_info.get("nodule_extraction", 0)
    if nodule_time > 0:
        timing_data.append({
            "Stage": "🔬 Nodule Extraction",
            "Time": format_time(nodule_time),
            "Percentage": format_percentage(nodule_time, total_time),
            "Description": "Connected component analysis and feature extraction"
        })
    
    # Saving
    save_time = timing_info.get("saving", 0)
    if save_time > 0:
        timing_data.append({
            "Stage": "💾 Saving Results",
            "Time": format_time(save_time),
            "Percentage": format_percentage(save_time, total_time),
            "Description": "Writing prediction files to disk"
        })
    
    # Display as table
    if timing_data:
        st.dataframe(timing_data, use_container_width=True, hide_index=True)
    
    # Performance insights
    if total_time > 0:
        inference_pct = (inference_time / total_time * 100) if inference_time > 0 else 0
        
        st.markdown("**💡 Performance Insights:**")
        
        if inference_pct > 70:
            st.info(f"✅ **Model inference dominates** ({inference_pct:.1f}% of total time) - This is expected and indicates efficient preprocessing.")
        elif inference_pct < 30:
            st.warning(f"⚠️ **Preprocessing/other steps take longer** than inference ({inference_pct:.1f}% inference). Consider optimizing preprocessing pipeline.")
        else:
            st.info(f"ℹ️ **Balanced pipeline** - Inference takes {inference_pct:.1f}% of total time.")
        
        # Throughput calculation if we have image size info
        if 'original_shape' in st.session_state.get('last_inference_result', {}):
            original_shape = st.session_state['last_inference_result']['original_shape']
            total_voxels = np.prod(original_shape)
            voxels_per_second = total_voxels / inference_time if inference_time > 0 else 0
            st.caption(f"📊 Processing speed: {voxels_per_second:,.0f} voxels/second during inference")


def display_prediction_metrics(prediction: np.ndarray, spacing: Optional[Tuple[float, ...]] = None):
    """Display prediction statistics with physical measurements when spacing available"""
    
    unique_labels = np.unique(prediction)
    total_voxels = prediction.size
    foreground_voxels = int(np.sum(prediction > 0))
    
    # Calculate physical volume if spacing is available
    if spacing is not None:
        voxel_volume_mm3 = float(np.prod(spacing))
        foreground_volume_mm3 = foreground_voxels * voxel_volume_mm3
        foreground_volume_ml = foreground_volume_mm3 / 1000.0  # Convert to mL
    else:
        voxel_volume_mm3 = None
        foreground_volume_mm3 = None
        foreground_volume_ml = None
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🎯 Unique Labels", len(unique_labels))
    
    with col2:
        st.metric("📦 Total Voxels", f"{total_voxels:,}")
    
    with col3:
        st.metric("🔍 Foreground Voxels", f"{foreground_voxels:,}")
    
    with col4:
        coverage = (foreground_voxels / total_voxels) * 100
        st.metric("📊 Coverage", f"{coverage:.4f}%")
    
    # Physical measurements row
    if spacing is not None:
        st.markdown("**Physical Measurements:**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📏 Voxel Size", f"{spacing[0]:.2f}×{spacing[1]:.2f}×{spacing[2]:.2f} mm")
        
        with col2:
            st.metric("📐 Voxel Volume", f"{voxel_volume_mm3:.3f} mm³")
        
        with col3:
            st.metric("🫁 Foreground Volume", f"{foreground_volume_mm3:.1f} mm³")
        
        with col4:
            st.metric("💧 Volume (mL)", f"{foreground_volume_ml:.3f} mL")
    
    # Per-class breakdown
    if len(unique_labels) > 1:
        st.markdown("**Per-Class Breakdown:**")
        class_data = []
        for label in unique_labels:
            count = int(np.sum(prediction == label))
            pct = (count / total_voxels) * 100
            row = {
                "Label": int(label),
                "Voxels": f"{count:,}",
                "Percentage": f"{pct:.4f}%"
            }
            if spacing is not None:
                vol_mm3 = count * voxel_volume_mm3
                row["Volume (mm³)"] = f"{vol_mm3:.1f}"
            class_data.append(row)
        st.dataframe(class_data, use_container_width=True)


def display_uncertainty_results(
    uncertainty_metrics: Dict,
    uncertainty_map: Optional[np.ndarray] = None,
    prediction: Optional[np.ndarray] = None
):
    """Display uncertainty quantification results
    
    Args:
        uncertainty_metrics: Dict with uncertainty statistics
        uncertainty_map: Optional uncertainty map array (for live results)
        prediction: Optional prediction array (for live results)
    """
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        mean_unc = uncertainty_metrics.get("mean_uncertainty", 0)
        st.metric("📊 Mean Uncertainty", f"{mean_unc:.4f}")
    
    with col2:
        std_unc = uncertainty_metrics.get("std_uncertainty", 0)
        st.metric("📈 Std Uncertainty", f"{std_unc:.4f}")
    
    with col3:
        max_unc = uncertainty_metrics.get("max_uncertainty", 0)
        st.metric("⚠️ Max Uncertainty", f"{max_unc:.4f}")
    
    with col4:
        high_unc_voxels = uncertainty_metrics.get("high_uncertainty_voxels", 0)
        total_voxels = uncertainty_metrics.get("total_voxels", 1)
        high_unc_pct = (high_unc_voxels / total_voxels) * 100
        st.metric("🔴 High Uncertainty Regions", f"{high_unc_pct:.2f}%")
    
    # Uncertainty interpretation
    mean_unc = uncertainty_metrics.get("mean_uncertainty", 0)
    if mean_unc < 0.05:
        st.success("✅ Low overall uncertainty - predictions are confident")
    elif mean_unc < 0.15:
        st.warning("⚠️ Moderate uncertainty - some regions may need review")
    else:
        st.error("🔴 High uncertainty - predictions should be reviewed carefully")
    
    # Uncertainty in foreground regions (only available for live results with arrays)
    if uncertainty_map is not None and prediction is not None:
        foreground_mask = prediction > 0
        if np.sum(foreground_mask) > 0:
            foreground_uncertainty = uncertainty_map[foreground_mask]
            mean_fg_unc = float(np.mean(foreground_uncertainty))
            max_fg_unc = float(np.max(foreground_uncertainty))
            
            st.markdown("**Uncertainty in Detected Regions:**")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Mean (foreground)", f"{mean_fg_unc:.4f}")
            with col2:
                st.metric("Max (foreground)", f"{max_fg_unc:.4f}")


def _get_nodule_attr(nodule: Any, attr: str, default: Any = None) -> Any:
    """Helper to get attribute from NoduleCandidate object or dict"""
    if hasattr(nodule, attr):
        return getattr(nodule, attr)
    elif isinstance(nodule, dict):
        return nodule.get(attr, default)
    return default


def display_nodule_results(candidates: List, stats: Dict, show_export: bool = True):
    """Display nodule detection results with detailed information and export options
    
    Args:
        candidates: List of NoduleCandidate objects or dicts
        stats: Aggregate statistics dict
        show_export: Whether to show export options (disable for history view)
    """
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🎯 Nodules Detected", stats.get("total_nodules", 0))
    
    with col2:
        suspicious = stats.get("suspicious_count", 0)
        st.metric("⚠️ Suspicious (4A+)", suspicious, 
                  delta=None if suspicious == 0 else "Needs review",
                  delta_color="inverse" if suspicious > 0 else "off")
    
    with col3:
        total_vol = stats.get("total_volume_mm3", 0)
        st.metric("📐 Total Volume", f"{total_vol:.1f} mm³")
    
    with col4:
        max_diam = stats.get("max_diameter_mm", 0)
        st.metric("📏 Max Diameter", f"{max_diam:.1f} mm")
    
    # Lung-RADS category breakdown
    category_counts = stats.get("category_counts", {})
    if category_counts:
        st.markdown("**Lung-RADS Category Distribution:**")
        cat_cols = st.columns(len(category_counts))
        for i, (cat, count) in enumerate(sorted(category_counts.items())):
            with cat_cols[i]:
                color = "🟢" if cat in ["1", "2"] else "🟡" if cat == "3" else "🔴"
                st.metric(f"{color} Category {cat}", count)
    
    # Detailed nodule table
    if candidates:
        st.markdown("**Individual Nodule Details:**")
        
        # Create DataFrame-like structure for display
        # Works with both NoduleCandidate objects and dicts
        nodule_data = []
        for c in candidates:
            # Get centroid - handle both object and dict formats
            centroid = _get_nodule_attr(c, 'centroid_mm', (0, 0, 0))
            if isinstance(centroid, (list, tuple)) and len(centroid) >= 3:
                location_str = f"({centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f})"
            else:
                location_str = "N/A"
            
            row = {
                "ID": _get_nodule_attr(c, 'id', 0),
                "Location (mm)": location_str,
                "Diameter (mm)": f"{_get_nodule_attr(c, 'equivalent_diameter_mm', 0):.1f}",
                "Volume (mm³)": f"{_get_nodule_attr(c, 'volume_mm3', 0):.1f}",
                "Confidence": f"{_get_nodule_attr(c, 'mean_confidence', 0):.2f}",
                "Sphericity": f"{_get_nodule_attr(c, 'sphericity', 0):.2f}",
                "Lung-RADS": _get_nodule_attr(c, 'lung_rads_category', 'N/A')
            }
            nodule_data.append(row)
        
        st.dataframe(nodule_data, use_container_width=True)
        
        # Export options (only for live results, not history view)
        if show_export and REPORT_AVAILABLE:
            display_export_options(candidates, stats)
        
        # Lung-RADS interpretation guide
        with st.expander("📖 Lung-RADS Interpretation Guide"):
            st.markdown("""
            | Category | Risk | Management |
            |----------|------|------------|
            | **1** | Negative | Continue annual screening |
            | **2** | Benign appearance | Continue annual screening |
            | **3** | Probably benign | 6-month follow-up CT |
            | **4A** | Suspicious | 3-month follow-up CT or PET/CT |
            | **4B** | Very suspicious | Tissue sampling or surgical consultation |
            | **4X** | Additional features | Based on specific findings |
            
            *Note: This is AI-assisted detection. All findings should be reviewed by a qualified radiologist.*
            """)


def display_export_options(candidates: List, stats: Dict):
    """Display export options for nodule detection report"""
    
    st.markdown("**📥 Export Report**")
    
    # Get inference result from session state for metadata
    result = st.session_state.get('last_inference_result', {})
    
    image_info = {
        "image_path": result.get('image_path', ''),
        "original_shape": result.get('original_shape', []),
        "original_spacing": result.get('original_spacing', []),
        "subject_id": Path(result.get('image_path', '')).parent.parent.name if result.get('image_path') else ''
    }
    
    model_info = {
        "name": result.get('model_name', ''),
        "type": "SwinUNETR",  # From config
        "timestamp": result.get('timestamp', datetime.now().isoformat())
    }
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # JSON Report
        if st.button("📋 JSON Report", key="export_json", use_container_width=True):
            try:
                json_report = generate_nodule_report(
                    candidates, stats, image_info, model_info, "json"
                )
                st.download_button(
                    label="⬇️ Download JSON",
                    data=json_report,
                    file_name=f"nodule_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    key="download_json"
                )
            except Exception as e:
                st.error(f"Error generating JSON report: {e}")
    
    with col2:
        # CSV Report
        if st.button("📊 CSV Export", key="export_csv", use_container_width=True):
            try:
                csv_report = generate_nodule_report(
                    candidates, stats, image_info, model_info, "csv"
                )
                st.download_button(
                    label="⬇️ Download CSV",
                    data=csv_report,
                    file_name=f"nodule_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="download_csv"
                )
            except Exception as e:
                st.error(f"Error generating CSV report: {e}")
    
    with col3:
        # Summary Report
        if st.button("📄 Summary Report", key="export_summary", use_container_width=True):
            try:
                summary_report = generate_nodule_report(
                    candidates, stats, image_info, model_info, "summary"
                )
                st.download_button(
                    label="⬇️ Download Summary",
                    data=summary_report,
                    file_name=f"nodule_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    key="download_summary"
                )
            except Exception as e:
                st.error(f"Error generating summary report: {e}")
    
    # Preview summary report
    with st.expander("👁️ Preview Summary Report"):
        try:
            summary = generate_nodule_report(
                candidates, stats, image_info, model_info, "summary"
            )
            st.text(summary)
        except Exception as e:
            st.error(f"Error generating preview: {e}")


# CT Window presets (Window Width, Window Level)
CT_WINDOW_PRESETS = {
    "Lung": (1500, -600),
    "Mediastinum": (400, 40),
    "Bone": (2000, 300),
    "Soft Tissue": (400, 50),
    "Brain": (80, 40),
    "Liver": (150, 30),
    "Custom": None
}


def apply_window_level(image: np.ndarray, window: float, level: float) -> np.ndarray:
    """
    Apply window/level (contrast/brightness) to CT image.
    
    Args:
        image: Input image array (Hounsfield units for CT)
        window: Window width (contrast)
        level: Window level/center (brightness)
        
    Returns:
        Windowed image normalized to [0, 1]
    """
    min_val = level - window / 2
    max_val = level + window / 2
    
    windowed = np.clip(image, min_val, max_val)
    windowed = (windowed - min_val) / (max_val - min_val)
    
    return windowed


def display_prediction_overlay_aligned(
    image_path: Path,
    prediction: np.ndarray,
    confidence: Optional[np.ndarray] = None,
    uncertainty: Optional[np.ndarray] = None
):
    """
    Display image with prediction overlay - using original image space (aligned).
    Enhanced with windowing controls, opacity slider, and multiple overlay options.
    """
    
    # Load original image directly (no preprocessing needed - prediction is aligned!)
    img = nib.load(str(image_path))
    image_data = img.get_fdata()
    
    # Verify shapes match (they should now!)
    if image_data.shape != prediction.shape:
        st.warning(f"⚠️ Shape mismatch: Image {image_data.shape} vs Prediction {prediction.shape}")
        min_shape = tuple(min(a, b) for a, b in zip(image_data.shape[:3], prediction.shape[:3]))
        image_data = image_data[:min_shape[0], :min_shape[1], :min_shape[2]]
        prediction = prediction[:min_shape[0], :min_shape[1], :min_shape[2]]
        if confidence is not None:
            confidence = confidence[:min_shape[0], :min_shape[1], :min_shape[2]]
        if uncertainty is not None:
            uncertainty = uncertainty[:min_shape[0], :min_shape[1], :min_shape[2]]
    else:
        st.success("✅ Prediction aligned with original image space")
    
    # ============ VISUALIZATION CONTROLS ============
    st.markdown("#### Display Controls")
    
    # Row 1: View and Overlay type
    col1, col2, col3 = st.columns(3)
    
    with col1:
        view_type = st.radio("View", ["Axial", "Coronal", "Sagittal"], horizontal=True, key="view_aligned")
    
    with col2:
        overlay_options = ["Segmentation", "Confidence"]
        if uncertainty is not None:
            overlay_options.append("Uncertainty")
        overlay_type = st.radio("Overlay", overlay_options, horizontal=True, key="overlay_type")
    
    with col3:
        overlay_opacity = st.slider("Overlay Opacity", 0.0, 1.0, 0.5, 0.1, key="overlay_opacity")
    
    # Row 2: Window/Level Controls
    st.markdown("#### CT Window/Level")
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        window_preset = st.selectbox(
            "Preset", 
            list(CT_WINDOW_PRESETS.keys()),
            index=0,  # Default to Lung for nodule detection
            key="window_preset"
        )
    
    # Get default values from preset
    if window_preset != "Custom":
        default_window, default_level = CT_WINDOW_PRESETS[window_preset]
    else:
        default_window, default_level = 1500, -600
    
    with col2:
        window_width = st.number_input("Window (W)", 1, 4000, default_window, 50, key="window_width")
    
    with col3:
        window_level = st.number_input("Level (L)", -2000, 2000, default_level, 25, key="window_level")
    
    # Quick preset buttons
    preset_cols = st.columns(6)
    presets_to_show = ["Lung", "Mediastinum", "Bone", "Soft Tissue", "Brain", "Liver"]
    for i, preset_name in enumerate(presets_to_show):
        with preset_cols[i]:
            if st.button(f"🔘 {preset_name}", key=f"preset_{preset_name}", use_container_width=True):
                st.session_state.window_preset = preset_name
                w, l = CT_WINDOW_PRESETS[preset_name]
                st.session_state.window_width = w
                st.session_state.window_level = l
                st.rerun()
    
    # ============ SLICE NAVIGATION ============
    if view_type == "Axial":
        max_slice = image_data.shape[2] - 1
        axis_idx = 2
    elif view_type == "Coronal":
        max_slice = image_data.shape[1] - 1
        axis_idx = 1
    else:  # Sagittal
        max_slice = image_data.shape[0] - 1
        axis_idx = 0
    
    # Find slice with most foreground for default
    foreground_per_slice = []
    for i in range(max_slice + 1):
        if axis_idx == 2:
            foreground_per_slice.append(np.sum(prediction[:, :, i] > 0))
        elif axis_idx == 1:
            foreground_per_slice.append(np.sum(prediction[:, i, :] > 0))
        else:
            foreground_per_slice.append(np.sum(prediction[i, :, :] > 0))
    
    default_slice = int(np.argmax(foreground_per_slice)) if max(foreground_per_slice) > 0 else max_slice // 2
    
    # Slice slider with navigation buttons
    col1, col2, col3 = st.columns([1, 6, 1])
    
    with col1:
        if st.button("◀", key="prev_slice", use_container_width=True):
            if 'current_slice' not in st.session_state:
                st.session_state.current_slice = default_slice
            st.session_state.current_slice = max(0, st.session_state.current_slice - 1)
    
    with col2:
        if 'current_slice' not in st.session_state:
            st.session_state.current_slice = default_slice
        slice_idx = st.slider("Slice", 0, max_slice, st.session_state.current_slice, 
                             key=f"inf_{view_type.lower()}_aligned")
        st.session_state.current_slice = slice_idx
    
    with col3:
        if st.button("▶", key="next_slice", use_container_width=True):
            if 'current_slice' not in st.session_state:
                st.session_state.current_slice = default_slice
            st.session_state.current_slice = min(max_slice, st.session_state.current_slice + 1)
    
    # Jump to max foreground slice button
    if st.button("🎯 Jump to Max Detection", key="jump_max", use_container_width=True):
        st.session_state.current_slice = int(np.argmax(foreground_per_slice))
        st.rerun()
    
    # ============ EXTRACT AND PROCESS SLICES ============
    if view_type == "Axial":
        img_slice = image_data[:, :, slice_idx]
        pred_slice = prediction[:, :, slice_idx]
        conf_slice = confidence[:, :, slice_idx] if confidence is not None else None
        unc_slice = uncertainty[:, :, slice_idx] if uncertainty is not None else None
    elif view_type == "Coronal":
        img_slice = image_data[:, slice_idx, :]
        pred_slice = prediction[:, slice_idx, :]
        conf_slice = confidence[:, slice_idx, :] if confidence is not None else None
        unc_slice = uncertainty[:, slice_idx, :] if uncertainty is not None else None
    else:  # Sagittal
        img_slice = image_data[slice_idx, :, :]
        pred_slice = prediction[slice_idx, :, :]
        conf_slice = confidence[slice_idx, :, :] if confidence is not None else None
        unc_slice = uncertainty[slice_idx, :, :] if uncertainty is not None else None
    
    # Apply windowing to image
    img_windowed = apply_window_level(img_slice, window_width, window_level)
    
    # ============ CREATE VISUALIZATION ============
    num_panels = 3
    fig, axes = plt.subplots(1, num_panels, figsize=(5 * num_panels, 5))
    
    # Panel 1: Windowed original image
    axes[0].imshow(img_windowed.T, cmap="gray", origin="lower", vmin=0, vmax=1)
    axes[0].set_title(f"Original (W:{window_width} L:{window_level})")
    axes[0].axis("off")
    
    # Panel 2: Segmentation mask
    axes[1].imshow(pred_slice.T, cmap="jet", origin="lower", vmin=0, vmax=max(1, prediction.max()))
    axes[1].set_title("Segmentation Mask")
    axes[1].axis("off")
    
    # Panel 3: Overlay based on selection
    axes[2].imshow(img_windowed.T, cmap="gray", origin="lower", vmin=0, vmax=1)
    
    if overlay_type == "Segmentation":
        pred_overlay = np.ma.masked_where(pred_slice == 0, pred_slice)
        axes[2].imshow(pred_overlay.T, cmap="jet", alpha=overlay_opacity, origin="lower")
        axes[2].set_title("Segmentation Overlay")
    elif overlay_type == "Confidence" and conf_slice is not None:
        conf_overlay = np.ma.masked_where(pred_slice == 0, conf_slice)
        im = axes[2].imshow(conf_overlay.T, cmap="RdYlGn", alpha=overlay_opacity, origin="lower", vmin=0, vmax=1)
        axes[2].set_title("Confidence Overlay")
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04, label="Confidence")
    elif overlay_type == "Uncertainty" and unc_slice is not None:
        unc_overlay = np.ma.masked_where(pred_slice == 0, unc_slice)
        im = axes[2].imshow(unc_overlay.T, cmap="hot", alpha=overlay_opacity, origin="lower")
        axes[2].set_title("Uncertainty Overlay")
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04, label="Uncertainty")
    else:
        pred_overlay = np.ma.masked_where(pred_slice == 0, pred_slice)
        axes[2].imshow(pred_overlay.T, cmap="jet", alpha=overlay_opacity, origin="lower")
        axes[2].set_title("Overlay")
    
    axes[2].axis("off")
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    
    # ============ SLICE INFO ============
    foreground_in_slice = int(np.sum(pred_slice > 0))
    st.caption(f"Slice {slice_idx}/{max_slice} | {view_type} | Foreground voxels: {foreground_in_slice:,}")
    
    # Show stats for this slice based on overlay type
    if foreground_in_slice > 0:
        mask = pred_slice > 0
        if overlay_type == "Confidence" and conf_slice is not None:
            mean_conf = float(np.mean(conf_slice[mask]))
            min_conf = float(np.min(conf_slice[mask]))
            st.caption(f"Confidence: mean={mean_conf:.3f}, min={min_conf:.3f}")
        elif overlay_type == "Uncertainty" and unc_slice is not None:
            mean_unc = float(np.mean(unc_slice[mask]))
            max_unc = float(np.max(unc_slice[mask]))
            st.caption(f"Uncertainty: mean={mean_unc:.4f}, max={max_unc:.4f}")


def display_history_visualization(
    image_path: str,
    prediction_path: str,
    key_prefix: str = "hist"
):
    """
    Display visualization for a historical inference result.
    Loads image and prediction from disk and displays with controls.
    
    Args:
        image_path: Path to original image NIfTI file
        prediction_path: Path to saved prediction NIfTI file
        key_prefix: Prefix for Streamlit widget keys to avoid conflicts
    """
    from pathlib import Path
    
    image_path = Path(image_path)
    prediction_path = Path(prediction_path)
    
    # Check if files exist
    if not image_path.exists():
        st.warning(f"⚠️ Original image not found: {image_path}")
        return
    
    if not prediction_path.exists():
        st.warning(f"⚠️ Prediction file not found: {prediction_path}")
        return
    
    try:
        # Load image and prediction
        img = nib.load(str(image_path))
        image_data = img.get_fdata()
        
        pred_img = nib.load(str(prediction_path))
        prediction = pred_img.get_fdata()
        
        # Verify shapes match
        if image_data.shape != prediction.shape:
            st.warning(f"⚠️ Shape mismatch: Image {image_data.shape} vs Prediction {prediction.shape}")
            min_shape = tuple(min(a, b) for a, b in zip(image_data.shape[:3], prediction.shape[:3]))
            image_data = image_data[:min_shape[0], :min_shape[1], :min_shape[2]]
            prediction = prediction[:min_shape[0], :min_shape[1], :min_shape[2]]
        else:
            st.success("✅ Prediction aligned with original image space")
        
        # ============ VISUALIZATION CONTROLS ============
        st.markdown("#### Display Controls")
        
        # Row 1: View and overlay controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            view_type = st.radio("View", ["Axial", "Coronal", "Sagittal"], horizontal=True, key=f"{key_prefix}_view")
        
        with col2:
            overlay_type = st.radio("Overlay", ["Segmentation"], horizontal=True, key=f"{key_prefix}_overlay_type")
        
        with col3:
            overlay_opacity = st.slider("Overlay Opacity", 0.0, 1.0, 0.5, 0.1, key=f"{key_prefix}_opacity")
        
        # Row 2: Window/Level Controls
        st.markdown("#### CT Window/Level")
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            window_preset = st.selectbox(
                "Preset", 
                list(CT_WINDOW_PRESETS.keys()),
                index=0,
                key=f"{key_prefix}_preset"
            )
        
        # Get default values from preset
        if window_preset != "Custom":
            default_window, default_level = CT_WINDOW_PRESETS[window_preset]
        else:
            default_window, default_level = 1500, -600
        
        with col2:
            window_width = st.number_input("Window (W)", 1, 4000, default_window, 50, key=f"{key_prefix}_window")
        
        with col3:
            window_level = st.number_input("Level (L)", -2000, 2000, default_level, 25, key=f"{key_prefix}_level")
        
        # ============ SLICE NAVIGATION ============
        if view_type == "Axial":
            max_slice = image_data.shape[2] - 1
            axis_idx = 2
        elif view_type == "Coronal":
            max_slice = image_data.shape[1] - 1
            axis_idx = 1
        else:  # Sagittal
            max_slice = image_data.shape[0] - 1
            axis_idx = 0
        
        # Find slice with most foreground for default
        foreground_per_slice = []
        for i in range(max_slice + 1):
            if axis_idx == 2:
                foreground_per_slice.append(np.sum(prediction[:, :, i] > 0))
            elif axis_idx == 1:
                foreground_per_slice.append(np.sum(prediction[:, i, :] > 0))
            else:
                foreground_per_slice.append(np.sum(prediction[i, :, :] > 0))
        
        default_slice = int(np.argmax(foreground_per_slice)) if max(foreground_per_slice) > 0 else max_slice // 2
        
        # Slice slider with navigation
        col1, col2, col3 = st.columns([1, 6, 1])
        
        slice_state_key = f"{key_prefix}_current_slice"
        
        with col1:
            if st.button("◀", key=f"{key_prefix}_prev", use_container_width=True):
                if slice_state_key not in st.session_state:
                    st.session_state[slice_state_key] = default_slice
                st.session_state[slice_state_key] = max(0, st.session_state[slice_state_key] - 1)
        
        with col2:
            if slice_state_key not in st.session_state:
                st.session_state[slice_state_key] = default_slice
            slice_idx = st.slider("Slice", 0, max_slice, st.session_state[slice_state_key], key=f"{key_prefix}_slice")
            st.session_state[slice_state_key] = slice_idx
        
        with col3:
            if st.button("▶", key=f"{key_prefix}_next", use_container_width=True):
                if slice_state_key not in st.session_state:
                    st.session_state[slice_state_key] = default_slice
                st.session_state[slice_state_key] = min(max_slice, st.session_state[slice_state_key] + 1)
        
        # Jump to max detection button
        if st.button("🎯 Jump to Max Detection", key=f"{key_prefix}_jump", use_container_width=True):
            st.session_state[slice_state_key] = int(np.argmax(foreground_per_slice))
            st.rerun()
        
        # ============ EXTRACT AND PROCESS SLICES ============
        if view_type == "Axial":
            img_slice = image_data[:, :, slice_idx]
            pred_slice = prediction[:, :, slice_idx]
        elif view_type == "Coronal":
            img_slice = image_data[:, slice_idx, :]
            pred_slice = prediction[:, slice_idx, :]
        else:  # Sagittal
            img_slice = image_data[slice_idx, :, :]
            pred_slice = prediction[slice_idx, :, :]
        
        # Apply windowing to image
        img_windowed = apply_window_level(img_slice, window_width, window_level)
        
        # ============ CREATE VISUALIZATION ============
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Panel 1: Windowed original image
        axes[0].imshow(img_windowed.T, cmap="gray", origin="lower", vmin=0, vmax=1)
        axes[0].set_title(f"Original (W:{window_width} L:{window_level})")
        axes[0].axis("off")
        
        # Panel 2: Segmentation mask
        axes[1].imshow(pred_slice.T, cmap="jet", origin="lower", vmin=0, vmax=max(1, prediction.max()))
        axes[1].set_title("Segmentation Mask")
        axes[1].axis("off")
        
        # Panel 3: Overlay
        axes[2].imshow(img_windowed.T, cmap="gray", origin="lower", vmin=0, vmax=1)
        pred_overlay = np.ma.masked_where(pred_slice == 0, pred_slice)
        axes[2].imshow(pred_overlay.T, cmap="jet", alpha=overlay_opacity, origin="lower")
        axes[2].set_title("Segmentation Overlay")
        axes[2].axis("off")
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        # ============ SLICE INFO ============
        foreground_in_slice = int(np.sum(pred_slice > 0))
        st.caption(f"Slice {slice_idx}/{max_slice} | {view_type} | Foreground voxels: {foreground_in_slice:,}")
        
    except Exception as e:
        st.error(f"❌ Error loading visualization: {e}")
        st.exception(e)


def display_prediction_overlay(
    image_path: Path,
    prediction: np.ndarray,
    transforms: Compose
):
    """Display image with prediction overlay (legacy function - uses preprocessed space)"""
    
    # Load and preprocess image for display
    data_dict = {"image": str(image_path)}
    transformed = transforms(data_dict)
    image_data = transformed["image"].numpy()[0]  # Remove channel dim
    
    # Ensure shapes match (crop or pad prediction if needed)
    if image_data.shape != prediction.shape:
        st.warning(f"⚠️ Shape mismatch: Image {image_data.shape} vs Prediction {prediction.shape}")
        # Use the smaller dimensions
        min_shape = tuple(min(a, b) for a, b in zip(image_data.shape, prediction.shape))
        image_data = image_data[:min_shape[0], :min_shape[1], :min_shape[2]]
        prediction = prediction[:min_shape[0], :min_shape[1], :min_shape[2]]
    
    # View selector
    view_type = st.radio("View", ["Axial", "Coronal", "Sagittal"], horizontal=True)
    
    if view_type == "Axial":
        max_slice = image_data.shape[2] - 1
        default_slice = max_slice // 2
        slice_idx = st.slider("Slice", 0, max_slice, default_slice, key="inf_axial")
        img_slice = image_data[:, :, slice_idx]
        pred_slice = prediction[:, :, slice_idx]
    elif view_type == "Coronal":
        max_slice = image_data.shape[1] - 1
        default_slice = max_slice // 2
        slice_idx = st.slider("Slice", 0, max_slice, default_slice, key="inf_coronal")
        img_slice = image_data[:, slice_idx, :]
        pred_slice = prediction[:, slice_idx, :]
    else:  # Sagittal
        max_slice = image_data.shape[0] - 1
        default_slice = max_slice // 2
        slice_idx = st.slider("Slice", 0, max_slice, default_slice, key="inf_sagittal")
        img_slice = image_data[slice_idx, :, :]
        pred_slice = prediction[slice_idx, :, :]
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(img_slice.T, cmap="gray", origin="lower")
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    # Prediction only
    axes[1].imshow(pred_slice.T, cmap="jet", origin="lower")
    axes[1].set_title("Prediction")
    axes[1].axis("off")
    
    # Overlay
    axes[2].imshow(img_slice.T, cmap="gray", origin="lower")
    pred_overlay = np.ma.masked_where(pred_slice == 0, pred_slice)
    axes[2].imshow(pred_overlay.T, cmap="jet", alpha=0.5, origin="lower")
    axes[2].set_title("Overlay")
    axes[2].axis("off")
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    
    # Slice info
    foreground_in_slice = np.sum(pred_slice > 0)
    st.caption(f"Slice {slice_idx} | Foreground voxels in slice: {foreground_in_slice:,}")


def render_history_tab(bids_dir):
    """Render the inference history tab"""
    
    if not HISTORY_AVAILABLE:
        st.warning("History module not available.")
        return
    
    if bids_dir is None:
        st.info("Select a dataset to view inference history.")
        return
    
    # Initialize history manager
    try:
        history_manager = get_history_manager(str(bids_dir))
    except Exception as e:
        st.error(f"Error loading history: {e}")
        return
    
    # Statistics overview
    stats = history_manager.get_statistics()
    
    st.markdown("#### Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Runs", stats.get("total_runs", 0))
    with col2:
        st.metric("Unique Subjects", stats.get("unique_subjects", 0))
    with col3:
        st.metric("Total Nodules", stats.get("total_nodules_detected", 0))
    with col4:
        st.metric("Total Suspicious", stats.get("total_suspicious", 0))
    
    st.markdown("---")
    
    # Filters
    st.markdown("#### Filters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        subject_filter = st.text_input("Filter by Subject ID", "", key="hist_subject_filter")
    with col2:
        model_filter = st.text_input("Filter by Model Name", "", key="hist_model_filter")
    with col3:
        limit = st.number_input("Max entries", 10, 100, 20, key="hist_limit")
    
    # Get history
    history = history_manager.get_history(
        limit=limit,
        subject_filter=subject_filter if subject_filter else None,
        model_filter=model_filter if model_filter else None
    )
    
    if not history:
        st.info("No inference history found. Run inference to create entries.")
        return
    
    st.markdown("#### History Entries")
    
    # Create table data
    history_data = []
    for entry in history:
        timestamp = entry.get("timestamp", "")
        try:
            # Format timestamp nicely
            dt = datetime.fromisoformat(timestamp)
            formatted_time = dt.strftime("%Y-%m-%d %H:%M")
        except:
            formatted_time = timestamp[:16] if len(timestamp) > 16 else timestamp
        
        row = {
            "ID": entry.get("id", "")[:15] + "...",
            "Time": formatted_time,
            "Subject": entry.get("subject_id", "N/A"),
            "Model": entry.get("model_name", "N/A")[:20],
            "Nodules": entry.get("nodule_count", 0),
            "Suspicious": entry.get("suspicious_count", 0),
            "Max Ø (mm)": f"{entry.get('stats', {}).get('max_diameter_mm', 0):.1f}"
        }
        history_data.append(row)
    
    st.dataframe(history_data, use_container_width=True)
    
    # Entry details viewer
    st.markdown("---")
    st.markdown("#### Entry Details")
    
    entry_ids = [e.get("id", "") for e in history]
    selected_entry_id = st.selectbox(
        "Select entry to view details",
        entry_ids,
        format_func=lambda x: x[:30] + "..." if len(x) > 30 else x,
        key="selected_history_entry"
    )
    
    if selected_entry_id:
        entry = history_manager.get_entry(selected_entry_id)
        if entry:
            display_history_entry_details(entry)
    
    # Comparison tool
    st.markdown("---")
    st.markdown("#### Compare Entries")
    
    if len(entry_ids) >= 2:
        col1, col2 = st.columns(2)
        
        with col1:
            compare_entry_1 = st.selectbox(
                "First Entry",
                entry_ids,
                index=0,
                key="compare_entry_1"
            )
        
        with col2:
            compare_entry_2 = st.selectbox(
                "Second Entry",
                entry_ids,
                index=min(1, len(entry_ids) - 1),
                key="compare_entry_2"
            )
        
        if st.button("🔍 Compare Entries", key="compare_btn"):
            if compare_entry_1 and compare_entry_2 and compare_entry_1 != compare_entry_2:
                comparison = history_manager.compare_entries(compare_entry_1, compare_entry_2)
                display_comparison_results(comparison)
            else:
                st.warning("Please select two different entries to compare.")
    else:
        st.info("Need at least 2 history entries to compare.")


def display_history_entry_details(entry: Dict, entry_key: str = ""):
    """Display detailed information for a history entry - matching live inference results view
    
    Args:
        entry: History entry dict
        entry_key: Unique key suffix for Streamlit widgets to avoid conflicts
    """
    
    # Show inference source indicator (same as live results)
    inference_source = entry.get("inference_source", "local")
    if inference_source == "cloud":
        st.markdown("### ☁️ Cloud Inference Results")
        st.caption("Inference ran on Modal's serverless T4 GPU")
    elif inference_source == "cloud_fallback":
        st.markdown("### 🔄 Fallback Inference Results")
        st.caption("Cloud was unavailable, inference ran locally")
    else:
        st.markdown("### 🖥️ Local Inference Results")
        st.caption("Inference ran on local device")
    
    # Basic info section (collapsible)
    with st.expander("📋 Basic Info", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"- **ID:** {entry.get('id', 'N/A')}")
            st.write(f"- **Timestamp:** {entry.get('timestamp', 'N/A')}")
            st.write(f"- **Subject:** {entry.get('subject_id', 'N/A')}")
        
        with col2:
            st.write(f"- **Model:** {entry.get('model_name', 'N/A')}")
            st.write(f"- **Image:** {entry.get('image_path', 'N/A')}")
            pred_path = entry.get("prediction_path")
            if pred_path:
                st.write(f"- **Saved to:** `{pred_path}`")
        
        # Parameters used
        params = entry.get("parameters", {})
        if params:
            st.markdown("**Parameters Used:**")
            st.json(params)
    
    # Timing section (same layout as live results)
    timing_info = entry.get("timing_info", {})
    if timing_info:
        st.markdown("---")
        st.markdown("### ⏱️ Inference Timing")
        display_timing_info(timing_info, inference_source=inference_source)
    
    st.markdown("---")
    st.markdown("### 📊 Results")
    
    # Prediction metrics - show basic stats
    stats = entry.get("stats", {})
    metadata = entry.get("metadata", {})
    original_spacing = metadata.get("original_spacing")
    
    # Convert spacing from strings if needed (historical data may have string values)
    if original_spacing:
        try:
            original_spacing = tuple(float(s) for s in original_spacing)
        except (TypeError, ValueError):
            original_spacing = None
    
    # Basic metrics display
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎯 Nodules Detected", entry.get('nodule_count', 0))
    with col2:
        suspicious = entry.get('suspicious_count', 0)
        st.metric("⚠️ Suspicious (4A+)", suspicious,
                  delta=None if suspicious == 0 else "Needs review",
                  delta_color="inverse" if suspicious > 0 else "off")
    with col3:
        st.metric("📐 Total Volume", f"{stats.get('total_volume_mm3', 0):.1f} mm³")
    with col4:
        st.metric("📏 Max Diameter", f"{stats.get('max_diameter_mm', 0):.1f} mm")
    
    # Additional metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Mean Confidence", f"{stats.get('mean_confidence', 0):.3f}")
    with col2:
        st.metric("📏 Mean Diameter", f"{stats.get('mean_diameter_mm', 0):.1f} mm")
    with col3:
        original_shape = metadata.get("original_shape", [])
        if original_shape:
            st.metric("📐 Image Shape", f"{original_shape[0]}×{original_shape[1]}×{original_shape[2]}")
    with col4:
        if original_spacing:
            st.metric("📏 Voxel Size", f"{original_spacing[0]:.2f}×{original_spacing[1]:.2f}×{original_spacing[2]:.2f} mm")
    
    # Uncertainty results (if available)
    uncertainty_metrics = entry.get("uncertainty_metrics")
    if uncertainty_metrics:
        st.markdown("---")
        st.markdown("### 🎲 Uncertainty Analysis")
        display_uncertainty_results(uncertainty_metrics)
    
    # Nodule detection results (same layout as live results)
    nodule_candidates = entry.get("nodule_candidates", [])
    if nodule_candidates or entry.get("nodule_count", 0) > 0:
        st.markdown("---")
        st.markdown("### 🔬 Nodule Detection")
        display_nodule_results(nodule_candidates, stats, show_export=False)
    
    # Visualization section (load from saved files)
    image_path = entry.get("image_path")
    prediction_path = entry.get("prediction_path")
    
    if image_path and prediction_path:
        st.markdown("---")
        st.markdown("### 🖼️ Visualization")
        
        # Use entry ID as key prefix to ensure unique widget keys
        entry_id = entry.get("id", "default")
        # Clean the ID to be a valid key (remove special characters)
        key_prefix = f"hist_{entry_id.replace('-', '_').replace('.', '_')}"
        
        display_history_visualization(image_path, prediction_path, key_prefix=key_prefix)


def display_comparison_results(comparison: Dict):
    """Display comparison between two history entries"""
    
    if "error" in comparison:
        st.error(comparison["error"])
        return
    
    st.markdown("**Comparison Results:**")
    
    col1, col2 = st.columns(2)
    
    e1 = comparison.get("entry_1", {})
    e2 = comparison.get("entry_2", {})
    diff = comparison.get("differences", {})
    
    with col1:
        st.markdown(f"**Entry 1:** {e1.get('id', 'N/A')[:20]}...")
        st.write(f"- Time: {e1.get('timestamp', 'N/A')[:16]}")
        st.write(f"- Model: {e1.get('model', 'N/A')}")
        st.write(f"- Nodules: {e1.get('nodules', 0)}")
        st.write(f"- Suspicious: {e1.get('suspicious', 0)}")
    
    with col2:
        st.markdown(f"**Entry 2:** {e2.get('id', 'N/A')[:20]}...")
        st.write(f"- Time: {e2.get('timestamp', 'N/A')[:16]}")
        st.write(f"- Model: {e2.get('model', 'N/A')}")
        st.write(f"- Nodules: {e2.get('nodules', 0)}")
        st.write(f"- Suspicious: {e2.get('suspicious', 0)}")
    
    st.markdown("**Changes:**")
    
    nodule_diff = diff.get("nodule_count_diff", 0)
    suspicious_diff = diff.get("suspicious_diff", 0)
    diameter_diff = diff.get("max_diameter_diff", 0)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        delta_color = "normal" if nodule_diff == 0 else ("inverse" if nodule_diff > 0 else "off")
        st.metric("Nodule Count Change", nodule_diff, delta=f"{'+' if nodule_diff > 0 else ''}{nodule_diff}")
    
    with col2:
        st.metric("Suspicious Change", suspicious_diff, delta=f"{'+' if suspicious_diff > 0 else ''}{suspicious_diff}")
    
    with col3:
        st.metric("Max Diameter Change", f"{diameter_diff:.1f} mm", delta=f"{'+' if diameter_diff > 0 else ''}{diameter_diff:.1f}")
    
    # Interpretation
    time_diff = diff.get("time_diff_hours", 0)
    same_model = diff.get("same_model", False)
    
    st.markdown("**Interpretation:**")
    if same_model:
        if nodule_diff == 0 and suspicious_diff == 0:
            st.success("✅ Results are consistent between runs.")
        elif nodule_diff > 0 or suspicious_diff > 0:
            st.warning(f"⚠️ More nodules/suspicious findings in Entry 2. Time elapsed: {time_diff:.1f} hours.")
        else:
            st.info(f"ℹ️ Fewer nodules in Entry 2. This may indicate different preprocessing or thresholds.")
    else:
        st.info(f"ℹ️ Different models used. Direct comparison may not be meaningful.")



# ============================================================
# PAGE ENTRY POINT
# ============================================================

render()

