"""
Modal Labs Cloud Inference Provider

Provides serverless GPU inference using Modal Labs.
Features:
- Model weights stored in Modal Volumes for fast loading
- GPU-accelerated inference (T4 by default)
- Automatic preprocessing on the cloud
- Fallback to local inference on failure

Setup:
1. pip install modal
2. modal token new
3. Deploy: modal deploy src/cloud/modal_inference.py
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

# Check if Modal is available
try:
    import modal
    MODAL_AVAILABLE = True
except ImportError:
    MODAL_AVAILABLE = False
    modal = None
    logger.warning("Modal not installed. Install with: pip install modal")


def is_modal_available() -> bool:
    """Check if Modal is installed and configured"""
    if not MODAL_AVAILABLE:
        return False
    try:
        # Check if token is configured
        modal_config_path = os.path.expanduser("~/.modal.toml")
        return os.path.exists(modal_config_path)
    except Exception:
        return False


# ============================================================================
# MODAL APP DEFINITION (for deployment)
# These functions are SELF-CONTAINED and run in Modal's cloud container
# They should NOT import from local project modules
# ============================================================================

if MODAL_AVAILABLE:
    # Define the Modal app
    app = modal.App("medai-inference")
    
    # Create a volume to store model weights (persistent across runs)
    model_volume = modal.Volume.from_name("medai-models", create_if_missing=True)
    
    # Define the container image with all dependencies
    inference_image = (
        modal.Image.debian_slim(python_version="3.10")
        .pip_install([
            "torch>=2.0.0",
            "monai>=1.4.0",
            "nibabel>=5.1.0",
            "numpy>=1.24.0",
            "scipy>=1.11.0",
            "einops>=0.7.0",
        ])
    )
    
    @app.function(
        image=inference_image,
        gpu="T4",
        timeout=600,
        volumes={"/models": model_volume},
    )
    def run_inference_modal(
        image_data: bytes,
        model_name: str,
        model_config: Dict[str, Any],
        inference_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Modal function for running inference on the cloud.
        This function is SELF-CONTAINED and runs in Modal's cloud container.
        """
        import torch
        import nibabel as nib
        import numpy as np
        import tempfile
        import time as time_module
        import os as os_module
        from pathlib import Path as PathLib
        
        timing = {"total": 0, "model_loading": 0, "preprocessing": 0, "inference": 0}
        start_total = time_module.time()
        
        try:
            # Import MONAI components
            from monai.networks.nets import SwinUNETR, SegResNet
            from monai.inferers import sliding_window_inference
            from monai.transforms import (
                Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
                Orientationd, ScaleIntensityRanged, CropForegroundd,
                DivisiblePadd, ToTensord
            )
            from scipy import ndimage
            
            # Save image to temp file for MONAI loading
            with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
                f.write(image_data)
                temp_image_path = f.name
            
            # Load original image metadata
            original_img = nib.load(temp_image_path)
            original_shape = original_img.shape
            original_spacing = original_img.header.get_zooms()[:3]
            
            # Device setup
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Model loading
            start_model = time_module.time()
            model_path = PathLib(f"/models/{model_name}")
            checkpoint_path = model_path / "best_model.pth"
            
            if not checkpoint_path.exists():
                return {
                    "error": f"Model not found in volume: {model_name}. Please upload the model first.",
                    "timing": timing
                }
            
            # Create model based on config
            model_type = model_config.get("model", {}).get("type", "SwinUNETR")
            in_channels = model_config.get("model", {}).get("in_channels", 1)
            out_channels = model_config.get("model", {}).get("out_channels", 2)
            img_size = tuple(model_config.get("model", {}).get("img_size", [96, 96, 96]))
            feature_size = model_config.get("model", {}).get("feature_size", 48)
            
            if model_type == "SwinUNETR":
                # Note: MONAI SwinUNETR doesn't use img_size parameter
                # It handles variable input sizes automatically
                model = SwinUNETR(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    feature_size=feature_size,
                    use_checkpoint=False,
                    spatial_dims=3,
                )
            else:
                model = SegResNet(
                    spatial_dims=3,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    init_filters=16,
                )
            
            # Load weights
            checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint)
            
            model = model.to(device)
            model.eval()
            timing["model_loading"] = time_module.time() - start_model
            
            # Preprocessing
            start_preprocess = time_module.time()
            preproc = model_config.get("preprocessing", {})
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
            
            data_dict = {"image": temp_image_path}
            transformed = transforms(data_dict)
            image_tensor = transformed["image"].unsqueeze(0).to(device)
            preprocessed_shape = image_tensor.shape[2:]
            timing["preprocessing"] = time_module.time() - start_preprocess
            
            # Inference
            start_inference = time_module.time()
            roi_size = img_size
            overlap = inference_params.get("overlap", 0.5)
            sw_batch_size = inference_params.get("sw_batch_size", 2)
            
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=True):
                    pred_tensor = sliding_window_inference(
                        inputs=image_tensor,
                        roi_size=roi_size,
                        sw_batch_size=sw_batch_size,
                        predictor=model,
                        overlap=overlap
                    )
            
            timing["inference"] = time_module.time() - start_inference
            
            # Post-process
            pred_np = pred_tensor.cpu().numpy()[0]
            
            if pred_np.shape[0] > 1:
                pred_softmax = np.exp(pred_np) / np.sum(np.exp(pred_np), axis=0, keepdims=True)
                pred_seg = np.argmax(pred_softmax, axis=0).astype(np.uint8)
                confidence = np.max(pred_softmax, axis=0).astype(np.float32)
            else:
                pred_seg = (pred_np[0] > 0.5).astype(np.uint8)
                confidence = pred_np[0].astype(np.float32)
            
            # Resample to original space
            def resample_to_original(prediction, orig_shape, preproc_shape, mode='nearest'):
                zoom_factors = tuple(o / p for o, p in zip(orig_shape[:3], preproc_shape[:3]))
                order = 0 if mode == 'nearest' else 1
                resampled = ndimage.zoom(prediction, zoom_factors, order=order)
                final = np.zeros(orig_shape[:3], dtype=prediction.dtype)
                min_shape = tuple(min(a, b) for a, b in zip(resampled.shape, orig_shape[:3]))
                final[:min_shape[0], :min_shape[1], :min_shape[2]] = \
                    resampled[:min_shape[0], :min_shape[1], :min_shape[2]]
                return final
            
            pred_resampled = resample_to_original(
                pred_seg, original_shape, tuple(preprocessed_shape), 'nearest'
            )
            confidence_resampled = resample_to_original(
                confidence, original_shape, tuple(preprocessed_shape), 'linear'
            )
            
            timing["total"] = time_module.time() - start_total
            
            # Clean up temp file
            os_module.unlink(temp_image_path)
            
            return {
                "prediction": pred_resampled.tobytes(),
                "prediction_shape": list(pred_resampled.shape),
                "prediction_dtype": str(pred_resampled.dtype),
                "confidence": confidence_resampled.tobytes(),
                "confidence_shape": list(confidence_resampled.shape),
                "confidence_dtype": str(confidence_resampled.dtype),
                "original_spacing": list(original_spacing),
                "timing": timing,
                "device": str(device),
                "error": None
            }
            
        except Exception as e:
            import traceback
            return {
                "error": f"{str(e)}\n{traceback.format_exc()}",
                "timing": timing
            }
    
    @app.function(
        image=inference_image,
        volumes={"/models": model_volume},
        timeout=300,
    )
    def upload_model_to_volume(model_name: str, model_data: Dict[str, bytes]) -> bool:
        """Upload model files to Modal volume for persistent storage."""
        from pathlib import Path as PathLib
        
        model_path = PathLib(f"/models/{model_name}")
        model_path.mkdir(parents=True, exist_ok=True)
        
        for filename, data in model_data.items():
            file_path = model_path / filename
            with open(file_path, "wb") as f:
                f.write(data)
        
        # Commit changes to volume
        model_volume.commit()
        
        return True
    
    @app.function(
        image=inference_image,
        volumes={"/models": model_volume},
        timeout=60,
    )
    def check_model_exists(model_name: str) -> bool:
        """Check if a model exists in the Modal volume."""
        from pathlib import Path as PathLib
        model_path = PathLib(f"/models/{model_name}")
        return (model_path / "best_model.pth").exists() and (model_path / "config.json").exists()
    
    @app.function(
        image=inference_image,
        timeout=30,
    )
    def health_check() -> Dict[str, Any]:
        """Health check function to verify Modal is working."""
        import torch
        return {
            "status": "healthy",
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }


# ============================================================================
# LOCAL CLIENT CODE
# This code runs locally and imports from the project
# ============================================================================

# Import base classes only for local use (not in Modal container)
# These imports are at module level but will only be used by the client class
try:
    from src.cloud.base import (
        CloudInferenceProvider,
        InferenceResult,
        InferenceSource,
        CloudInferenceError
    )
    BASE_CLASSES_AVAILABLE = True
except ImportError:
    BASE_CLASSES_AVAILABLE = False
    # Define minimal stubs if base classes aren't available
    CloudInferenceProvider = object
    InferenceResult = None
    InferenceSource = None
    CloudInferenceError = Exception


class ModalInferenceProvider(CloudInferenceProvider if BASE_CLASSES_AVAILABLE else object):
    """
    Modal Labs cloud inference provider.
    
    Provides serverless GPU inference using Modal's infrastructure.
    Models are cached in Modal Volumes for fast subsequent inference.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Modal provider."""
        default_config = {
            "gpu": "T4",
            "timeout": 600,
            "fallback_to_local": True,
            "app_name": "medai-inference",
        }
        self.config = {**default_config, **(config or {})}
        self._uploaded_models: set = set()
        self._app_name = self.config.get("app_name", "medai-inference")
        self._is_available: Optional[bool] = None
    
    @property
    def name(self) -> str:
        return "modal"
    
    @property
    def is_available(self) -> bool:
        """Cached availability check"""
        if self._is_available is None:
            self._is_available = self.check_availability()
        return self._is_available
    
    def _get_function(self, function_name: str):
        """Get a reference to a deployed Modal function."""
        if not MODAL_AVAILABLE:
            return None
        return modal.Function.from_name(self._app_name, function_name)
    
    def check_availability(self) -> bool:
        """Check if Modal is available and configured"""
        if not MODAL_AVAILABLE:
            logger.warning("Modal SDK not installed")
            return False
        
        try:
            # Check if Modal token is configured
            modal_config_path = os.path.expanduser("~/.modal.toml")
            if not os.path.exists(modal_config_path):
                logger.warning("Modal token not configured. Run 'modal token new'")
                return False
            
            # Try to look up the deployed function
            try:
                self._get_function("health_check")
                logger.info(f"Modal app '{self._app_name}' found and available")
                return True
            except Exception as e:
                if "NotFoundError" in str(type(e).__name__) or "not found" in str(e).lower():
                    logger.warning(f"Modal app '{self._app_name}' not deployed. Run: modal deploy src/cloud/modal_inference.py")
                else:
                    logger.warning(f"Could not connect to Modal: {e}")
                return False
                
        except Exception as e:
            logger.warning(f"Modal availability check failed: {e}")
            return False
    
    def run_inference(
        self,
        image_path: str,
        model_path: str,
        model_config: Dict[str, Any],
        inference_params: Optional[Dict[str, Any]] = None
    ):
        """Run inference using Modal's serverless GPU."""
        if not BASE_CLASSES_AVAILABLE:
            raise RuntimeError("Base classes not available")
        
        if not MODAL_AVAILABLE:
            return InferenceResult(
                error="Modal SDK not installed. Install with: pip install modal",
                source=InferenceSource.LOCAL
            )
        
        inference_params = inference_params or {}
        timing = {"total": 0, "upload": 0, "cloud_inference": 0}
        start_total = time.time()
        
        try:
            model_path_obj = Path(model_path)
            model_name = model_path_obj.name
            
            # Ensure model is uploaded to Modal volume
            start_upload = time.time()
            if model_name not in self._uploaded_models:
                logger.info(f"Checking if model {model_name} exists in Modal volume...")
                try:
                    check_fn = self._get_function("check_model_exists")
                    exists = check_fn.remote(model_name)
                    if not exists:
                        logger.info(f"Uploading model {model_name} to Modal volume...")
                        self.upload_model(str(model_path))
                    self._uploaded_models.add(model_name)
                except Exception as e:
                    logger.warning(f"Could not check/upload model: {e}")
            timing["upload"] = time.time() - start_upload
            
            # Read image data
            with open(image_path, "rb") as f:
                image_data = f.read()
            
            # Call Modal function
            start_inference = time.time()
            logger.info(f"Running inference on Modal for {Path(image_path).name}...")
            
            inference_fn = self._get_function("run_inference_modal")
            result = inference_fn.remote(
                image_data=image_data,
                model_name=model_name,
                model_config=model_config,
                inference_params=inference_params
            )
            
            timing["cloud_inference"] = time.time() - start_inference
            timing["total"] = time.time() - start_total
            
            # Check for errors
            if result.get("error"):
                raise CloudInferenceError(result["error"], "modal")
            
            # Reconstruct numpy arrays from bytes
            prediction = np.frombuffer(
                result["prediction"], 
                dtype=result["prediction_dtype"]
            ).reshape(result["prediction_shape"])
            
            confidence = np.frombuffer(
                result["confidence"],
                dtype=result["confidence_dtype"]
            ).reshape(result["confidence_shape"])
            
            # Merge timing info
            cloud_timing = result.get("timing", {})
            timing.update({f"cloud_{k}": v for k, v in cloud_timing.items()})
            
            return InferenceResult(
                prediction=prediction,
                confidence=confidence,
                source=InferenceSource.CLOUD,
                timing=timing,
                metadata={
                    "device": result.get("device", "unknown"),
                    "original_spacing": result.get("original_spacing"),
                    "provider": "modal",
                }
            )
            
        except Exception as e:
            logger.error(f"Modal inference failed: {e}")
            timing["total"] = time.time() - start_total
            
            return InferenceResult(
                error=str(e),
                source=InferenceSource.CLOUD,
                timing=timing,
            )
    
    def upload_model(self, model_path: str) -> bool:
        """Upload model to Modal volume for persistent storage."""
        if not MODAL_AVAILABLE:
            return False
        
        model_path_obj = Path(model_path)
        model_name = model_path_obj.name
        
        try:
            # Read model files
            model_data = {}
            
            # Required files
            checkpoint_path = model_path_obj / "best_model.pth"
            config_path = model_path_obj / "config.json"
            
            if not checkpoint_path.exists():
                logger.error(f"Checkpoint not found: {checkpoint_path}")
                return False
            
            if not config_path.exists():
                logger.error(f"Config not found: {config_path}")
                return False
            
            with open(checkpoint_path, "rb") as f:
                model_data["best_model.pth"] = f.read()
            
            with open(config_path, "rb") as f:
                model_data["config.json"] = f.read()
            
            # Optional: training history
            history_path = model_path_obj / "training_history.json"
            if history_path.exists():
                with open(history_path, "rb") as f:
                    model_data["training_history.json"] = f.read()
            
            # Upload to Modal
            logger.info(f"Uploading {len(model_data)} files to Modal volume...")
            upload_fn = self._get_function("upload_model_to_volume")
            success = upload_fn.remote(model_name, model_data)
            
            if success:
                logger.info(f"Model {model_name} uploaded successfully")
                self._uploaded_models.add(model_name)
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to upload model: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get Modal provider status"""
        return {
            "provider": self.name,
            "available": self.is_available,
            "config": {k: v for k, v in self.config.items() if k not in ["token", "secret", "api_key"]},
            "uploaded_models": list(self._uploaded_models),
            "app_name": self._app_name
        }


# ============================================================================
# CONVENIENCE FUNCTIONS FOR DIRECT USE
# ============================================================================

def predict_remote(
    image_path: str,
    model_path: str,
    overlap: float = 0.5,
    sw_batch_size: int = 2
):
    """
    Convenience function for running inference on Modal.
    
    Args:
        image_path: Path to NIfTI image
        model_path: Path to model directory
        overlap: Sliding window overlap (default: 0.5)
        sw_batch_size: Sliding window batch size (default: 2)
        
    Returns:
        InferenceResult with prediction
    """
    if not BASE_CLASSES_AVAILABLE:
        raise RuntimeError("Base classes not available")
    
    # Load config
    config_path = Path(model_path) / "config.json"
    with open(config_path, "r") as f:
        model_config = json.load(f)
    
    # Create provider and run
    provider = ModalInferenceProvider()
    return provider.run_inference(
        image_path=image_path,
        model_path=model_path,
        model_config=model_config,
        inference_params={
            "overlap": overlap,
            "sw_batch_size": sw_batch_size
        }
    )
