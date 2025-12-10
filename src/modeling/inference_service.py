"""
Unified Inference Service

Provides a single interface for running model inference with automatic
cloud/local selection and fallback capabilities.

Usage:
    from src.modeling.inference_service import InferenceService
    
    service = InferenceService()
    result = service.run_inference(
        image_path="path/to/image.nii.gz",
        model_path="path/to/model",
        use_cloud=True  # Will fallback to local if cloud fails
    )
"""

import os
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from enum import Enum
import json

import torch
import numpy as np
import nibabel as nib

from src.cloud.base import (
    CloudInferenceProvider,
    InferenceResult,
    InferenceSource,
    CloudInferenceError
)

logger = logging.getLogger(__name__)


class InferenceMode(Enum):
    """Inference execution mode"""
    LOCAL = "local"           # Always use local inference
    CLOUD = "cloud"           # Always use cloud (fail if unavailable)
    CLOUD_PREFERRED = "cloud_preferred"  # Prefer cloud, fallback to local
    AUTO = "auto"             # Automatically select based on availability and image size


class InferenceService:
    """
    Unified inference service with cloud/local abstraction.
    
    Features:
    - Automatic cloud provider detection and selection
    - Fallback to local inference when cloud is unavailable
    - Consistent result format regardless of execution location
    - Performance monitoring and logging
    """
    
    def __init__(
        self,
        default_mode: InferenceMode = InferenceMode.CLOUD_PREFERRED,
        cloud_config: Optional[Dict[str, Any]] = None,
        local_device: Optional[str] = None
    ):
        """
        Initialize the inference service.
        
        Args:
            default_mode: Default inference mode
            cloud_config: Configuration for cloud provider
            local_device: Device for local inference ('cuda', 'cpu', or None for auto)
        """
        self.default_mode = default_mode
        self.cloud_config = cloud_config or {}
        self.local_device = local_device
        
        # Initialize cloud provider lazily
        self._cloud_provider: Optional[CloudInferenceProvider] = None
        self._cloud_checked = False
        
        # Track usage statistics
        self._stats = {
            "total_inferences": 0,
            "cloud_inferences": 0,
            "local_inferences": 0,
            "cloud_failures": 0,
            "fallbacks": 0,
        }
    
    @property
    def cloud_provider(self) -> Optional[CloudInferenceProvider]:
        """Lazy initialization of cloud provider"""
        if not self._cloud_checked:
            self._cloud_checked = True
            try:
                from src.cloud.modal_inference import ModalInferenceProvider, is_modal_available
                if is_modal_available():
                    self._cloud_provider = ModalInferenceProvider(self.cloud_config)
                    logger.info("Modal cloud provider initialized")
                else:
                    logger.info("Modal not available, cloud inference disabled")
            except ImportError as e:
                logger.warning(f"Could not import cloud provider: {e}")
        return self._cloud_provider
    
    @property
    def cloud_available(self) -> bool:
        """Check if cloud inference is available"""
        return self.cloud_provider is not None and self.cloud_provider.is_available
    
    def get_device(self) -> torch.device:
        """Get the appropriate device for local inference"""
        if self.local_device:
            return torch.device(self.local_device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def run_inference(
        self,
        image_path: str,
        model_path: str,
        mode: Optional[InferenceMode] = None,
        inference_params: Optional[Dict[str, Any]] = None,
        enable_uncertainty: bool = False,
        num_mc_passes: int = 10,
    ) -> InferenceResult:
        """
        Run model inference on an image.
        
        Args:
            image_path: Path to NIfTI image file
            model_path: Path to model directory (containing config.json and best_model.pth)
            mode: Inference mode (overrides default)
            inference_params: Additional parameters (overlap, sw_batch_size, etc.)
            enable_uncertainty: Enable Monte Carlo dropout for uncertainty estimation
            num_mc_passes: Number of MC passes for uncertainty (if enabled)
            
        Returns:
            InferenceResult with prediction, confidence, timing, and metadata
        """
        mode = mode or self.default_mode
        inference_params = inference_params or {}
        
        # Set defaults for inference params
        inference_params.setdefault("overlap", 0.5)
        inference_params.setdefault("sw_batch_size", 2)
        
        # Load model config
        model_path = Path(model_path)
        config_path = model_path / "config.json"
        
        if not config_path.exists():
            return InferenceResult(
                error=f"Model config not found: {config_path}",
                source=InferenceSource.LOCAL
            )
        
        with open(config_path, "r") as f:
            model_config = json.load(f)
        
        self._stats["total_inferences"] += 1
        
        # Determine execution location
        use_cloud = self._should_use_cloud(mode, image_path)
        
        if use_cloud:
            result = self._run_cloud_inference(
                image_path, str(model_path), model_config, inference_params
            )
            
            # Handle fallback
            if not result.success and mode == InferenceMode.CLOUD_PREFERRED:
                logger.warning(f"Cloud inference failed: {result.error}. Falling back to local.")
                self._stats["fallbacks"] += 1
                result = self._run_local_inference(
                    image_path, str(model_path), model_config, inference_params,
                    enable_uncertainty, num_mc_passes
                )
                result.source = InferenceSource.CLOUD_FALLBACK
        else:
            result = self._run_local_inference(
                image_path, str(model_path), model_config, inference_params,
                enable_uncertainty, num_mc_passes
            )
        
        return result
    
    def _should_use_cloud(self, mode: InferenceMode, image_path: str) -> bool:
        """Determine if cloud inference should be used"""
        if mode == InferenceMode.LOCAL:
            return False
        
        if mode == InferenceMode.CLOUD:
            return True
        
        if mode in (InferenceMode.CLOUD_PREFERRED, InferenceMode.AUTO):
            if not self.cloud_available:
                logger.info("Cloud not available, using local inference")
                return False
            
            if mode == InferenceMode.AUTO:
                # For AUTO mode, consider image size
                # Use cloud for large images (>100MB) if available
                try:
                    file_size = os.path.getsize(image_path)
                    if file_size > 100 * 1024 * 1024:  # 100MB
                        return True
                except OSError:
                    pass
            
            return True
        
        return False
    
    def _run_cloud_inference(
        self,
        image_path: str,
        model_path: str,
        model_config: Dict[str, Any],
        inference_params: Dict[str, Any]
    ) -> InferenceResult:
        """Run inference on cloud"""
        if not self.cloud_provider:
            return InferenceResult(
                error="Cloud provider not available",
                source=InferenceSource.CLOUD
            )
        
        try:
            logger.info(f"Running cloud inference for {Path(image_path).name}")
            result = self.cloud_provider.run_inference(
                image_path=image_path,
                model_path=model_path,
                model_config=model_config,
                inference_params=inference_params
            )
            
            if result.success:
                self._stats["cloud_inferences"] += 1
            else:
                self._stats["cloud_failures"] += 1
            
            return result
            
        except Exception as e:
            self._stats["cloud_failures"] += 1
            logger.error(f"Cloud inference error: {e}")
            return InferenceResult(
                error=str(e),
                source=InferenceSource.CLOUD
            )
    
    def _run_local_inference(
        self,
        image_path: str,
        model_path: str,
        model_config: Dict[str, Any],
        inference_params: Dict[str, Any],
        enable_uncertainty: bool = False,
        num_mc_passes: int = 10
    ) -> InferenceResult:
        """Run inference locally"""
        timing = {"total": 0, "model_loading": 0, "preprocessing": 0, "inference": 0, "postprocessing": 0}
        start_total = time.time()
        
        try:
            from src.modeling.inference import (
                load_model_from_config,
                predict_volume,
                resample_to_original
            )
            from monai.transforms import (
                Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
                Orientationd, ScaleIntensityRanged, CropForegroundd,
                DivisiblePadd, ToTensord
            )
            
            device = self.get_device()
            logger.info(f"Running local inference on {device} for {Path(image_path).name}")
            
            # Load original image metadata
            original_img = nib.load(image_path)
            original_shape = original_img.shape
            original_spacing = original_img.header.get_zooms()[:3]
            original_affine = original_img.affine
            
            # Load model
            start_model = time.time()
            model, _ = load_model_from_config(model_path, device=device)
            timing["model_loading"] = time.time() - start_model
            
            # Build transforms
            start_preprocess = time.time()
            preproc = model_config.get("preprocessing", {})
            transforms = Compose([
                LoadImaged(keys=["image"], image_only=False),
                EnsureChannelFirstd(keys=["image"]),
                Spacingd(keys=["image"], pixdim=preproc.get("spacing", [1.5, 1.5, 1.5]), mode="bilinear"),
                Orientationd(keys=["image"], axcodes=preproc.get("orientation", "RAS"), labels=None),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=preproc.get("intensity_range", [-1000.0, 400.0])[0],
                    a_max=preproc.get("intensity_range", [-1000.0, 400.0])[1],
                    b_min=preproc.get("target_range", [0.0, 1.0])[0],
                    b_max=preproc.get("target_range", [0.0, 1.0])[1],
                    clip=True
                ),
                CropForegroundd(keys=["image"], source_key="image", margin=8, allow_smaller=True),
                DivisiblePadd(keys=["image"], k=preproc.get("divisible_k", 32)),
                ToTensord(keys=["image"])
            ])
            
            # Preprocess
            data_dict = {"image": image_path}
            transformed = transforms(data_dict)
            image_tensor = transformed["image"].unsqueeze(0)
            preprocessed_shape = image_tensor.shape[2:]
            timing["preprocessing"] = time.time() - start_preprocess
            
            # Run inference
            start_inference = time.time()
            roi_size = tuple(model_config.get("model", {}).get("img_size", [96, 96, 96]))
            overlap = inference_params.get("overlap", 0.5)
            sw_batch_size = inference_params.get("sw_batch_size", 2)
            
            uncertainty_map = None
            
            if enable_uncertainty:
                try:
                    from src.modeling.uncertainty import monte_carlo_dropout_inference, compute_uncertainty_metrics
                    mean_probs, uncertainty_tensor = monte_carlo_dropout_inference(
                        model=model,
                        image=image_tensor,
                        num_passes=num_mc_passes,
                        roi_size=roi_size,
                        sw_batch_size=sw_batch_size,
                        overlap=overlap,
                        device=device
                    )
                    pred_tensor = mean_probs
                    uncertainty_map = uncertainty_tensor.cpu().numpy()[0, 0]
                except ImportError:
                    logger.warning("Uncertainty module not available, running standard inference")
                    enable_uncertainty = False
            
            if not enable_uncertainty:
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    pred_tensor = predict_volume(
                        model=model,
                        image=image_tensor,
                        roi_size=roi_size,
                        sw_batch_size=sw_batch_size,
                        overlap=overlap,
                        device=device
                    )
            
            timing["inference"] = time.time() - start_inference
            
            # Post-process
            start_postprocess = time.time()
            pred_np = pred_tensor.cpu().numpy()[0]
            
            if pred_np.shape[0] > 1:
                pred_softmax = np.exp(pred_np) / np.sum(np.exp(pred_np), axis=0, keepdims=True)
                pred_seg = np.argmax(pred_softmax, axis=0).astype(np.uint8)
                confidence = np.max(pred_softmax, axis=0).astype(np.float32)
            else:
                pred_seg = (pred_np[0] > 0.5).astype(np.uint8)
                confidence = pred_np[0].astype(np.float32)
            
            # Resample to original space
            pred_resampled = resample_to_original(
                prediction=pred_seg,
                original_shape=original_shape,
                preprocessed_shape=tuple(preprocessed_shape),
                mode='nearest'
            )
            
            confidence_resampled = resample_to_original(
                prediction=confidence,
                original_shape=original_shape,
                preprocessed_shape=tuple(preprocessed_shape),
                mode='linear'
            )
            
            uncertainty_resampled = None
            if uncertainty_map is not None:
                uncertainty_resampled = resample_to_original(
                    prediction=uncertainty_map.astype(np.float32),
                    original_shape=original_shape,
                    preprocessed_shape=tuple(preprocessed_shape),
                    mode='linear'
                )
            
            timing["postprocessing"] = time.time() - start_postprocess
            timing["total"] = time.time() - start_total
            
            self._stats["local_inferences"] += 1
            
            # Clean up
            del model, pred_tensor, image_tensor
            torch.cuda.empty_cache()
            
            return InferenceResult(
                prediction=pred_resampled,
                confidence=confidence_resampled,
                uncertainty=uncertainty_resampled,
                source=InferenceSource.LOCAL,
                timing=timing,
                metadata={
                    "device": str(device),
                    "original_shape": list(original_shape),
                    "original_spacing": list(original_spacing),
                    "preprocessed_shape": list(preprocessed_shape),
                }
            )
            
        except Exception as e:
            timing["total"] = time.time() - start_total
            logger.error(f"Local inference error: {e}")
            return InferenceResult(
                error=str(e),
                source=InferenceSource.LOCAL,
                timing=timing
            )
    
    def get_status(self) -> Dict[str, Any]:
        """Get current service status"""
        return {
            "default_mode": self.default_mode.value,
            "cloud_available": self.cloud_available,
            "cloud_provider": self.cloud_provider.get_status() if self.cloud_provider else None,
            "local_device": str(self.get_device()),
            "cuda_available": torch.cuda.is_available(),
            "statistics": self._stats.copy(),
        }
    
    def upload_model_to_cloud(self, model_path: str) -> bool:
        """
        Upload a model to cloud storage for faster subsequent inference.
        
        Args:
            model_path: Path to model directory
            
        Returns:
            True if upload successful
        """
        if not self.cloud_provider:
            logger.warning("No cloud provider available")
            return False
        
        return self.cloud_provider.upload_model(model_path)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

# Global service instance for convenience
_default_service: Optional[InferenceService] = None


def get_inference_service(
    mode: InferenceMode = InferenceMode.CLOUD_PREFERRED,
    cloud_config: Optional[Dict[str, Any]] = None
) -> InferenceService:
    """
    Get the default inference service instance.
    
    Args:
        mode: Default inference mode
        cloud_config: Cloud provider configuration
        
    Returns:
        InferenceService instance
    """
    global _default_service
    
    if _default_service is None:
        _default_service = InferenceService(
            default_mode=mode,
            cloud_config=cloud_config
        )
    
    return _default_service


def run_inference(
    image_path: str,
    model_path: str,
    use_cloud: bool = True,
    **kwargs
) -> InferenceResult:
    """
    Convenience function for running inference.
    
    Args:
        image_path: Path to NIfTI image
        model_path: Path to model directory
        use_cloud: Whether to prefer cloud inference
        **kwargs: Additional inference parameters
        
    Returns:
        InferenceResult with prediction
    """
    service = get_inference_service()
    mode = InferenceMode.CLOUD_PREFERRED if use_cloud else InferenceMode.LOCAL
    return service.run_inference(
        image_path=image_path,
        model_path=model_path,
        mode=mode,
        inference_params=kwargs
    )


