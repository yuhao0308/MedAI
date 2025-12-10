"""
Base classes for Cloud Inference Providers
Abstract interface for cloud-based model inference
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import numpy as np
from enum import Enum
import time


class InferenceSource(Enum):
    """Enum indicating where inference was performed"""
    LOCAL = "local"
    CLOUD = "cloud"
    CLOUD_FALLBACK = "cloud_fallback"  # Cloud requested but fell back to local


@dataclass
class InferenceResult:
    """
    Standardized result from inference (cloud or local).
    
    Attributes:
        prediction: Segmentation mask array (resampled to original space)
        confidence: Confidence map array (optional)
        uncertainty: Uncertainty map array (optional)
        source: Where inference was performed (local/cloud)
        timing: Dictionary of timing information
        metadata: Additional metadata from inference
        error: Error message if inference failed
    """
    prediction: Optional[np.ndarray] = None
    confidence: Optional[np.ndarray] = None
    uncertainty: Optional[np.ndarray] = None
    source: InferenceSource = InferenceSource.LOCAL
    timing: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    @property
    def success(self) -> bool:
        """Check if inference was successful"""
        return self.prediction is not None and self.error is None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for serialization, excluding numpy arrays)"""
        return {
            "source": self.source.value,
            "timing": self.timing,
            "metadata": self.metadata,
            "error": self.error,
            "prediction_shape": list(self.prediction.shape) if self.prediction is not None else None,
            "has_confidence": self.confidence is not None,
            "has_uncertainty": self.uncertainty is not None,
        }


class CloudInferenceProvider(ABC):
    """
    Abstract base class for cloud inference providers.
    
    Implementations should handle:
    - Model loading/caching on the cloud
    - Preprocessing on cloud (to minimize data transfer)
    - GPU-accelerated inference
    - Proper error handling and timeouts
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the cloud provider.
        
        Args:
            config: Provider-specific configuration
        """
        self.config = config
        self._is_available: Optional[bool] = None
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g., 'modal', 'replicate')"""
        pass
    
    @abstractmethod
    def check_availability(self) -> bool:
        """
        Check if the cloud provider is available and configured.
        
        Returns:
            True if provider is ready for inference
        """
        pass
    
    @property
    def is_available(self) -> bool:
        """Cached availability check"""
        if self._is_available is None:
            self._is_available = self.check_availability()
        return self._is_available
    
    @abstractmethod
    def run_inference(
        self,
        image_path: str,
        model_path: str,
        model_config: Dict[str, Any],
        inference_params: Optional[Dict[str, Any]] = None
    ) -> InferenceResult:
        """
        Run inference on a single image using the cloud provider.
        
        Args:
            image_path: Path to NIfTI image file
            model_path: Path to model directory (with config.json and best_model.pth)
            model_config: Model configuration dictionary
            inference_params: Additional parameters (overlap, sw_batch_size, etc.)
            
        Returns:
            InferenceResult with prediction and metadata
        """
        pass
    
    @abstractmethod
    def upload_model(self, model_path: str) -> bool:
        """
        Upload/sync model to cloud storage for faster subsequent inference.
        
        Args:
            model_path: Local path to model directory
            
        Returns:
            True if upload successful
        """
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of the cloud provider.
        
        Returns:
            Dictionary with status information
        """
        return {
            "provider": self.name,
            "available": self.is_available,
            "config": {k: v for k, v in self.config.items() if k not in ["token", "secret", "api_key"]}
        }


class CloudInferenceError(Exception):
    """Exception raised for cloud inference errors"""
    
    def __init__(self, message: str, provider: str, recoverable: bool = True):
        self.message = message
        self.provider = provider
        self.recoverable = recoverable
        super().__init__(f"[{provider}] {message}")


