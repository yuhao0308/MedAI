"""
Cloud Inference Module for Medical Imaging AI
Supports Modal Labs serverless GPU inference with local fallback
"""

from src.cloud.base import CloudInferenceProvider, InferenceResult
from src.cloud.modal_inference import ModalInferenceProvider, is_modal_available

__all__ = [
    "CloudInferenceProvider",
    "InferenceResult",
    "ModalInferenceProvider",
    "is_modal_available",
]


