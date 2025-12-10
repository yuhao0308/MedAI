"""
Inference History Management
Tracks inference runs, stores results, and enables comparison
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class InferenceHistory:
    """Manages inference history storage and retrieval"""
    
    def __init__(self, history_dir: str = "./data"):
        """
        Initialize inference history manager.
        
        Args:
            history_dir: Base directory for storing history
        """
        self.history_dir = Path(history_dir)
        self.history_file = self.history_dir / "inference_history.json"
        self._ensure_history_file()
    
    def _ensure_history_file(self):
        """Ensure history file exists"""
        self.history_dir.mkdir(parents=True, exist_ok=True)
        if not self.history_file.exists():
            self._save_history([])
    
    def _load_history(self) -> List[Dict[str, Any]]:
        """Load history from file"""
        try:
            with open(self.history_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []
    
    def _save_history(self, history: List[Dict[str, Any]]):
        """Save history to file"""
        with open(self.history_file, 'w') as f:
            json.dump(history, f, indent=2, default=str)
    
    def add_entry(
        self,
        image_path: str,
        model_name: str,
        prediction_path: str,
        nodule_count: int,
        suspicious_count: int,
        stats: Dict[str, Any],
        parameters: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        timing_info: Optional[Dict[str, float]] = None,
        inference_source: Optional[str] = None,
        nodule_candidates: Optional[List[Dict[str, Any]]] = None,
        uncertainty_metrics: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a new inference entry to history.
        
        Args:
            image_path: Path to input image
            model_name: Name of model used
            prediction_path: Path to saved prediction
            nodule_count: Number of nodules detected
            suspicious_count: Number of suspicious nodules
            stats: Nodule statistics
            parameters: Inference parameters used
            metadata: Additional metadata
            timing_info: Timing breakdown (total, inference, preprocessing, etc.)
            inference_source: Where inference ran ("local", "cloud", "cloud_fallback")
            nodule_candidates: List of individual nodule details
            uncertainty_metrics: Uncertainty analysis results (if enabled)
            
        Returns:
            Entry ID
        """
        history = self._load_history()
        
        entry_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(history)}"
        
        entry = {
            "id": entry_id,
            "timestamp": datetime.now().isoformat(),
            "image_path": str(image_path),
            "subject_id": Path(image_path).parent.parent.name if image_path else "",
            "model_name": model_name,
            "prediction_path": str(prediction_path) if prediction_path else None,
            "nodule_count": nodule_count,
            "suspicious_count": suspicious_count,
            "stats": {
                "total_volume_mm3": stats.get("total_volume_mm3", 0),
                "max_diameter_mm": stats.get("max_diameter_mm", 0),
                "mean_diameter_mm": stats.get("mean_diameter_mm", 0),
                "mean_confidence": stats.get("mean_confidence", 0),
                "category_counts": stats.get("category_counts", {})
            },
            "parameters": parameters,
            "metadata": metadata or {},
            # New fields for full result preservation
            "timing_info": timing_info or {},
            "inference_source": inference_source or "local",
            "nodule_candidates": nodule_candidates or [],
            "uncertainty_metrics": uncertainty_metrics
        }
        
        history.append(entry)
        self._save_history(history)
        
        logger.info(f"Added inference history entry: {entry_id}")
        return entry_id
    
    def get_history(
        self,
        limit: int = 50,
        subject_filter: Optional[str] = None,
        model_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get inference history with optional filtering.
        
        Args:
            limit: Maximum number of entries to return
            subject_filter: Filter by subject ID (partial match)
            model_filter: Filter by model name (partial match)
            
        Returns:
            List of history entries (newest first)
        """
        history = self._load_history()
        
        # Apply filters
        if subject_filter:
            history = [e for e in history if subject_filter.lower() in e.get("subject_id", "").lower()]
        
        if model_filter:
            history = [e for e in history if model_filter.lower() in e.get("model_name", "").lower()]
        
        # Sort by timestamp (newest first)
        history.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        return history[:limit]
    
    def get_entry(self, entry_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific history entry by ID"""
        history = self._load_history()
        for entry in history:
            if entry.get("id") == entry_id:
                return entry
        return None
    
    def get_subject_history(self, subject_id: str) -> List[Dict[str, Any]]:
        """Get all history entries for a specific subject"""
        return self.get_history(limit=100, subject_filter=subject_id)
    
    def compare_entries(
        self,
        entry_id_1: str,
        entry_id_2: str
    ) -> Dict[str, Any]:
        """
        Compare two inference entries.
        
        Args:
            entry_id_1: First entry ID
            entry_id_2: Second entry ID
            
        Returns:
            Comparison dictionary with differences
        """
        entry1 = self.get_entry(entry_id_1)
        entry2 = self.get_entry(entry_id_2)
        
        if not entry1 or not entry2:
            return {"error": "One or both entries not found"}
        
        comparison = {
            "entry_1": {
                "id": entry1["id"],
                "timestamp": entry1["timestamp"],
                "model": entry1["model_name"],
                "nodules": entry1["nodule_count"],
                "suspicious": entry1["suspicious_count"]
            },
            "entry_2": {
                "id": entry2["id"],
                "timestamp": entry2["timestamp"],
                "model": entry2["model_name"],
                "nodules": entry2["nodule_count"],
                "suspicious": entry2["suspicious_count"]
            },
            "differences": {
                "nodule_count_diff": entry2["nodule_count"] - entry1["nodule_count"],
                "suspicious_diff": entry2["suspicious_count"] - entry1["suspicious_count"],
                "max_diameter_diff": entry2["stats"].get("max_diameter_mm", 0) - entry1["stats"].get("max_diameter_mm", 0),
                "same_model": entry1["model_name"] == entry2["model_name"],
                "time_diff_hours": _calculate_time_diff(entry1["timestamp"], entry2["timestamp"])
            }
        }
        
        return comparison
    
    def delete_entry(self, entry_id: str) -> bool:
        """Delete a history entry"""
        history = self._load_history()
        original_len = len(history)
        history = [e for e in history if e.get("id") != entry_id]
        
        if len(history) < original_len:
            self._save_history(history)
            logger.info(f"Deleted inference history entry: {entry_id}")
            return True
        return False
    
    def clear_history(self, confirm: bool = False) -> bool:
        """Clear all history (requires confirmation)"""
        if confirm:
            self._save_history([])
            logger.info("Cleared all inference history")
            return True
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get overall statistics from history"""
        history = self._load_history()
        
        if not history:
            return {
                "total_runs": 0,
                "unique_subjects": 0,
                "unique_models": 0,
                "total_nodules_detected": 0,
                "total_suspicious": 0
            }
        
        return {
            "total_runs": len(history),
            "unique_subjects": len(set(e.get("subject_id", "") for e in history)),
            "unique_models": len(set(e.get("model_name", "") for e in history)),
            "total_nodules_detected": sum(e.get("nodule_count", 0) for e in history),
            "total_suspicious": sum(e.get("suspicious_count", 0) for e in history),
            "first_run": min(e.get("timestamp", "") for e in history),
            "last_run": max(e.get("timestamp", "") for e in history)
        }


def _calculate_time_diff(timestamp1: str, timestamp2: str) -> float:
    """Calculate time difference in hours between two ISO timestamps"""
    try:
        dt1 = datetime.fromisoformat(timestamp1)
        dt2 = datetime.fromisoformat(timestamp2)
        diff = abs((dt2 - dt1).total_seconds())
        return round(diff / 3600, 2)  # Convert to hours
    except (ValueError, TypeError):
        return 0.0


# Singleton instance for easy access
_history_instance = None


def get_history_manager(history_dir: str = "./data") -> InferenceHistory:
    """Get or create the inference history manager singleton"""
    global _history_instance
    if _history_instance is None:
        _history_instance = InferenceHistory(history_dir)
    return _history_instance

