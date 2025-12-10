"""
MONAI Auto3DSeg Integration
2025 Standard: Automated 3D segmentation pipeline
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import subprocess
import json

logger = logging.getLogger(__name__)


def run_auto3dseg_data_analyzer(
    dataroot: str,
    output_path: Optional[str] = None
) -> str:
    """
    Runs Auto3DSeg data analyzer to fingerprint the dataset.
    
    Args:
        dataroot: Path to BIDS dataset root
        output_path: Optional path for analysis output
        
    Returns:
        Path to analysis output directory
    """
    if output_path is None:
        output_path = os.path.join(dataroot, "auto3dseg_analysis")
    
    os.makedirs(output_path, exist_ok=True)
    
    try:
        from monai.apps.auto3dseg import DataAnalyzer
        
        analyzer = DataAnalyzer(
            datalist=None,  # Will auto-detect from BIDS
            dataroot=dataroot,
            output_path=output_path
        )
        
        logger.info("Running Auto3DSeg data analyzer...")
        datalist = analyzer.generate_datalist()
        
        logger.info(f"Data analysis complete. Output: {output_path}")
        return output_path
        
    except ImportError:
        logger.warning(
            "MONAI Auto3DSeg not available. "
            "Install with: pip install monai[auto3dseg]"
        )
        raise RuntimeError("Auto3DSeg module not available")


def run_auto3dseg_algo_generator(
    dataroot: str,
    analysis_output: str,
    num_fold: int = 5
) -> str:
    """
    Generates algorithm configurations based on data analysis.
    
    Args:
        dataroot: Path to BIDS dataset root
        analysis_output: Path from data analyzer output
        num_fold: Number of cross-validation folds
        
    Returns:
        Path to generated algorithm configurations
    """
    try:
        from monai.apps.auto3dseg import AlgoGen
        
        algo_gen = AlgoGen(
            data_stats_filename=os.path.join(analysis_output, "datalist.json"),
            output_path=os.path.join(analysis_output, "algo_configs")
        )
        
        logger.info("Generating algorithm configurations...")
        algo_gen.generate(num_fold=num_fold)
        
        config_path = os.path.join(analysis_output, "algo_configs")
        logger.info(f"Algorithm configurations generated: {config_path}")
        return config_path
        
    except ImportError:
        raise RuntimeError("Auto3DSeg module not available")


def run_auto3dseg_train(
    dataroot: str,
    algo_configs_path: str,
    output_path: Optional[str] = None,
    num_fold: int = 5
) -> str:
    """
    Trains all generated algorithms with cross-validation.
    
    Args:
        dataroot: Path to BIDS dataset root
        algo_configs_path: Path to algorithm configurations
        output_path: Optional path for training output
        num_fold: Number of cross-validation folds
        
    Returns:
        Path to trained models
    """
    if output_path is None:
        output_path = os.path.join(dataroot, "auto3dseg_models")
    
    os.makedirs(output_path, exist_ok=True)
    
    try:
        from monai.apps.auto3dseg import BundleGen
        
        bundle_gen = BundleGen(
            algo_path=algo_configs_path,
            data_stats_filename=os.path.join(
                os.path.dirname(algo_configs_path),
                "datalist.json"
            ),
            data_src_cfg_name=os.path.join(
                os.path.dirname(algo_configs_path),
                "data_src_cfg.yaml"
            )
        )
        
        logger.info("Starting Auto3DSeg training...")
        bundle_gen.set_num_fold(num_fold=num_fold)
        bundle_gen.train(output_path=output_path)
        
        logger.info(f"Training complete. Models saved: {output_path}")
        return output_path
        
    except ImportError:
        raise RuntimeError("Auto3DSeg module not available")


def run_auto3dseg_ensemble(
    models_path: str,
    output_path: Optional[str] = None
) -> str:
    """
    Creates ensemble configuration from best-performing models.
    
    Args:
        models_path: Path to trained models
        output_path: Optional path for ensemble output
        
    Returns:
        Path to ensemble configuration
    """
    if output_path is None:
        output_path = os.path.join(models_path, "ensemble")
    
    os.makedirs(output_path, exist_ok=True)
    
    try:
        from monai.apps.auto3dseg import EnsembleRunner
        
        ensemble_runner = EnsembleRunner(
            models_path=models_path
        )
        
        logger.info("Creating ensemble configuration...")
        ensemble_runner.ensemble(output_path=output_path)
        
        logger.info(f"Ensemble configuration created: {output_path}")
        return output_path
        
    except ImportError:
        raise RuntimeError("Auto3DSeg module not available")


def run_full_auto3dseg_pipeline(
    dataroot: str,
    output_base: Optional[str] = None,
    num_fold: int = 5
) -> Dict[str, str]:
    """
    Runs the complete Auto3DSeg pipeline end-to-end.
    
    Args:
        dataroot: Path to BIDS dataset root
        output_base: Base directory for all outputs
        num_fold: Number of cross-validation folds
        
    Returns:
        Dictionary with paths to all outputs
    """
    if output_base is None:
        output_base = os.path.join(dataroot, "auto3dseg_output")
    
    os.makedirs(output_base, exist_ok=True)
    
    logger.info("Starting full Auto3DSeg pipeline...")
    
    # Step 1: Data analysis
    analysis_output = run_auto3dseg_data_analyzer(
        dataroot=dataroot,
        output_path=os.path.join(output_base, "analysis")
    )
    
    # Step 2: Algorithm generation
    algo_configs = run_auto3dseg_algo_generator(
        dataroot=dataroot,
        analysis_output=analysis_output,
        num_fold=num_fold
    )
    
    # Step 3: Training
    models_path = run_auto3dseg_train(
        dataroot=dataroot,
        algo_configs_path=algo_configs,
        output_path=os.path.join(output_base, "models"),
        num_fold=num_fold
    )
    
    # Step 4: Ensemble
    ensemble_path = run_auto3dseg_ensemble(
        models_path=models_path,
        output_path=os.path.join(output_base, "ensemble")
    )
    
    results = {
        "analysis": analysis_output,
        "algo_configs": algo_configs,
        "models": models_path,
        "ensemble": ensemble_path
    }
    
    logger.info("Auto3DSeg pipeline complete!")
    return results


