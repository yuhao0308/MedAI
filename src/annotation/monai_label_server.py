"""
MONAI Label Server Configuration
2025 Standard: Integration with MONAI Label for active learning annotation
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


def create_monai_label_config(
    bids_root: str,
    output_dir: str,
    model_name: str = "VISTA-3D"
) -> Dict[str, Any]:
    """
    Creates configuration for MONAI Label server.
    
    Args:
        bids_root: Path to BIDS dataset root
        output_dir: Directory for MONAI Label output
        model_name: Name of AI model to use (e.g., "VISTA-3D")
        
    Returns:
        Configuration dictionary
    """
    config = {
        "name": "MONAI Label Server",
        "version": "0.5.0",
        "data": {
            "bids_root": bids_root,
            "derivatives_path": os.path.join(bids_root, "derivatives", "labels")
        },
        "models": {
            "primary": {
                "name": model_name,
                "type": "foundation_model" if model_name == "VISTA-3D" else "segmentation"
            }
        },
        "output": {
            "directory": output_dir,
            "format": "BIDS"
        },
        "active_learning": {
            "enabled": True,
            "retrain_after_n_labels": 5
        }
    }
    
    return config


def setup_monai_label_server(
    bids_root: str,
    port: int = 8000,
    host: str = "0.0.0.0"
) -> str:
    """
    Sets up and returns instructions for running MONAI Label server.
    
    Args:
        bids_root: Path to BIDS dataset root
        port: Server port
        host: Server host
        
    Returns:
        Command string to run MONAI Label server
    """
    # MONAI Label is typically run via command line
    # This function provides the command and setup instructions
    
    command = f"""
# Install MONAI Label (if not already installed)
pip install monai-label

# Run MONAI Label server
monailabel start_server \\
    --app radiology \\
    --studies {bids_root} \\
    --host {host} \\
    --port {port}

# Or use Docker:
# docker run --gpus all --rm -it \\
#     -p {port}:8000 \\
#     -v {bids_root}:/workspace/data \\
#     projectmonai/monailabel:latest \\
#     monailabel start_server --app radiology --studies /workspace/data
"""
    
    logger.info("MONAI Label server setup instructions generated")
    return command


def export_annotations_to_bids(
    monai_label_output: str,
    bids_derivatives: str
) -> None:
    """
    Exports annotations from MONAI Label to BIDS derivatives format.
    
    Args:
        monai_label_output: MONAI Label output directory
        bids_derivatives: BIDS derivatives/labels directory
    """
    import shutil
    from pathlib import Path
    
    ml_output = Path(monai_label_output)
    bids_deriv = Path(bids_derivatives)
    
    # Find all segmentation files
    seg_files = list(ml_output.rglob("*.nii.gz")) + list(ml_output.rglob("*.nii"))
    
    for seg_file in seg_files:
        # Extract subject ID from filename or path
        # This is a simplified version - actual implementation would parse MONAI Label output format
        filename = seg_file.name
        
        # Try to match BIDS naming: sub-XXX_*_seg-*.nii.gz
        if "sub-" in filename:
            parts = filename.split("_")
            sub_id = None
            for part in parts:
                if part.startswith("sub-"):
                    sub_id = part
                    break
            
            if sub_id:
                # Organize into BIDS structure
                target_dir = bids_deriv / sub_id / "anat"
                target_dir.mkdir(parents=True, exist_ok=True)
                
                # Copy with BIDS naming
                target_file = target_dir / filename
                shutil.copy2(seg_file, target_file)
                logger.info(f"Exported annotation: {target_file}")


def get_3d_slicer_connection_instructions() -> str:
    """
    Returns instructions for connecting 3D Slicer to MONAI Label.
    
    Returns:
        Instructions string
    """
    instructions = """
Connecting 3D Slicer to MONAI Label:

1. Install 3D Slicer:
   - Download from: https://www.slicer.org/
   - Install MONAI Label extension in 3D Slicer

2. Connect to MONAI Label Server:
   - Open 3D Slicer
   - Go to: Modules > MONAI Label
   - Enter server URL: http://localhost:8000
   - Click "Connect"

3. Load BIDS Dataset:
   - Select "Load Study" from MONAI Label module
   - Browse to BIDS dataset directory
   - Select subject and series

4. Start Annotation:
   - Use VISTA-3D tool for AI-assisted segmentation
   - Place point prompts (include/exclude)
   - Review and correct segmentation
   - Click "Submit" to save annotation

5. Active Learning:
   - MONAI Label will automatically retrain models
   - New annotations improve model performance
   - Iterate until satisfactory accuracy

For more information:
- MONAI Label docs: https://docs.monai.io/projects/label/
- 3D Slicer docs: https://www.slicer.org/wiki/Documentation
"""
    
    return instructions


