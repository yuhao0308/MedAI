#!/usr/bin/env python3
"""
Upload trained model to Modal volume for cloud inference.
Run this locally after training to make model available on Railway.

Usage:
    python scripts/upload_model_to_modal.py

Or with a specific model path:
    python scripts/upload_model_to_modal.py --model-path models/my_model/v1
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def upload_model(model_path: str):
    """Upload model to Modal volume"""
    try:
        from src.cloud.modal_inference import (
            ModalInferenceProvider,
            is_modal_available,
            MODAL_AVAILABLE
        )
    except ImportError as e:
        print(f"❌ Could not import Modal modules: {e}")
        print("   Make sure modal is installed: pip install modal")
        return False
    
    if not MODAL_AVAILABLE:
        print("❌ Modal SDK not installed. Run: pip install modal")
        return False
    
    if not is_modal_available():
        print("❌ Modal not configured. Run: modal token new")
        return False
    
    model_path = Path(model_path)
    
    # Validate model files exist
    checkpoint_path = model_path / "best_model.pth"
    config_path = model_path / "config.json"
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return False
    
    if not config_path.exists():
        print(f"❌ Config not found: {config_path}")
        return False
    
    checkpoint_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
    print(f"📦 Model: {model_path.name}")
    print(f"   Checkpoint: {checkpoint_size_mb:.1f} MB")
    print(f"   Config: {config_path.name}")
    
    print("\n⏳ Uploading to Modal volume...")
    print("   (This may take a few minutes for large models)")
    
    try:
        provider = ModalInferenceProvider()
        success = provider.upload_model(str(model_path))
        
        if success:
            print(f"\n✅ Model '{model_path.name}' uploaded successfully!")
            print("\n📌 Your model is now available for cloud inference on Railway.")
            print("   Just enable '☁️ Use Cloud GPU' when running inference.")
            return True
        else:
            print("\n❌ Upload failed. Check Modal dashboard for details.")
            return False
            
    except Exception as e:
        print(f"\n❌ Upload error: {e}")
        return False


def find_available_models():
    """Find models in the models directory"""
    models_dir = Path("./models")
    available = []
    
    if not models_dir.exists():
        return available
    
    for model_dir in models_dir.iterdir():
        if model_dir.is_dir():
            # Check direct structure
            if (model_dir / "config.json").exists() and (model_dir / "best_model.pth").exists():
                available.append(model_dir)
            # Check versioned structure
            for version_dir in model_dir.iterdir():
                if version_dir.is_dir():
                    if (version_dir / "config.json").exists() and (version_dir / "best_model.pth").exists():
                        available.append(version_dir)
    
    return available


def main():
    parser = argparse.ArgumentParser(description="Upload model to Modal for cloud inference")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model directory (containing config.json and best_model.pth)"
    )
    args = parser.parse_args()
    
    print("=" * 50)
    print("Modal Model Uploader")
    print("=" * 50)
    
    if args.model_path:
        model_path = args.model_path
    else:
        # Auto-detect models
        available = find_available_models()
        
        if not available:
            print("❌ No models found in ./models directory")
            print("   Models must have config.json and best_model.pth")
            return
        
        print(f"\n📂 Found {len(available)} model(s):\n")
        for i, model in enumerate(available):
            size_mb = (model / "best_model.pth").stat().st_size / (1024 * 1024)
            print(f"   [{i+1}] {model.relative_to('.')} ({size_mb:.1f} MB)")
        
        if len(available) == 1:
            model_path = str(available[0])
            print(f"\n→ Using: {model_path}")
        else:
            print("\nSpecify which model to upload:")
            print("   python scripts/upload_model_to_modal.py --model-path <path>")
            return
    
    print()
    success = upload_model(model_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
