# Training Environment Guide

## Model Being Trained

### SwinUNETR (Swin Transformer U-Net for 3D Medical Image Segmentation)

**Model Details:**
- **Architecture**: SwinUNETR (Swin Transformer-based U-Net)
- **Parameters**: ~15.7 million trainable parameters
- **Input Size**: 96×96×96 voxels per patch
- **Output**: Binary segmentation (background + foreground)
- **Task**: 3D medical image segmentation (organ/pathology segmentation)

**Why SwinUNETR?**
- State-of-the-art transformer-based architecture for 3D medical imaging
- Excellent performance on medical segmentation tasks
- Part of MONAI's recommended models for 3D segmentation
- Handles large 3D volumes efficiently with patch-based training

---

## Computational Requirements

### Memory Requirements
- **Model Size**: ~60-120 MB (model weights)
- **Training Memory**: **8-16 GB GPU memory** recommended
- **Batch Size**: Currently 1 (can increase with more GPU memory)
- **Patch Size**: 96×96×96 voxels

### Training Speed
- **On GPU (NVIDIA)**: ~1-5 minutes per batch (depending on GPU)
- **On CPU**: **~20-40 minutes per batch** (as you experienced)
- **Full Epoch (35 batches)**: 
  - GPU: ~1-3 hours
  - CPU: **~12-24 hours** (not practical)

### Why Your Mac Mini Struggles
1. **No GPU**: Training on CPU is 10-50x slower than GPU
2. **Memory**: Process was killed - likely ran out of RAM
3. **3D Volumes**: Processing 96×96×96×3D patches is computationally intensive
4. **Transformer Architecture**: SwinUNETR uses attention mechanisms that are memory-intensive

---

## Recommended Training Environments

### Option 1: Cloud GPU Services (Recommended)

#### **Google Colab Pro** (Easiest)
- **Cost**: ~$10/month
- **GPU**: NVIDIA T4 or V100 (16GB)
- **Pros**: Easy setup, free tier available, Jupyter interface
- **Cons**: Session timeouts, limited storage
- **Setup**: Upload code, install dependencies, train

#### **AWS EC2** (Most Flexible)
- **Instance**: `g4dn.xlarge` or `p3.2xlarge`
- **Cost**: ~$0.50-3.00/hour (pay-as-you-go)
- **GPU**: NVIDIA T4 or V100 (16-32GB)
- **Pros**: Full control, persistent storage, scalable
- **Cons**: Requires AWS setup, more complex

#### **Google Cloud Platform (GCP)**
- **Instance**: `n1-standard-4` with NVIDIA T4
- **Cost**: ~$0.35-0.70/hour
- **GPU**: NVIDIA T4 (16GB)
- **Pros**: Good integration with Google services
- **Cons**: Requires GCP account setup

#### **Azure ML**
- **Compute**: GPU-enabled compute instances
- **Cost**: ~$0.50-2.00/hour
- **GPU**: Various options (T4, V100, A100)
- **Pros**: Good ML tooling, managed notebooks
- **Cons**: Azure-specific setup

### Option 2: Local GPU Workstation

#### **Requirements**
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3060, RTX 3070, or better)
- **RAM**: 16GB+ system RAM
- **Storage**: SSD recommended for faster data loading
- **CUDA**: CUDA 11.0+ and cuDNN installed

#### **Setup**
```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU
python -c "import torch; print(torch.cuda.is_available())"
```

### Option 3: University/Research Computing

Many universities provide:
- **HPC Clusters**: Shared GPU resources
- **Research Computing**: Dedicated GPU nodes
- **SLURM**: Job scheduling system for training

---

## Training Configuration for Different Environments

### For Cloud GPU (Recommended Settings)
```yaml
training:
  spatial_size: [96, 96, 96]  # Keep current
  batch_size: 2-4  # Can increase with more GPU memory
  num_epochs: 100
  learning_rate: 0.0001
  num_workers: 4  # Can use multiprocessing on Linux
  cache_type: "memory"
```

### For Mac Mini (Testing Only - Not Recommended)
```yaml
training:
  spatial_size: [64, 64, 64]  # Smaller patches
  batch_size: 1
  num_epochs: 1-2  # Just for testing
  learning_rate: 0.0001
  num_workers: 0  # Required on macOS
  cache_type: "none"  # Reduce memory usage
```

---

## Alternative: Use Smaller Model for Local Testing

### Option: SegResNet (Lighter Alternative)
```yaml
model:
  type: "SegResNet"  # Instead of SwinUNETR
  in_channels: 1
  out_channels: 2
  img_size: [96, 96, 96]
```

**SegResNet Benefits:**
- Fewer parameters (~5-10M vs 15.7M)
- Lower memory footprint
- Faster training
- Still good performance for segmentation

**Trade-off**: Slightly lower accuracy than SwinUNETR, but much more feasible on CPU.

---

## Recommended Workflow

### For Development/Testing on Mac Mini
1. **Use SegResNet** with smaller patches (64×64×64)
2. **Train for 1-2 epochs** just to verify pipeline works
3. **Use minimal dataset** (5-10 subjects) for quick tests
4. **Focus on code development**, not full training

### For Actual Model Training
1. **Use Cloud GPU** (Colab Pro, AWS, GCP)
2. **Use SwinUNETR** with full configuration
3. **Train for 100+ epochs** for best results
4. **Use full dataset** (all 50 subjects)

---

## Cost Estimates

### Training One Model (100 epochs, 35 batches/epoch)
- **Google Colab Pro**: ~$10/month (unlimited training time)
- **AWS EC2 g4dn.xlarge**: ~$15-30 (10-20 hours of training)
- **GCP n1-standard-4 + T4**: ~$10-20 (10-20 hours)

### Monthly Development (Multiple experiments)
- **Colab Pro**: $10/month (best value for development)
- **AWS/GCP**: $50-200/month (depending on usage)

---

## Quick Start: Google Colab

### Step 1: Upload Code
```python
# In Colab
!git clone <your-repo>  # Or upload files
!cd MedAI && pip install -r requirements.txt
```

### Step 2: Upload Data
```python
# Upload BIDS dataset to Colab (or use Google Drive)
from google.colab import drive
drive.mount('/content/drive')
```

### Step 3: Train
```python
!cd MedAI && python scripts/train_model.py \
    --config configs/pipeline_config.yaml \
    --epochs 100
```

### Step 4: Download Model
```python
# Download trained model
from google.colab import files
files.download('models/best_model.pth')
```

---

## Mac Mini Limitations Summary

| Aspect | Mac Mini | Required |
|--------|---------|----------|
| GPU | None (CPU only) | NVIDIA GPU (8GB+) |
| Training Speed | ~30 min/batch | ~1-5 min/batch |
| Memory | Limited (killed process) | 16GB+ RAM |
| Practical Use | Code development only | Full training |

**Recommendation**: Use Mac Mini for:
- ✅ Code development
- ✅ Testing pipeline fixes
- ✅ Running inference (once model is trained)
- ✅ Data preprocessing
- ❌ NOT for actual model training

---

## Next Steps

1. **For Now**: Continue using Mac Mini for testing/development
2. **For Training**: Set up cloud GPU account (Colab Pro recommended)
3. **Alternative**: Use SegResNet with smaller patches for local testing
4. **Long-term**: Consider dedicated GPU workstation if training frequently

---

## References

- **SwinUNETR Paper**: [MONAI SwinUNETR Documentation](https://docs.monai.io/en/stable/networks.html#swinunetr)
- **Cloud GPU Comparison**: [Cloud GPU Pricing Comparison](https://www.paperspace.com/gpu-cloud-compute)
- **Colab Pro**: https://colab.research.google.com/signup

