# MONAI Medical Imaging AI Pipeline

A 2025-standard, MONAI-centric medical imaging AI pipeline for diagnostic prediction, supporting DICOM ingestion, BIDS organization, 3D segmentation, and multimodal reasoning.

## Overview

This pipeline implements the 2025 standard architecture for medical imaging AI:

- **DICOM Ingestion**: Parse and convert DICOM files to NIfTI format
- **BIDS Organization**: Organize data according to Brain Imaging Data Structure standard
- **MONAI Label Integration**: Active learning annotation workflow with VISTA-3D
- **Preprocessing**: Standardized MONAI transform pipelines
- **Modeling**: Auto3DSeg automation and custom training pipelines
- **Uncertainty Quantification**: Monte Carlo Dropout for predictive uncertainty
- **Multimodal AI**: Foundation for Radiology Agent Framework

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- dcm2niix (for DICOM conversion)

### Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Install System Dependencies

**macOS:**
```bash
brew install dcm2niix
```

**Linux:**
```bash
sudo apt-get install dcm2niix
```

**Windows:**
Download from: https://github.com/rordenlab/dcm2niix/releases

## Quick Start

### 1. Prepare Dataset

If you have DICOM files, ingest them:

```bash
# Generic ingestion
python scripts/ingest_data.py --input-dir ./data/raw_dicom --output-dir ./data/bids_dataset

# Or use Streamlit frontend
python3 -m streamlit run app.py
# Then go to "Upload & Process" tab
```

### 2. Validate Dataset

```bash
python scripts/validate_bids.py data/bids_dataset
```

### 3. Train Model

```bash
# Basic training
python scripts/train_model.py --config configs/pipeline_config.yaml

# With custom epochs
python scripts/train_model.py --config configs/pipeline_config.yaml --epochs 50
```

### 4. Run Inference

```bash
# Basic inference
python scripts/run_inference.py --checkpoint models/best_model.pth --config configs/pipeline_config.yaml

# With evaluation
python scripts/run_inference.py --checkpoint models/best_model.pth --config configs/pipeline_config.yaml --evaluate
```

### 5. Evaluate Model

```bash
python scripts/evaluate_model.py --checkpoint models/best_model.pth --config configs/pipeline_config.yaml
```

### 6. Run Full Pipeline

```bash
# Complete pipeline (validate + train + infer)
python scripts/run_pipeline.py full --config configs/pipeline_config.yaml
```

## Detailed Usage

### Data Ingestion

See `QUICKSTART.md` for detailed ingestion instructions.

### Training

See `docs/TRAINING.md` for comprehensive training guide including:
- Configuration options
- Hyperparameter tuning
- Troubleshooting
- Best practices

### Inference

Run inference on new data:

```bash
python scripts/run_inference.py \
    --checkpoint models/best_model.pth \
    --config configs/pipeline_config.yaml \
    --subjects sub-001 sub-002 \
    --evaluate
```

### Evaluation

Evaluate model performance with detailed metrics:

```bash
python scripts/evaluate_model.py \
    --checkpoint models/best_model.pth \
    --config configs/pipeline_config.yaml \
    --include-hausdorff \
    --format json
```

## Advanced Usage

### Custom Training (Programmatic)

```python
from src.preprocessing.dataloader import read_bids_dataset, create_dataset, create_dataloader, split_dataset
from src.preprocessing.transforms import get_train_transforms, get_val_transforms
from src.modeling.trainer import create_model, train_model

# Load data
data_dicts = read_bids_dataset("./data/bids_dataset")
splits = split_dataset(data_dicts)

# Create datasets
train_dataset = create_dataset(
    splits['train'],
    transform=get_train_transforms(),
    cache_type="memory"
)
val_dataset = create_dataset(
    splits['val'],
    transform=get_val_transforms(),
    cache_type="memory"
)

# Create model and train
model = create_model(model_type="SwinUNETR")
history = train_model(model, train_loader, val_loader, num_epochs=100)
```

### Inference with Uncertainty

```python
from src.modeling.uncertainty import monte_carlo_dropout_inference

mean_pred, uncertainty = monte_carlo_dropout_inference(
    model=model,
    image=image_tensor,
    num_passes=20
)
```

### Auto3DSeg (Automated Training)

```python
from src.modeling.auto3dseg_pipeline import run_full_auto3dseg_pipeline

results = run_full_auto3dseg_pipeline(
    dataroot="./data/bids_dataset",
    num_fold=5
)
```

## Project Structure

```
MedAI/
├── data/
│   ├── raw_dicom/          # Downloaded DICOM files
│   ├── bids_dataset/       # BIDS-organized NIfTI files
│   └── derivatives/        # Labels and model outputs
├── src/
│   ├── ingestion/          # DICOM parsing and conversion
│   ├── preprocessing/      # MONAI transform pipelines
│   ├── modeling/           # Training and inference
│   ├── annotation/         # MONAI Label integration
│   ├── multimodal/        # Multimodal AI components
│   └── visualization/      # Visualization utilities
├── configs/                # Configuration files
├── scripts/                # Utility scripts
└── notebooks/              # Exploration notebooks
```

## Configuration

Edit `configs/pipeline_config.yaml` to customize:
- Preprocessing parameters (spacing, intensity ranges)
- Training hyperparameters (epochs, batch size, learning rate)
- Model architecture (SwinUNETR or SegResNet)
- Inference settings (sliding window parameters)
- Data augmentation settings

See `docs/TRAINING.md` for detailed configuration guide.

## Public Datasets

### Recommended Datasets

1. **TCIA (The Cancer Imaging Archive)**
   - LIDC-IDRI: Lung nodule detection
   - BRATS: Brain tumor segmentation
   - CT-ORG: Organ segmentation

2. **IDC (Imaging Data Commons)**
   - tcga-luad: Lung adenocarcinoma
   - tcga-gbm: Glioblastoma

3. **MIDRC Data Commons**
   - COVID-19 imaging datasets

### Download Instructions

See `scripts/download_datasets.py` for programmatic downloads, or use web interfaces:
- TCIA: https://www.cancerimagingarchive.net/
- IDC: https://portal.imaging.datacommons.cancer.gov/
- MIDRC: https://data.midrc.org/

## MONAI Label Setup

1. **Install MONAI Label:**
   ```bash
   pip install monai-label
   ```

2. **Start Server:**
   ```bash
   monailabel start_server --app radiology --studies ./data/bids_dataset
   ```

3. **Connect 3D Slicer:**
   - Install 3D Slicer: https://www.slicer.org/
   - Install MONAI Label extension
   - Connect to server at http://localhost:8000
   - Use VISTA-3D for AI-assisted annotation

## Multimodal Diagnostic Prediction

The pipeline includes foundation components for multimodal AI:

```python
from src.multimodal.agent_framework import create_radiology_agent
from src.multimodal.data_loader import pair_image_text_data

# Create agent
agent = create_radiology_agent()

# Pair imaging with EHR/text data
paired_data = pair_image_text_data(
    image_paths=image_list,
    ehr_data=ehr_df,
    clinical_notes=notes_dict
)

# Process multimodal prompt
response = agent.process_multimodal_prompt(
    prompt="Review this CT scan and check for new lung nodules",
    image_path=image_path,
    ehr_data=patient_ehr,
    clinical_note=clinical_note
)
```

## Citation

This pipeline implements the 2025 standard architecture for medical imaging AI using the MONAI ecosystem. Key references:

- MONAI: https://monai.io/
- BIDS: https://bids.neuroimaging.io/
- MONAI Label: https://docs.monai.io/projects/label/
- Auto3DSeg: https://docs.monai.io/projects/auto3dseg/

## License

This project is provided as-is for research and educational purposes. Please ensure compliance with dataset licenses and HIPAA/GDPR regulations when working with medical data.

## Contributing

This is a reference implementation. For production use, ensure:
- HIPAA-compliant de-identification
- Proper data governance
- Clinical validation
- Regulatory compliance

## Support

For issues and questions:
- MONAI Documentation: https://docs.monai.io/
- MONAI Forum: https://github.com/Project-MONAI/MONAI/discussions

