# Medical Imaging AI Pipeline - System Design Overview

**Version**: 1.0  
**Date**: 2025-11-13  
**Status**: Phase 0 - Foundation Complete (~60%)

---

## 1. Project Purpose

### Primary Goal
Develop a **2025-standard medical imaging AI pipeline** for **3D medical image segmentation** and **diagnostic prediction** using the MONAI ecosystem. The system processes DICOM medical images, trains deep learning models, and provides AI-assisted analysis for clinical research.

### Key Objectives
- **Automate Medical Image Analysis**: Segment anatomical structures and pathologies in 3D medical images (CT, MRI)
- **Standardize Data Processing**: Convert DICOM to BIDS format for reproducible research
- **Enable AI-Assisted Annotation**: Integrate with MONAI Label for active learning workflows
- **Support Clinical Research**: Provide tools for training, inference, and evaluation of segmentation models
- **Foundation for Production**: Build towards HIPAA-compliant, production-ready clinical deployment

---

## 2. System Architecture

### 2.1 High-Level Workflow

```
DICOM Files → Ingestion → BIDS Organization → Preprocessing → Training → Inference → Evaluation
     ↓            ↓              ↓                ↓            ↓          ↓           ↓
  Raw Data    Conversion    Standardized    Normalized    Model      Predictions  Metrics
                           Format          Images        Checkpoint
```

### 2.2 Core Components

#### **Data Ingestion Layer**
- **DICOM Parser**: Extracts metadata and loads DICOM series
- **DICOM-to-NIfTI Converter**: Converts to standard neuroimaging format
- **BIDS Organizer**: Structures data according to Brain Imaging Data Structure standard
- **Validation**: Ensures data quality and BIDS compliance

#### **Preprocessing Layer**
- **MONAI Transform Pipeline**: Standardized 3D image preprocessing
  - Spatial normalization (resampling, orientation)
  - Intensity normalization
  - Data augmentation (rotation, elastic deformation)
  - Label normalization (binary/multi-class)
- **DataLoader**: Efficient data loading with caching support

#### **Modeling Layer**
- **Training Pipeline**: Custom training with SwinUNETR/SegResNet
- **Inference Pipeline**: Sliding window inference for full 3D volumes
- **Evaluation**: Comprehensive metrics (Dice, IoU, Hausdorff distance)
- **Uncertainty Quantification**: Monte Carlo Dropout for prediction confidence
- **Auto3DSeg Integration**: Automated algorithm selection and training

#### **Annotation Layer**
- **MONAI Label Integration**: AI-assisted annotation with VISTA-3D foundation model
- **3D Slicer Connection**: Interactive annotation interface

#### **Visualization Layer**
- **Streamlit Frontend**: Web-based UI for data exploration and inference
- **Visualization Utilities**: 3D rendering and reporting tools

#### **Multimodal AI Layer** (Foundation)
- **Radiology Agent Framework**: Structure for vision-language models
- **Data Loader**: EHR and clinical notes integration
- **Tool Execution**: MONAI Bundle tool calling (placeholder)

---

## 3. System Capabilities

### 3.1 Data Processing
- ✅ **DICOM Ingestion**: Parse and convert DICOM files to NIfTI format
- ✅ **BIDS Organization**: Organize datasets according to BIDS standard
- ✅ **Data Validation**: Comprehensive quality checks and consistency validation
- ✅ **Multi-Organ Support**: Handle multiple organ masks per subject

### 3.2 Model Training
- ✅ **Custom Training**: Train SwinUNETR or SegResNet models
- ✅ **Automated Training**: Auto3DSeg for algorithm selection
- ✅ **Checkpointing**: Save/resume training from checkpoints
- ✅ **Training Monitoring**: Track loss and metrics over epochs

### 3.3 Inference & Evaluation
- ✅ **Sliding Window Inference**: Process full 3D volumes efficiently
- ✅ **Batch Processing**: Process multiple subjects
- ✅ **Uncertainty Maps**: Generate prediction confidence maps
- ✅ **Comprehensive Metrics**: Dice, IoU, Sensitivity, Specificity, Hausdorff distance

### 3.4 Annotation & Active Learning
- ✅ **MONAI Label Integration**: AI-assisted annotation workflow
- ✅ **3D Slicer Connection**: Interactive annotation interface
- ⚠️ **Active Learning**: Foundation only (not fully automated)

### 3.5 Visualization & Reporting
- ✅ **Streamlit Frontend**: Web-based data exploration and inference
- ✅ **3D Visualization**: Interactive volume rendering
- ✅ **Quantitative Reports**: JSON-based evaluation reports
- ⚠️ **DICOM-SR**: Not yet implemented (planned)

### 3.6 Multimodal AI (Foundation)
- ✅ **Framework Structure**: Radiology Agent architecture
- ✅ **Data Integration**: EHR and clinical notes loading
- ⚠️ **VLM Integration**: Placeholder only (VILA-M3, Llama 3 not integrated)
- ⚠️ **Tool Execution**: Not yet implemented

---

## 4. Technology Stack

### 4.1 Core Framework
- **MONAI**: Medical Open Network for AI - Core deep learning framework
  - Transform pipelines
  - Model architectures (SwinUNETR, SegResNet)
  - Loss functions (DiceCELoss, DiceFocalLoss)
  - Inference utilities (sliding window)
- **PyTorch**: Deep learning backend
- **MONAI Label**: AI-assisted annotation tool

### 4.2 Data Processing
- **pydicom**: DICOM file parsing
- **SimpleITK**: Medical image I/O and processing
- **nibabel**: NIfTI file handling
- **dcm2niix**: DICOM to NIfTI conversion (external tool)
- **BIDS Validator**: BIDS format validation

### 4.3 Data Science & ML
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation
- **scipy**: Scientific computing
- **einops**: Tensor operations (required for SwinUNETR)

### 4.4 Visualization & UI
- **Streamlit**: Web-based frontend
- **matplotlib**: Plotting and visualization
- **plotly**: Interactive visualizations
- **itkwidgets**: 3D medical image widgets

### 4.5 Infrastructure
- **PyYAML**: Configuration management
- **pytest**: Testing framework
- **tqdm**: Progress bars

### 4.6 Optional/Advanced
- **transformers**: For future VLM integration
- **google-cloud-storage**: For IDC dataset access
- **nbia-client**: For TCIA dataset access

---

## 5. Data Flow

### 5.1 Training Pipeline
```
DICOM Files
    ↓
[Ingestion] → BIDS Dataset
    ↓
[Validation] → Validated BIDS
    ↓
[Preprocessing] → Normalized Images + Labels
    ↓
[DataLoader] → Batched Training Data
    ↓
[Model Training] → Trained Model Checkpoint
    ↓
[Evaluation] → Performance Metrics
```

### 5.2 Inference Pipeline
```
New DICOM/Image
    ↓
[Preprocessing] → Normalized Image
    ↓
[Trained Model] → Segmentation Prediction
    ↓
[Post-processing] → Refined Segmentation
    ↓
[Evaluation] → Metrics (if ground truth available)
    ↓
[Reporting] → JSON Report + Visualization
```

---

## 6. Current Status

### 6.1 Implementation Status: ~60% Complete

#### ✅ **Fully Implemented**
- Data ingestion (DICOM → BIDS)
- Preprocessing pipelines
- Model training (SwinUNETR, SegResNet)
- Inference pipeline
- Evaluation metrics
- Streamlit frontend
- MONAI Label integration (basic)
- Uncertainty quantification

#### ⚠️ **Partially Implemented**
- Metadata preservation (limited fields)
- Active learning (foundation only)
- Multimodal AI (framework only, no VLM)
- 3D Slicer integration (instructions only)
- MONAI Bundles (Auto3DSeg only)

#### ❌ **Not Yet Implemented**
- HIPAA de-identification pipeline
- MONAI Deploy (MAP creation, Informatics Gateway)
- MLOps monitoring (drift detection)
- FHIR integration
- OHIF Viewer
- DICOM-SR generation
- Full VLM integration (VILA-M3, Llama 3)

### 6.2 Recent Fixes (Phase 0 Week 1)
- ✅ Fixed label preprocessing (binary normalization)
- ✅ Fixed training hanging on macOS (multiprocessing)
- ✅ Fixed MONAI deprecation warnings
- ✅ Added comprehensive test infrastructure

---

## 7. Use Cases

### 7.1 Research Use Cases
- **Organ Segmentation**: Segment organs in CT/MRI scans (lungs, liver, kidneys, etc.)
- **Pathology Detection**: Identify and segment tumors, lesions, abnormalities
- **Dataset Preparation**: Convert DICOM datasets to BIDS format for research
- **Model Development**: Train custom segmentation models
- **Performance Evaluation**: Comprehensive model evaluation with multiple metrics

### 7.2 Clinical Use Cases (Future)
- **AI-Assisted Diagnosis**: Provide segmentation predictions to radiologists
- **Workflow Integration**: Integrate with PACS systems via MONAI Deploy
- **Clinical Reporting**: Generate DICOM-SR reports for clinical systems
- **Multimodal Analysis**: Combine imaging with EHR data for comprehensive analysis

---

## 8. Key Features

### 8.1 Standardization
- **BIDS Compliance**: Follows Brain Imaging Data Structure standard
- **MONAI Best Practices**: Uses 2025-standard MONAI architecture
- **Reproducibility**: Configuration-driven, version-controlled workflows

### 8.2 Flexibility
- **Multiple Model Architectures**: SwinUNETR, SegResNet
- **Configurable Preprocessing**: Adjustable spacing, intensity ranges, augmentation
- **Multiple Loss Functions**: DiceCELoss, DiceFocalLoss
- **Custom or Automated Training**: Choose custom training or Auto3DSeg

### 8.3 Performance
- **Efficient Data Loading**: CacheDataset for fast training iterations
- **Sliding Window Inference**: Process large 3D volumes efficiently
- **GPU Acceleration**: CUDA support for fast training/inference
- **Batch Processing**: Process multiple subjects efficiently

### 8.4 Quality Assurance
- **Comprehensive Validation**: BIDS structure, file integrity, data consistency
- **Uncertainty Quantification**: Monte Carlo Dropout for prediction confidence
- **Detailed Evaluation**: Multiple metrics for thorough model assessment

---

## 9. System Requirements

### 9.1 Hardware
- **CPU**: Multi-core processor recommended
- **GPU**: CUDA-capable GPU highly recommended (training on CPU is very slow)
- **RAM**: 16GB+ recommended (for caching datasets)
- **Storage**: Sufficient space for datasets and model checkpoints

### 9.2 Software
- **Python**: 3.8 or higher
- **Operating System**: macOS, Linux, Windows
- **External Tools**: dcm2niix (for DICOM conversion)

### 9.3 Dependencies
- Core: MONAI, PyTorch, pydicom, SimpleITK, nibabel
- See `requirements.txt` for complete list

---

## 10. Project Structure

```
MedAI/
├── data/                    # Data storage
│   ├── raw_dicom/          # Input DICOM files
│   ├── bids_dataset/       # BIDS-organized data
│   └── derivatives/        # Labels and outputs
├── src/                     # Source code
│   ├── ingestion/          # DICOM processing
│   ├── preprocessing/      # Transform pipelines
│   ├── modeling/           # Training & inference
│   ├── annotation/        # MONAI Label integration
│   ├── multimodal/        # Multimodal AI framework
│   └── visualization/     # Visualization utilities
├── scripts/                # Executable scripts
├── configs/                # Configuration files
├── models/                 # Trained model checkpoints
├── tests/                  # Test suite
├── docs/                   # Documentation
└── notebooks/              # Jupyter notebooks
```

---

## 11. Future Roadmap

### Phase 1: Production Readiness (4-6 weeks)
- HIPAA-compliant de-identification
- MONAI Bundle conversion
- DICOM-SR generation

### Phase 2: Clinical Deployment (6-8 weeks)
- MONAI Deploy integration
- Informatics Gateway
- Workflow Manager

### Phase 3: MLOps (4-6 weeks)
- Model monitoring
- Drift detection
- Performance tracking

### Phase 4: Clinical Integration (6-8 weeks)
- FHIR interoperability
- OHIF Viewer
- PACS integration

### Phase 5-6: Advanced Features (18-22 weeks)
- Full VLM integration
- Federated learning
- Generative AI

---

## 12. Key Documentation

- **README.md**: Quick start and overview
- **PROJECT_PLAN.md**: Detailed project plan and roadmap
- **QUICKSTART.md**: Step-by-step getting started guide
- **docs/TRAINING.md**: Comprehensive training guide
- **docs/ISSUE_TRACKING.md**: Known issues and fixes
- **docs/FIXES_APPLIED.md**: Recent fixes and improvements

---

**For detailed technical information, see PROJECT_PLAN.md and component-specific documentation.**

