# 🚀 Frontend Quick Start Guide

## Installation

```bash
# Install Streamlit and dependencies
pip install streamlit

# Or install all requirements
pip install -r requirements.txt
```

## Run the App

```bash
# Simple way
streamlit run app.py

# Or use the run script
./run_app.sh
```

The app will automatically open in your browser at `http://localhost:8501`

## What You Can Do

### 1. **Home Dashboard** 🏠
- View pipeline statistics
- Check system status
- See overview of features

### 2. **Upload & Process** 📤
- Upload DICOM files directly
- Process existing directories
- Convert to BIDS format automatically

### 3. **Dataset Explorer** 📂
- Browse your BIDS dataset
- View subject details
- Explore images and masks
- Check metadata

### 4. **Visualization** 🖼️
- View 3D medical images
- Navigate through slices (Axial, Coronal, Sagittal)
- Overlay segmentation masks
- See mask statistics

### 5. **Model Inference** 🤖
- Run AI models on images
- Get predictions
- View uncertainty maps
- Analyze results

## Features

✅ **Interactive UI** - Easy-to-use web interface  
✅ **Real-time Processing** - See progress as files are processed  
✅ **3D Visualization** - Multi-planar medical image viewer  
✅ **Dataset Management** - Browse and explore your data  
✅ **Model Integration** - Run AI inference directly from UI  

## Troubleshooting

**Port already in use?**
```bash
streamlit run app.py --server.port 8502
```

**Module not found?**
```bash
pip install -r requirements.txt
```

**Images not showing?**
- Make sure you've processed DICOM files first
- Check that `./data/bids_dataset/` exists

## Next Steps

1. **Process your data**: Use "Upload & Process" to convert DICOM files
2. **Explore dataset**: Browse your processed data in "Dataset Explorer"
3. **Visualize**: View images and segmentations in "Visualization"
4. **Train models**: Use the backend scripts to train models
5. **Run inference**: Use "Model Inference" to test your models

Enjoy your medical imaging AI pipeline! 🎉


