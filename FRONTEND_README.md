# Frontend Documentation

## 🚀 Quick Start

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the App

```bash
# Option 1: Use the run script
./run_app.sh

# Option 2: Direct Streamlit command
streamlit run app.py

# Option 3: Custom port
streamlit run app.py --server.port 8501
```

The app will open automatically in your browser at `http://localhost:8501`

## 📱 Features

### 1. Home Page
- Dashboard overview
- System status
- Pipeline statistics
- Quick start guide

### 2. Upload & Process
- **Upload Files**: Upload DICOM files directly through the web interface
- **Process Existing Directory**: Process DICOM files from existing directories
- Automatic BIDS conversion
- Progress tracking

### 3. Dataset Explorer
- Browse processed BIDS dataset
- View subject details
- Explore images and segmentation masks
- Metadata inspection

### 4. Visualization
- Interactive 3D medical image viewer
- Multi-planar views (Axial, Coronal, Sagittal)
- Slice-by-slice navigation
- Segmentation overlay
- Mask statistics

### 5. Model Inference
- Run AI models on medical images
- Uncertainty quantification
- Prediction visualization
- Model selection

## 🎨 Interface Overview

```
┌─────────────────────────────────────────┐
│  🏥 Medical Imaging AI Pipeline        │
├─────────────────────────────────────────┤
│                                         │
│  [Home] [Upload] [Explorer] [Viz] [AI] │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │                                 │   │
│  │     Main Content Area           │   │
│  │                                 │   │
│  └─────────────────────────────────┘   │
│                                         │
└─────────────────────────────────────────┘
```

## 🔧 Configuration

Edit `.streamlit/config.toml` to customize:
- Theme colors
- Server settings
- Port configuration

## 📦 Deployment

### Local Development
```bash
streamlit run app.py
```

### Production Deployment

#### Option 1: Streamlit Cloud
1. Push code to GitHub
2. Connect to Streamlit Cloud
3. Deploy automatically

#### Option 2: Docker
```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

#### Option 3: Custom Server
```bash
# Use nginx as reverse proxy
# Configure SSL certificates
# Set up authentication if needed
```

## 🐛 Troubleshooting

### Port Already in Use
```bash
# Kill existing Streamlit process
pkill -f streamlit

# Or use a different port
streamlit run app.py --server.port 8502
```

### Module Not Found
```bash
# Ensure you're in the project root
cd /path/to/MedAI

# Install dependencies
pip install -r requirements.txt
```

### Images Not Loading
- Check that BIDS dataset exists at `./data/bids_dataset/`
- Verify file permissions
- Ensure NIfTI files are valid

## 📝 Notes

- The frontend is built with Streamlit for rapid development
- All visualization uses matplotlib for compatibility
- For production, consider migrating to a more robust framework (React + FastAPI)
- Model inference requires trained models in `./models/` directory

## 🔗 Integration

The frontend integrates with:
- `src/ingestion/` - DICOM processing
- `src/preprocessing/` - Data transforms
- `src/modeling/` - AI models
- `src/visualization/` - Image viewers

All backend functionality is accessible through the Python modules.


