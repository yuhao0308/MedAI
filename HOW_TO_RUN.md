# 🚀 How to Run the Frontend

## Quick Start

### Option 1: Using the Run Script (Recommended)

```bash
./run_app.sh
```

### Option 2: Using Python Module

```bash
python3 -m streamlit run app.py
```

### Option 3: Direct Command (if streamlit is in PATH)

```bash
streamlit run app.py
```

## If You Get "command not found: streamlit"

This happens when streamlit is installed but not in your PATH. Use one of these solutions:

### Solution 1: Use Python Module (Easiest)
```bash
python3 -m streamlit run app.py
```

### Solution 2: Add to PATH
```bash
# Add Python user bin to PATH (add to ~/.zshrc or ~/.bashrc)
export PATH="$HOME/Library/Python/3.9/bin:$PATH"

# Then reload shell
source ~/.zshrc  # or source ~/.bashrc

# Now you can use:
streamlit run app.py
```

### Solution 3: Use Full Path
```bash
~/Library/Python/3.9/bin/streamlit run app.py
```

## Custom Port

If port 8501 is already in use:

```bash
python3 -m streamlit run app.py --server.port 8502
```

## What Happens When You Run

1. **Streamlit starts** - The web server launches
2. **Browser opens** - Automatically opens at `http://localhost:8501`
3. **App loads** - You'll see the Medical Imaging AI Pipeline interface

## Troubleshooting

### Port Already in Use
```bash
# Kill existing Streamlit process
pkill -f streamlit

# Or use different port
python3 -m streamlit run app.py --server.port 8502
```

### Module Not Found Errors
```bash
# Make sure you're in the project directory
cd /Users/yuhaowang/Documents/Itlize/MedAI

# Install missing dependencies
pip3 install -r requirements.txt
```

### Can't Find BIDS Dataset
- Make sure you've processed DICOM files first
- Check that `./data/bids_dataset/` exists
- Use the "Upload & Process" page to convert DICOM files

## First Time Setup

1. **Install Streamlit** (if not already installed):
   ```bash
   pip3 install streamlit
   ```

2. **Install other dependencies**:
   ```bash
   pip3 install -r requirements.txt
   ```

3. **Run the app**:
   ```bash
   python3 -m streamlit run app.py
   ```

## Accessing the App

Once running, the app will be available at:
- **Local**: http://localhost:8501
- **Network**: http://YOUR_IP:8501 (if you want to access from other devices)

## Stopping the App

Press `Ctrl+C` in the terminal where Streamlit is running.

## Next Steps After Running

1. **Process Your Data**: Go to "Upload & Process" tab
   - Process your 50 thorax CT patients: Select `./New_thorax_ct_dicom` directory
   
2. **Explore Dataset**: Go to "Dataset Explorer" tab
   - Browse processed subjects
   - View images and masks
   
3. **Visualize**: Go to "Visualization" tab
   - View 3D medical images
   - Navigate through slices
   - Overlay segmentation masks
   
4. **Run Inference**: Go to "Model Inference" tab
   - Run AI models (after training)

Enjoy your medical imaging AI pipeline! 🎉

---

## ☁️ Cloud GPU Inference (Optional)

Enable faster inference using Modal Labs' serverless GPU computing.

### Why Use Cloud Inference?

- **Faster**: T4 GPU is significantly faster than CPU inference
- **Free Tier**: $30/month in free credits (~100-200 inference runs)
- **No GPU Required**: Run on any machine, processing happens in the cloud
- **Automatic Fallback**: Falls back to local inference if cloud is unavailable

### Setup Modal

1. **Install Modal SDK**:
   ```bash
   pip install modal
   ```

2. **Authenticate**:
   ```bash
   modal token new
   ```
   This opens a browser to authenticate with your Modal account.

3. **Deploy the Inference Endpoint**:
   ```bash
   modal deploy src/cloud/modal_inference.py
   ```

4. **Verify Setup**:
   The cloud status will appear in the Analyze page when you run the app.

### Using Cloud Inference

Once configured, you'll see a "☁️ Use Cloud GPU" checkbox in the Analyze page.

- **Enabled**: Inference runs on Modal's T4 GPU (~5-10x faster)
- **Disabled**: Inference runs locally (slower but free)
- **Fallback**: If cloud fails, automatically uses local inference

### Estimated Performance

| Method | Hardware | Time per Image |
|--------|----------|----------------|
| Local CPU | Intel i7 | ~5-10 minutes |
| Local GPU | GTX 1080 | ~30-60 seconds |
| Cloud GPU | Modal T4 | ~15-30 seconds |

### Cost Estimation

Modal T4 GPU: ~$0.20/hour
- 1 inference ≈ 30 seconds = ~$0.003
- $30 free credits ≈ 10,000 inference runs

### Troubleshooting Cloud Inference

**"Modal not available"**
```bash
pip install modal
modal token new
```

**"Model not found in volume"**
The first inference will automatically upload the model (~500MB).
Subsequent runs will be faster.

**Timeout errors**
Increase timeout in `config/cloud_config.yaml`:
```yaml
modal:
  function_timeout: 900  # 15 minutes
```
