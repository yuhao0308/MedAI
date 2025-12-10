#!/bin/bash
# Run Streamlit app

echo "Starting Medical Imaging AI Pipeline Frontend..."
echo "Open your browser to http://localhost:8501"

# Use python3 -m streamlit to avoid PATH issues
python3 -m streamlit run app.py --server.port 8501 --server.address localhost

