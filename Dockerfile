FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    dcm2niix \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies first (for better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Railway uses PORT env variable
EXPOSE 8501

CMD streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
