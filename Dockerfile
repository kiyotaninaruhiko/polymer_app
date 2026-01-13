# Polymer SMILES Descriptor Generator
# Python 3.12 + PyTorch 2.6+ (CPU版)

FROM python:3.12-slim

LABEL maintainer="Polymer Descriptor App"
LABEL description="Polymer SMILES Descriptor Generator with RDKit, Morgan FP, and Transformer Embeddings"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir 'torch>=2.6.0' --index-url https://download.pytorch.org/whl/cpu

# Copy application code
COPY . .

# Create cache directory
RUN mkdir -p /root/.poly_pred_cache

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.headless=true"]
