# Dockerfile (Optimized for CPU-only environments)
# ============================================================
FROM python:3.12-slim

# ---------- System dependencies ----------------------------------
# curl: needed for the HEALTHCHECK
# git:  needed by some Python packages that fetch from GitHub
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# ---------- Environment ------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ---------- Working directory ------------------------------------
WORKDIR /app

# ---------- Install PyTorch (CPU wheel) --------------------------
RUN pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# ---------- Install torch-geometric separately -------------------
# torch-geometric depends on torch being already installed
RUN pip install torch-geometric==2.7.0

# ---------- Install remaining dependencies -----------------------
COPY requirements.txt .
RUN pip install -r requirements.txt

# ---------- Copy application code --------------------------------
# .dockerignore excludes: .venv, saved_models, __pycache__, *.pt, *.gpickle
COPY . .

# ---------- Create directories that are volume-mounted at runtime -
RUN mkdir -p saved_models

# ---------- Expose Streamlit port --------------------------------
EXPOSE 8501

# ---------- Health-check (Streamlit readiness probe) --------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# ---------- Entrypoint -------------------------------------------
CMD ["streamlit", "run", "app.py", \
    "--server.address=0.0.0.0", \
    "--server.port=8501", \
    "--server.headless=true"]
