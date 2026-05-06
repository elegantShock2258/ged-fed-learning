# ============================================================
# Causal Proof of Reasoning — Federated Learning
# Dockerfile  (CPU-first; GPU support via docker-compose)
# ============================================================
FROM python:3.12-slim

# ---------- System dependencies ----------------------------------
# build-essential / gcc / g++: required to compile C extensions in
#   scipy, pgmpy, scikit-learn, grpcio, cryptography, ray
# libgomp1:  OpenMP runtime needed by numpy/scipy parallel kernels
# curl:      needed for the HEALTHCHECK
# git:       needed by some Python packages that fetch from GitHub
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libgomp1 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# ---------- Environment ------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg

# ---------- Working directory ------------------------------------
WORKDIR /app

# ---------- Install PyTorch (CPU wheel) --------------------------
# Installed before requirements.txt so torch-geometric can detect
# the correct torch version at install time.
#
# GPU users: override this in docker-compose by setting:
#   TORCH_INSTALL_URL=https://download.pytorch.org/whl/cu126
# and rebuilding. See docker-compose.yml for the GPU profile.
ARG TORCH_INSTALL_URL=https://download.pytorch.org/whl/cpu
RUN pip install torch torchvision torchaudio \
    --index-url ${TORCH_INSTALL_URL}

# ---------- Install torch-geometric ------------------------------
# Must come AFTER torch so PyG can detect the installed torch version.
# PyG 2.x no longer requires torch_scatter / torch_sparse C++ extensions.
RUN pip install torch-geometric==2.7.0

# ---------- Install remaining dependencies -----------------------
# NOTE: torch and torch-geometric are NOT listed in requirements.txt
#       to avoid double-installation and version conflicts.
COPY requirements.txt .
RUN pip install -r requirements.txt

# ---------- Copy application code --------------------------------
# .dockerignore excludes: .venv, saved_models, __pycache__, *.pt, *.gpickle
COPY . .

# ---------- Create runtime directories ---------------------------
RUN mkdir -p saved_models/asia saved_models/baseline results/figures

# ---------- Expose Streamlit port --------------------------------
EXPOSE 8501

# ---------- Health-check (Streamlit readiness probe) -------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# ---------- Default entrypoint: Streamlit dashboard --------------
# To run the FL simulation instead, use docker-compose (see sim service)
# or: docker run causal-por:latest python experiments/run_por_sim.py
CMD ["streamlit", "run", "dashboard/app.py", \
    "--server.address=0.0.0.0", \
    "--server.port=8501", \
    "--server.headless=true"]
