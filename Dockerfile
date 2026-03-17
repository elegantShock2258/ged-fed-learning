FROM python:3.12-slim

# Install system dependencies (no libGL needed for tabular data)
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install PyTorch + dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application (excludes .venv, saved_models via .dockerignore)
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run the Streamlit dashboard
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
