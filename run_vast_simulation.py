#!/usr/bin/env python3
import yaml
import subprocess
import sys
import os
import time

def run_cmd(cmd, shell=False, check=True, cwd=None):
    print(f"\n[HOST] Executing: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    result = subprocess.run(cmd, shell=shell, check=check, cwd=cwd, text=True)
    return result

def get_ssh_details(instance_id):
    """Gets the SSH IP and Port from the vastai CLI"""
    print(f"[HOST] Fetching SSH details for instance {instance_id}...")
    try:
        # Assuming vastai CLI is installed in the current environment
        result = subprocess.run(['vastai', 'ssh-url', str(instance_id)], capture_output=True, text=True, check=True)
        url = result.stdout.strip()
        if not url.startswith("ssh://"):
            raise ValueError(f"Unexpected output from vastai: {url}")
        
        # Format: ssh://root@IP:PORT
        url = url.replace("ssh://", "")
        user_host, port = url.split(":")
        user, host = user_host.split("@")
        return user, host, port
    except Exception as e:
        print(f"[HOST] Failed to get SSH url for instance {instance_id}: {e}")
        sys.exit(1)

def run_remote_cmd(user, host, port, cmd):
    ssh_cmd = [
        "ssh",
        "-p", port,
        "-o", "StrictHostKeyChecking=no",
        f"{user}@{host}",
        cmd
    ]
    return run_cmd(ssh_cmd)

def main():
    # Load constraints from params.yaml
    if not os.path.exists("params.yaml"):
        print("[HOST] Cannot find params.yaml!")
        sys.exit(1)

    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)

    if "vastai" not in config or "instance_id" not in config["vastai"]:
        print("[HOST] 'vastai.instance_id' is missing in params.yaml!")
        sys.exit(1)

    instance_id = config["vastai"]["instance_id"]
    
    # 1. Get SSH details
    user, host, port = get_ssh_details(instance_id)
    print(f"[HOST] Successfully retrieved SSH details -> {user}@{host}:{port}")

    print("[HOST] Syncing project files to remote instance...")
    rsync_cmd = [
        "rsync", "-avz",
        "-e", f"ssh -p {port} -o StrictHostKeyChecking=no",
        "--exclude", ".venv",
        "--exclude", ".git",
        "--exclude", "__pycache__",
        "--exclude", "datasets/isic2019", # don't sync heavy datasets
        "--exclude", "saved_models",
        "./",  # current directory (FYP)
        f"{user}@{host}:/root/FYP/"
    ]
    run_cmd(rsync_cmd)
        
    # Configure ~/.kaggle remotely so datasets can be downloaded
    print("[HOST] Syncing Kaggle credentials...")
    run_remote_cmd(user, host, port, "mkdir -p /root/.kaggle")
    kaggle_path = os.path.expanduser("~/.kaggle/kaggle.json")
    if os.path.exists(kaggle_path):
        run_cmd([
            "rsync", "-avz", "-e", f"ssh -p {port} -o StrictHostKeyChecking=no",
            kaggle_path,
            f"{user}@{host}:/root/.kaggle/kaggle.json"
        ])
        run_remote_cmd(user, host, port, "chmod 600 /root/.kaggle/kaggle.json")
    else:
        print("[HOST] WARNING: ~/.kaggle/kaggle.json not found locally. Dataset download will fail remotely!")

    # 3. Setup Remote Environment (create venv if missing, install requirements)
    print("[HOST] Ensuring remote Python environment is ready...")
    setup_script = """
    cd /root/FYP
    
    # Ensure a venv exists (using system site packages to leverage pre-installed PyTorch in vast images)
    if [ ! -d ".venv" ]; then
        echo "[REMOTE] Creating virtual environment..."
        apt-get update && apt-get install -y python3.12-venv
        python3 -m venv --system-site-packages .venv
    fi
    
    source .venv/bin/activate
    
    echo "[REMOTE] Installing dependencies..."
    # Install PyTorch 2.6 (stable) with CUDA 12.4 to support RTX 5090 (Blackwell architecture)
    PIP_CACHE_DIR=/root/tmp pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --upgrade
    # Install safely in case cache is full or system packages conflict
    PIP_CACHE_DIR=/root/tmp pip install -r requirements.txt
    """
    run_remote_cmd(user, host, port, setup_script)

    # 4. Run the remote simulation protocol
    print("[HOST] Starting the federated simulation on the remote machine...")
    sim_script = """
    cd /root/FYP
    source .venv/bin/activate

    echo "=================================="
    echo "[REMOTE] 1. Downloading Dataset via Aria2c..."
    echo "=================================="
    if [ ! -d "datasets/isic2019/ISIC_2019_Training_Input" ]; then
        apt-get update && apt-get install -y aria2 unzip
        mkdir -p datasets/isic2019
        aria2c -x 16 -s 16 https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip -d datasets/isic2019/
        cd datasets/isic2019/
        unzip -q ISIC_2019_Training_Input.zip
        rm ISIC_2019_Training_Input.zip
        cd ../../
    else
        echo "Dataset already exists natively."
    fi

    echo "=================================="
    echo "[REMOTE] Installing PyTorch Nightly for Blackwell (sm_120) compatibility..."
    echo "=================================="
    pip install --upgrade --pre "torch>=2.7" "torchvision>=0.22" "torchaudio>=2.6" --index-url https://download.pytorch.org/whl/nightly/cu128
    echo "=================================="
    echo "[REMOTE] 2. Generating Consensus..."
    echo "=================================="
    python server/generate_consensus.py

    echo "=================================="
    echo "[REMOTE] 3. Pre-training SimGNN..."
    echo "=================================="
    python server/train_simgnn.py

    echo "=================================="
    echo "[REMOTE] 4. Running Federated Sim..."
    echo "=================================="
    python federated_sim.py
    """
    
    try:
        run_remote_cmd(user, host, port, sim_script)
    except subprocess.CalledProcessError:
        print("\n[HOST] An error occurred during remote simulation execution.")
    finally:
        # 5. Always attempt to sync the models and logs back down to the local machine
        print("\n[HOST] Syncing resulting models and logs back to local machine...")
        pull_cmd = [
            "rsync", "-avz",
            "-e", f"ssh -p {port} -o StrictHostKeyChecking=no",
            f"{user}@{host}:/root/FYP/saved_models/",
            "./saved_models/"
        ]
        try:
            run_cmd(pull_cmd)
        except Exception as e:
            print(f"[HOST] Failed to sync models back: {e}")

    print("\n[HOST] Done!")

if __name__ == "__main__":
    main()
