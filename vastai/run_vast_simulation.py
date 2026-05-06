#!/usr/bin/env python3
"""
Vast.AI Remote Simulation Launcher
------------------------------------
Syncs the local FYP project to a Vast.AI GPU instance,
sets up the Python environment, runs the simulation pipeline,
and syncs results back to local machine.

Usage:
    cd /path/to/FYP      # Run from the project ROOT, not from vastai/
    python vastai/run_vast_simulation.py

Configuration:
    Set vastai.instance_id in params.yaml before running.
"""
import yaml
import subprocess
import sys
import os
import time

# The script is in vastai/ but should be run from the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def run_cmd(cmd, shell=False, check=True, cwd=None):
    print(f"\n[HOST] Executing: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    result = subprocess.run(cmd, shell=shell, check=check, cwd=cwd or PROJECT_ROOT, text=True)
    return result

def get_ssh_details(instance_id):
    """Gets the SSH IP and Port from the vastai CLI"""
    print(f"[HOST] Fetching SSH details for instance {instance_id}...")
    try:
        result = subprocess.run(['vastai', 'ssh-url', str(instance_id)], capture_output=True, text=True, check=True)
        url = result.stdout.strip()
        if not url.startswith("ssh://"):
            raise ValueError(f"Unexpected output from vastai: {url}")
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
    params_path = os.path.join(PROJECT_ROOT, "params.yaml")
    if not os.path.exists(params_path):
        print("[HOST] Cannot find params.yaml!")
        sys.exit(1)

    with open(params_path, "r") as f:
        config = yaml.safe_load(f)

    if "vastai" not in config or "instance_id" not in config["vastai"]:
        print("[HOST] 'vastai.instance_id' is missing in params.yaml!")
        sys.exit(1)

    instance_id = config["vastai"]["instance_id"]
    dataset_name = config.get("dataset", {}).get("name", "asia")
    
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
        "--exclude", "saved_models",  # don't sync models (they'll be generated remotely)
        f"{PROJECT_ROOT}/",
        f"{user}@{host}:/root/FYP/"
    ]
    run_cmd(rsync_cmd)

    # 2. Setup Remote Python Environment
    print("[HOST] Ensuring remote Python environment is ready...")
    setup_script = """
    cd /root/FYP
    if [ ! -d ".venv" ]; then
        echo "[REMOTE] Creating virtual environment..."
        apt-get update && apt-get install -y python3.12-venv
        python3 -m venv --system-site-packages .venv
    fi
    source .venv/bin/activate
    echo "[REMOTE] Installing dependencies..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --upgrade
    pip install -r requirements.txt
    """
    run_remote_cmd(user, host, port, setup_script)

    # 3. Run the remote simulation pipeline
    print(f"[HOST] Starting the federated simulation (dataset={dataset_name}) on the remote machine...")
    sim_script = f"""
    cd /root/FYP
    source .venv/bin/activate

    echo "=================================="
    echo "[REMOTE] 1. Generating Global Consensus Graph for {dataset_name}..."
    echo "=================================="
    python server/generate_consensus.py

    echo "=================================="
    echo "[REMOTE] 2. Pre-training SimGNN Logic Validator..."
    echo "=================================="
    python server/train_simgnn.py

    echo "=================================="
    echo "[REMOTE] 3. Running Federated Simulation..."
    echo "=================================="
    python experiments/run_por_sim.py
    """
    
    try:
        run_remote_cmd(user, host, port, sim_script)
    except subprocess.CalledProcessError:
        print("\n[HOST] An error occurred during remote simulation execution.")
    finally:
        # 4. Sync results back to local (dataset-specific saved_models folder)
        print(f"\n[HOST] Syncing saved_models/{dataset_name}/ results back to local machine...")
        pull_cmd = [
            "rsync", "-avz",
            "-e", f"ssh -p {port} -o StrictHostKeyChecking=no",
            f"{user}@{host}:/root/FYP/saved_models/",
            f"{PROJECT_ROOT}/saved_models/"
        ]
        try:
            run_cmd(pull_cmd)
        except Exception as e:
            print(f"[HOST] Failed to sync models back: {e}")

    print("\n[HOST] Done!")

if __name__ == "__main__":
    main()
