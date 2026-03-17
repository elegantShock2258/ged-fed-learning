#!/bin/bash
# remote_run.sh — Executed directly on the Vast.AI instance (called by run_vast_simulation.py)
# This script does NOT need to be run manually.

set -e  # Exit on any error

cd /root/FYP

echo '===== Checking environment ====='
if [ ! -d "/root/FYP/.venv" ]; then
    echo 'Creating virtual environment...'
    apt-get update && apt-get install -y python3.12-venv
    python3.12 -m venv --system-site-packages .venv
fi
source .venv/bin/activate

echo '===== Installing requirements ====='
pip install -r requirements.txt

# Read dataset name from params.yaml
DATASET=$(python3 -c "import yaml; c=yaml.safe_load(open('params.yaml')); print(c.get('dataset',{}).get('name','asia'))")
echo "Dataset: $DATASET"

echo "===== Checking for Global Consensus Graph (saved_models/${DATASET}/) ====="
if [ ! -f "saved_models/${DATASET}/consensus_graph.gpickle" ]; then
    echo "Generating global consensus graph for ${DATASET}..."
    mkdir -p "saved_models/${DATASET}"
    python server/generate_consensus.py
else
    echo "Consensus graph already present."
fi

echo "===== Checking for SimGNN model ====="
if [ ! -f "saved_models/${DATASET}/simgnn_pretrained.pt" ]; then
    echo "Training SimGNN..."
    python server/train_simgnn.py
else
    echo "SimGNN already trained."
fi

echo '===== Starting Federated Simulation ====='
python federated_sim.py

echo '===== Done! ====='