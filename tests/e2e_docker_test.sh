#!/bin/bash
set -e

echo "=============================================="
echo "Starting E2E Docker Integration Test"
echo "=============================================="

# Define paths
CONFIG_FILE="params.yaml"
BACKUP_CONFIG="params.yaml.bak"

# 1. Backup original configuration
echo "-> Backing up original parameters..."
cp $CONFIG_FILE $BACKUP_CONFIG

# Ensure backup is ALWAYS restored even if the script crashes or is aborted via Ctrl+C
trap 'mv $BACKUP_CONFIG $CONFIG_FILE' EXIT

# 2. Modify params.yaml dynamically for a rapid integration run (3 clients, 1 adversary, 1 round)
echo "-> Configuring params.yaml for rapid test execution..."
python3 -c "
import yaml

with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)

# Override with minimal parameters for CI pipeline testing
if 'simulation' not in config: config['simulation'] = {}
config['simulation']['num_rounds'] = 1
config['simulation']['num_clients'] = 3
if 'dataset' not in config: config['dataset'] = {}
config['dataset']['total_samples'] = 500

config['simulation']['num_false_nodes'] = 1 # The adversary

if 'server' not in config: config['server'] = {}
config['server']['consensus_samples'] = 50 # Small server partition so clients get data

# Keep SimGNN fine-tuning minimal
if 'core_logic' not in config: config['core_logic'] = {}
config['core_logic']['simgnn_epochs'] = 1
config['core_logic']['notears_max_iter'] = 3 # Fast NOTEARS

with open('$CONFIG_FILE', 'w') as f:
    yaml.dump(config, f)
"

# Clean up previous artifacts to ensure a fresh test
echo "-> Scrubbing previous models and test artifacts..."
rm -rf saved_models/asia/*
rm -rf graphs/G*.png

# 3. Spin up Docker Build and Execute Simulation
echo "-> Initiating Docker build (causal-por:latest)..."
docker compose build

echo "-> Executing FedNEAT Simulation (Causal PoR)..."
# We run the 'sim' service explicitly
docker compose --profile tools run --rm sim

# 4. Assessment and Validation
echo "-> Validating generated artifacts..."
DATASET_NAME="asia"
FAILED=0

# Ensure directories exist
if [ ! -d "saved_models/${DATASET_NAME}" ]; then
    echo "❌ ERROR: Model directory saved_models/${DATASET_NAME} was not created."
    FAILED=1
fi

# Essential output files
REQUIRED_FILES=(
    "saved_models/${DATASET_NAME}/consensus_graph.gpickle"
    "saved_models/${DATASET_NAME}/simgnn_pretrained.pt"
    "saved_models/${DATASET_NAME}/ged_scores.json"
    "saved_models/${DATASET_NAME}/simulation_logs.json"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ SUCCESS: Found $file"
    else
        echo "❌ ERROR: Missing $file"
        FAILED=1
    fi
done

# We expect both honest and rejected client graph samples if the simulation didn't statically collapse
if [ -f "saved_models/${DATASET_NAME}/honest_graph_sample.gpickle" ]; then
    echo "✅ SUCCESS: Found honest graph sample."
else
    echo "⚠️ WARNING: Missing honest graph. (Simulation data may have caused 100% rejection)."
fi

if [ -f "saved_models/${DATASET_NAME}/rejected_graph_sample.gpickle" ]; then
    echo "✅ SUCCESS: Found rejected adversarial graph sample."
else
    echo "⚠️ WARNING: Missing rejected graph. (Simulation data may have caused 100% acceptance)."
fi

# 5. Cleanup and Revert
if [ $FAILED -eq 1 ]; then
    echo "=============================================="
    echo "E2E INTEGRATION TEST FAILED!"
    echo "=============================================="
    exit 1
else
    echo "=============================================="
    echo "E2E INTEGRATION TEST PASSED SUCCESSFULLY! 🚀"
    echo "=============================================="
    exit 0
fi
