
    cd /root/FYP
    
    echo 'Checking environment...'
    if [ ! -d "/root/FYP/.venv" ]; then
        echo 'Creating virtual environment...'
        apt-get update && apt-get install -y python3.12-venv
        python3.12 -m venv --system-site-packages .venv
    fi
    source .venv/bin/activate
    
    echo 'Installing requirements...'
    pip3 install -r requirements.txt
    
    # Check if dataset is downloaded
    isic_path=$(cat params.yaml | grep isic_path | awk '{print $2}')
    if [ ! -d "$isic_path" ] || [ -z "$(ls -A $isic_path 2>/dev/null)" ]; then
        echo "Dataset not found at $isic_path. Downloading..."
        python datasets/download_isic.py
    else
        echo "Dataset already present."
    fi
    
    echo "Checking for Global Consensus Graph..."
    if [ ! -f "saved_models/global_consensus_graph.gpickle" ]; then
        echo "Generating global consensus graph..."
        mkdir -p saved_models
        python server/generate_consensus.py
    else
        echo "Global Consensus Graph already present."
    fi
    
    if [ ! -f "saved_models/simgnn_pretrained.pt" ]; then
        echo "Training SimGNN..."
        python server/train_simgnn.py
    else
        echo "SimGNN already trained."
    fi
    
    echo 'Starting Federated Simulation...'
    python federated_sim.py
    