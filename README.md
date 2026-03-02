# Causal Proof of Reasoning (PoR) for Agentic Federated Learning

This project implements a Causal Proof of Reasoning (PoR) framework to secure Agentic Federated Learning environments against Explanation Poisoning and Distributed Backdoor attacks. 

Instead of relying solely on model weights, clients additionally extract a **causal graph** from their internal latent representations. A server-side Governance Layer uses a Graph Neural Network (SimGNN) to measure the Graph Edit Distance (GED) between the submitted graph and the global consensus logic, rejecting structural anomalies before aggregating the updates.

---

## 🏗️ Architecture

1. **Client (Deliberative Agent):**
    - **Perception/Action Module:** Trains a Deep Learning model (ResNet50 feature extractor) on a local partition of the `Fed-ISIC2019` dataset.
    - **Cognitive Module:** Uses a PyTorch implementation of the NOTEARS continuous optimization algorithm to build a Directed Acyclic Graph (DAG) representing the dependencies forming the agent's logic.
2. **Adversary (Explanation Poisoning Attacker):**
    - Executes a clean-label backdoor attack (pixels masked in corners).
    - Spoofs the causal logic graph (Scaffolding attack) to try and trick the server's GED logic validator.
3. **Server (Governance Layer):**
    - **Logic Validator (SimGNN):** Evaluates if the Graph Edit Distance (GED) of incoming logic exceeds threshold $\tau$.
    - **PoR Aggregator (Dual Strategy):** Filters out malicious clients based on the validator's output. Averages the surviving weights and applies barycenter voting to update the global consensus graph.

---

## ⚙️ Configuration (params.yaml)

All critical parameters are managed natively through `params.yaml` at the root of the project. 

> [!WARNING]
> **Changing `core_logic` parameters requires full logic retraining!**
> If you change parameters like `latent_feature_dim` or `causal_edge_threshold`, it changes the foundational structure of the graphs. You MUST delete existing `saved_models/` and re-run `python server/train_simgnn.py` to pre-train the Logic Validator again on the new graph distributions. 

The file is split into tunable `simulation` parameters (like number of clients) and static `core_logic` parameters. See the inline comments in `params.yaml` for granular details.

---

## 🚀 Setup & Execution

### 1. Prerequisites and Dataset
The project was designed for Python 3.12 within a virtual environment. 

First, install the required dependencies:
```bash
pip install -r requirements.txt
```

To download the `Fed-ISIC2019` dataset, ensure your `~/.kaggle/kaggle.json` credentials are set, then run:

```bash
# This cleans corrupted downloads, fixes flamby yaml, and initiates the Kaggle download.
python datasets/rebuild_isic.py
# Verify your dataset path matches the one listed in `params.yaml`!
```

### 2. Pre-train the SimGNN Logic Validator
Before running the main simulation, the server's Graph Neural Network needs to understand what Logic Distances (GED) look like computationally. 

```bash
python server/train_simgnn.py 
# This runs locally and generates synthetic DAGs to pre-train the GED estimator. 
# It will save the resulting model to `simgnn_pretrained.pt`.
```

### 3. Run the Federated Learning Simulation
To execute the multi-round federated training run with real dataset paritions (Honest Nodes and False Nodes interacting):

```bash
python federated_sim.py
```
*Note: Due to massive inherent parallelism when loading 30 ResNet models simultaneously, `params.yaml` sets `ray_cpus_per_actor: 4` to artificially throttle the number of concurrent Flower/Ray actors running. If you hit OOM (Out of Memory) kills in your terminal, raise this number to force Ray to run slower/more sequentially.*

### 4. Weights and Models
Post-simulation, the framework creates a `saved_models` directory containing:
* `global_model_round_X.pt` - The global model parameters at round X.
* `global_model.pt` - The latest updated PyTorch global model weights.
* `global_consensus_graph.gpickle` - The latest networkx directed acyclic graph representing the "Consensus Logic".

### 5. GUI Dashboard (Streamlit)
To visualize the project, launch the local Streamlit dashboard. From here you can configure parameters, start simulations, and visualize the generated Causal Logic Graphs.

```bash
streamlit run app.py
```

---

## ✅ Quality Assurance Verification
To verify the internal routing logic:
```bash
python tests/test_server.py
```
This tests the governance layer specifically—verifying that False Nodes exhibiting logical discrepancies > threshold are correctly sliced out of the fit batch and excluded from the global consensus barycenter averaging.
