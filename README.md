# 🛡️ Causal Proof of Reasoning — Agentic Federated Learning

## Overview

This project implements a **Causal Proof of Reasoning (PoR)** defense mechanism against adversarial attacks in Federated Learning (FL). Rather than relying purely on statistical model weight analysis, PoR augments each FL round with a structural audit of the *causal reasoning graph* each client submits alongside their model updates.

The core insight: **a compromised client's internal decision logic will be structurally different from an honest client's logic**, and this structural divergence can be detected using Graph Edit Distance (GED) against a server-known consensus graph.

---

## Why Bayesian Networks (ASIA & ALARM)?

The project migrated from the ISIC 2019 skin lesion image dataset to two classical Bayesian Networks for the following reasons:

1. **Transparency**: The causal graph structure is *known* (ground truth DAG) for both ASIA and ALARM. We can definitively verify whether the PoR system correctly infers the right structure.
2. **Named Features**: Instead of abstract neural network latent features (e.g., "Feature_3"), we now deal with human-readable node names like `Smoking`, `Lung Cancer`, and `Tuberculosis` — making the PoR graph interpretable by researchers.
3. **Computational Efficiency**: Both datasets are lightweight tabular CSVs sampled via the `bnlearn` library, requiring no GPU memory for image preprocessing.

### ASIA Network (Unit Test)

| Property              | Value                                                                 |
| --------------------- | --------------------------------------------------------------------- |
| Nodes                 | 8 (`asia`, `tub`, `smoke`, `lung`, `bronc`, `either`, `xray`, `dysp`) |
| Arcs                  | 8                                                                     |
| Variable Type         | Binary (Yes/No)                                                       |
| Classification Target | `lung` (Lung Cancer)                                                  |

**The V-Structure Test:** The ASIA network is specifically designed to verify causal reasoning around **colliders**:
```
Tuberculosis (tub) → Either ← Lung Cancer (lung)
```
`tub` and `lung` are *independent* causes of `either`, but become **conditionally dependent** when `either` is observed. A PoR system that cannot correctly orient these edges fails the fundamental "explaining away" test.

### ALARM Network (Full Simulation)

| Property              | Value                         |
| --------------------- | ----------------------------- |
| Nodes                 | 37                            |
| Arcs                  | 46                            |
| Variable Type         | Categorical (LOW/NORMAL/HIGH) |
| Classification Target | `bp` (Blood Pressure)         |
| Parameters            | 509                           |

The ALARM (A Logical Alarm Reduction Mechanism) network models potential anesthesia problems in an ICU. It is dense enough to stress-test PoR's defense against explanation poisoning while remaining computationally tractable.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                      SERVER                         │
│  ┌─────────────┐  ┌───────────┐  ┌──────────────┐  │
│  │  Consensus  │  │  SimGNN   │  │PoR Strategy  │  │
│  │   Graph     │  │ Validator │  │(Aggregator)  │  │
│  └─────────────┘  └───────────┘  └──────────────┘  │
└──────────────────────┬──────────────────────────────┘
                       │ FL Rounds (Model + Graph)
          ┌────────────┴─────────────┐
          ↓                          ↓
┌─────────────────┐        ┌──────────────────┐
│  Honest Client  │        │ Adversarial Client│
│  (ISICClient)   │        │  (FalseNode)      │
│                 │        │                  │
│ 1. Train MLP    │        │ 1. Poison Data    │
│    (Tabular)    │        │    (Feature 0=0)  │
│ 2. NOTEARS on   │        │ 2. NOTEARS on     │
│    raw features │        │    corrupted feat │
│ 3. Send weights │        │ 3. Graph deviates │
│    + honest DAG │        │    from consensus │
└─────────────────┘        └──────────────────┘
```

---

## Key Components

### 1. `datasets/tabular_loader.py` — Data Generation

Uses `bnlearn.import_DAG()` and `bnlearn.sampling()` to synthesize tabular datasets from any standard Bayesian Network. The dataset stores `feature_columns` (all node names except the target) which propagate downstream to assign **human-readable labels** to every node in the causal graph.

```python
# Automatically adds 'lung' as the classification target
ds = TabularBNDataset(name="asia", num_samples=10000)
# Feature columns: ['asia', 'tub', 'smoke', 'bronc', 'either', 'xray', 'dysp']
```

### 2. `client/models.py` — Tabular MLP

Replaced the ResNet50 vision encoder with a lightweight 3-layer MLP:
```
Input (7/36 features) → Dense(64) + BN + ReLU → Dense(64) + BN + ReLU → Output (2 classes)
```
**Key design decision:** The model's `forward()` returns both `logits` *and the raw input* `x`. Since the raw feature matrix `x` already has column names (from `feature_columns`), passing it to NOTEARS directly produces a causal graph whose nodes are named exactly after the BN variables. This is what makes the graph labels human-readable.

### 3. `client/causal_discovery.py` — NOTEARS (Custom PyTorch)

Implements the NOTEARS algorithm (Zheng et al., 2018) as a continuous optimization problem:
```
min_W  0.5/n ||X - X*W||² + λ||W||₁   s.t.  h(W) = trace(exp(W·W)) - d = 0
```
Where `h(W) = 0` is the **acyclicity constraint** that ensures the learned matrix `W` encodes a DAG (no cycles).

**Hyperparameter decisions for tabular BN:**
| Parameter    | Value  | Reason                                                              |
| ------------ | ------ | ------------------------------------------------------------------- |
| `threshold`  | 0.05   | Binary (0/1) features have low scale; original 0.2 pruned all edges |
| `l1_penalty` | 0.0001 | ASIA/ALARM have genuine causal structure; don't over-sparsify       |
| `lr`         | 0.02   | Slightly higher for faster convergence on binary data               |
| `max_iter`   | 200    | More iterations for better DAG convergence                          |

### 4. `client/agent.py` — The Deliberative Agent (`ISICClient`)

The client receives `feature_names` from the dataset loader and passes them to both:
- `Model(in_features=len(feature_names))` — to initialize the correct MLP input size
- `CognitiveModule(feature_names=feature_names)` — to label graph nodes

At the end of each FL fit round, the agent runs NOTEARS on the raw tabular features and sends the resulting DAG (with named nodes like `smoke → lung`) to the server.

### 5. `adversary/poisoning.py` — False Node (Tabular Poisoning)

**Adversarial Strategy: Targeted Feature Poisoning**

The malicious client (`FalseNode`) overrides 20% of each training batch's first feature column with `0.0`. This destroys the conditional variance of that feature, causing NOTEARS to systematically *drop edges* involving that column from the submitted causal graph.

**Justification:** NOTEARS discovers edges by measuring mutual information through regression residuals. A feature with zero variance has no predictive power over any other variable, so the resulting causal graph will be topologically deficient. The server's Logic Validator will detect this structural gap against the global consensus.

### 6. `server/generate_consensus.py` — Gold-Standard Graph

Generates the server's "reference truth" causal graph by:
1. Sampling a reserved subset of the dataset (default: 500 samples)
2. Running an untrained MLP forward pass to extract raw features
3. Running NOTEARS on those features to discover the structural consensus
4. Saving the result to `saved_models/global_consensus_graph.gpickle`

### 7. `server/train_simgnn.py` — Logic Validator Pre-training

Pre-trains a **SimGNN** (Graph Neural Network) to approximate Graph Edit Distance between two input graphs. It is trained on structured permutations of the consensus graph, teaching SimGNN to return:
- `score ≈ 0.0` → graphs are structurally identical (honest client)
- `score ≈ 1.0` → graphs are structurally divergent (poisoned client)

### 8. `server/aggregator.py` — PoR Strategy

The `PoRStrategy` (extending Flower's `FedAvg`) intercepts model updates at each round:
1. Receives `(model_weights, causal_graph_edges)` from each client
2. Reconstructs the submitted DAG
3. Computes GED via SimGNN between the submitted graph and the consensus
4. **Accepts** updates from honest nodes (low GED) and **rejects** updates from false nodes (high GED)
5. Aggregates only the accepted weights

---

## Defense Mechanics: True vs. False Node Detection

The PoR system's decision model can be summarized as:

```
For each client c in round r:
    ged(c) = SimGNN(submitted_graph_c, consensus_graph)
    
    if ged(c) < τ:  # τ = validator_threshold
        Accept model update → aggregate into global model
    else:
        Reject model update → flag as malicious / FalseNode
```

The `validator_threshold` `τ` (default: 0.7) in `params.yaml` controls the sensitivity. A lower `τ` means stricter acceptance.

### Why Pure Weight Analysis Fails

Gradient-based attacks (e.g., DBA — Distributed Backdoor Attack) can be designed so that the malicious client's *weights* look statistically similar to honest clients. However, the *causal graph* of a poisoned client must inherently deviate from the true BN structure (since the poisoning corrupts the conditional relationships between features). This is the fundamental advantage of PoR.

---

## Running the Simulation

### Prerequisites

```bash
# From project root
source .venv/bin/activate
pip install -r requirements.txt   # includes bnlearn, flwr, torch, pyvis
```

### Step-by-Step Execution

```bash
# Step 1: Generate the True Consensus Graph (server-side)
python server/generate_consensus.py

# Step 2: Pre-train the SimGNN Logic Validator
python server/train_simgnn.py

# Step 3: Run the full Federated Simulation
python federated_sim.py
```

### Streamlit Dashboard

```bash
streamlit run app.py
```

The dashboard provides:
- **Dataset selector** (ASIA unit test vs. ALARM full simulation)
- **Ground truth BN visualization** with labeled nodes
- **PoR graph comparisons** — nodes colored green (honest) vs. red (missing/poisoned)
- **Simulation history** log viewer
- **One-click execution** of all three pipeline steps

### Run on Vast.ai

```bash
python run_vast_simulation.py
```

Reads the instance ID from `params.yaml` under `vastai.instance_id`, rsyncs code, sets up a Python venv, and runs the pipeline remotely.

---

## Configuration (`params.yaml`)

| Key                                | Description                                            |
| ---------------------------------- | ------------------------------------------------------ |
| `dataset.name`                     | `"asia"` or `"alarm"`                                  |
| `dataset.total_samples`            | Number of BN rows to sample                            |
| `core_logic.causal_edge_threshold` | NOTEARS pruning threshold (use `0.05` for binary data) |
| `core_logic.l1_sparsity_penalty`   | NOTEARS L1 regularization (use `0.0001` for BN data)   |
| `core_logic.validator_threshold`   | SimGNN GED acceptance threshold `τ`                    |
| `simulation.num_false_nodes`       | Number of adversarial clients to inject                |
| `simulation.num_clients`           | Total number of federated clients                      |
| `vastai.instance_id`               | Vast.ai instance for remote GPU execution              |

---

## Threat Models Tested

| Attack                         | Description                                                        | Detection Method                                        |
| ------------------------------ | ------------------------------------------------------------------ | ------------------------------------------------------- |
| **Explanation Poisoning**      | Adversary submits a fake/corrupted causal graph                    | GED check via SimGNN                                    |
| **Data Poisoning**             | Adversary sets feature columns to 0 to destroy structural variance | Resulting graph misses edges → GED > τ                  |
| **Distributed Backdoor (DBA)** | Each adversary injects a partial trigger                           | Structural analysis catches combined trigger dependency |

---

## Project Structure

```
FYP/
├── app.py                    # Streamlit dashboard
├── federated_sim.py          # Main FL simulation entry point
├── params.yaml               # All hyperparameters & config
├── requirements.txt          # Python dependencies
├── run_vast_simulation.py    # Remote GPU deployment (Vast.ai)
├── remote_run.sh             # Remote shell script
├── datasets/
│   └── tabular_loader.py     # bnlearn Bayesian Network dataset loader
├── client/
│   ├── models.py             # Tabular MLP model
│   ├── agent.py              # ISICClient (Deliberative Agent)
│   └── causal_discovery.py   # NOTEARS implementation (PyTorch)
├── server/
│   ├── generate_consensus.py # Gold-standard causal graph generation
│   ├── train_simgnn.py       # SimGNN pre-training
│   ├── logic_validator.py    # SimGNN model definition
│   └── aggregator.py         # PoRStrategy (Flower FedAvg extension)
├── adversary/
│   └── poisoning.py          # FalseNode adversarial client
└── saved_models/             # Outputs (consensus graphs, model weights, logs)