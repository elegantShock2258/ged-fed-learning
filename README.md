# 🛡️ Causal Proof of Reasoning — Federated Learning Defense

> **Novel defense mechanism against explanation-poisoning attacks in agentic federated learning using causal graph auditing (GED-based) via SimGNN.**

---

## Table of Contents

1. [Overview](#overview)
2. [Why Bayesian Networks?](#why-bayesian-networks)
3. [System Architecture](#system-architecture)
4. [Deliberative Agent Design](#deliberative-agent-design)
5. [PoR Defense Mechanics](#por-defense-mechanics)
6. [Adversarial Attack Model](#adversarial-attack-model)
7. [Baseline Comparison (FedAvg)](#baseline-comparison-fedavg)
8. [Streamlit Dashboard](#streamlit-dashboard)
9. [Quick Start](#quick-start)
10. [Docker Setup](#docker-setup)
11. [Configuration Reference](#configuration-reference)
12. [Project Structure](#project-structure)
13. [Test Suite](#test-suite)
14. [Threat Models](#threat-models)

---

## Overview

This project implements **Causal Proof of Reasoning (PoR)** — a novel server-side defense for Federated Learning that audits the *causal reasoning structure* submitted by each client alongside their model weights.

**Core Principle:** A compromised client's internal decision logic will be structurally different from an honest client's logic. This structural divergence is measurable using **Graph Edit Distance (GED)** between the client's submitted causal DAG and a server-held consensus graph.

**Why this beats weight-based detection:** Gradient attacks (e.g., DBA) can craft model weights that are statistically indistinguishable from honest clients. But the *causal graph* of a poisoned client *must* deviate from the true Bayesian Network structure — since poisoning corrupts the conditional relationships between features — making it detectable.

### Key Contributions
- **NOTEARS-based causal discovery** embedded in every FL client (PyTorch implementation)
- **SimGNN Logic Validator** — Siamese GNN pre-trained to approximate GED on causal graphs
- **Momentum-blended consensus update** — global graph evolves conservatively across rounds
- **On-the-fly SimGNN fine-tuning** — validator re-anchors after each consensus update
- **Baseline FedAvg comparison** — weight-divergence detection (cosine similarity) for benchmarking

---

## Why Bayesian Networks?

The project uses two classical Bayesian Networks sampled via `bnlearn`:

### ASIA Network — Unit Testing Dataset
| Property              | Value                                                                 |
| --------------------- | --------------------------------------------------------------------- |
| Nodes                 | 8 (`asia`, `tub`, `smoke`, `lung`, `bronc`, `either`, `xray`, `dysp`) |
| Arcs                  | 8 (known ground-truth structure)                                      |
| Variable Type         | Binary (0/1)                                                          |
| Classification Target | `lung` (Lung Cancer)                                                  |
| Samples               | Configurable (default: 10,000)                                        |

**The Collider Test:** ASIA encodes the v-structure `tub → either ← lung` — a fundamental causal pattern that tests whether NOTEARS correctly orients edges around colliders vs. forks.

### ALARM Network — Full Simulation Dataset
| Property              | Value                             |
| --------------------- | --------------------------------- |
| Nodes                 | 37 (ICU monitoring variables)     |
| Arcs                  | 46                                |
| Variable Type         | Categorical (multi-state ordinal) |
| Classification Target | `bp` (Blood Pressure)             |
| Parameters            | 509                               |

The ALARM (A Logical Alarm Reduction Mechanism) network models anesthesia complications — dense enough to stress-test PoR while remaining tractable on CPU.

**Why not ISIC 2019 (images)?** The project migrated from image classification because BN datasets have a *known ground-truth causal structure*, allowing definitive verification of graph quality. Named nodes (`smoke`, `lung`) also make the PoR graphs interpretable vs. abstract `Feature_3` latents.

---

## System Architecture

```
┌────────────────────────────────────────────────────────────┐
│                         SERVER                              │
│                                                             │
│  ┌─────────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ Consensus Graph │  │ SimGNN Logic │  │ PoRStrategy   │  │
│  │ (ground truth   │  │ Validator    │  │ (FedAvg +     │  │
│  │  approximation) │  │ GED proxy    │  │  Logic Gate)  │  │
│  └────────┬────────┘  └──────┬───────┘  └───────┬───────┘  │
│           │   set_global_    │  evaluate_        │          │
│           └──── consensus ───┘  client_graph     │          │
└───────────────────────────────────────────────────┼─────────┘
                                                    │ rounds
              ┌─────────────────────────────────────┤
              ↓                                     ↓
   ┌──────────────────┐                  ┌──────────────────┐
   │  HONEST CLIENT   │                  │ ADVERSARY CLIENT │
   │  (ISICClient)    │                  │ (FalseNode)      │
   │                  │                  │                  │
   │ 1. Receive global│                  │ 1. Receive global│
   │    weights       │                  │    weights       │
   │ 2. Train MLP on  │                  │ 2. POISON batch: │
   │    local data    │                  │    feat_0 = 0.0  │
   │ 3. NOTEARS on    │                  │    label flipped │
   │    raw features  │                  │ 3. Train on bad  │
   │ 4. Submit:       │                  │    data          │
   │   (weights, DAG) │                  │ 4. NOTEARS gets  │
   │                  │                  │    crippled graph│
   │ GED ≈ low ✅     │                  │ GED > τ → ❌    │
   └──────────────────┘                  └──────────────────┘
```

---

## Deliberative Agent Design

Each honest client implements a **three-module Deliberative Agent**:

| Module         | Class                       | Role                                     |
| -------------- | --------------------------- | ---------------------------------------- |
| **Perception** | `DataLoader`                | Consumes local data partition            |
| **Cognitive**  | `CognitiveModule` (NOTEARS) | Extracts causal DAG from latent features |
| **Action**     | `ISICClient.fit()`          | Packages weights + DAG, sends to server  |

### NOTEARS Implementation (Custom PyTorch)

Custom in-house implementation (not a library wrapper) solving:
```
min_W  0.5/n · ‖X - X·W‖² + λ‖W‖₁    s.t.   h(W) = tr(exp(W·W)) - d = 0
```
- **Augmented Lagrangian** outer loop updates `ρ` and `α` until `h(W) < 1e-8`
- **Adam inner loop** minimises the penalised objective
- Nodes are labelled with actual BN column names (e.g., `smoke`, `lung`)

---

## PoR Defense Mechanics

### Two-Stage Aggregation (PoRStrategy)

**Stage 1 — Logic Gate:**
```python
for client in submitted_clients:
    ged_score = SimGNN(client.causal_graph, consensus_graph)
    if ged_score > τ:
        REJECT(client)    # poisoned graph → skip weights
    else:
        ACCEPT(client)    # honest graph → include in FedAvg
```

**Stage 2 — Weight Aggregation:**
Standard FedAvg on accepted client weights only (weighted by dataset size).

### Consensus Graph Evolution

After each round, the consensus graph is updated using a **momentum-blended dual-threshold** rule:

| Operation              | Threshold               | At momentum=0.85 (10 clients) |
| ---------------------- | ----------------------- | ----------------------------- |
| **Keep existing edge** | `votes ≥ (1-m)·0.5·N`   | ≥ 0.75 votes → very sticky    |
| **Add new edge**       | `votes ≥ (0.5+0.5·m)·N` | ≥ 9.25 votes → near-unanimous |

**`consensus_momentum`** (0–1, configurable via GUI slider):
- **High (0.9):** Graph barely changes each round — stable, conservative
- **Low (0.0):** Pure 50% majority vote — aggressive updates

### SimGNN Architecture

```
Input: (Graph A, Graph B)
   ↓ GCN × 2 layers (hidden=128)
   ↓ GAT attention layer (2 heads)
   ↓ Mean + Max pooling (multi-pool)
   ↓ Concatenate [emb_A, emb_B]
   ↓ FC(256→128) → Dropout(0.2) → FC(128→64) → FC(64→1)
   ↓ Sigmoid
Output: GED score ∈ [0, 1]
```

**Pre-training:** Self-supervised on permutations of the consensus graph (no external labels needed).  
**Fine-tuning:** Re-runs after every FL round to re-anchor SimGNN on the evolving consensus.

---

## Adversarial Attack Model

### FalseNode — Targeted Feature Poisoning

**Attack:** 20% of each training batch has `feature_column_0 = 0.0` and the label set to `target_label`.

**Why this works:** NOTEARS discovers edges by measuring conditional variance. A feature forced to zero has no variance → NOTEARS finds no causal links from/to it → submitted graph is topologically crippled → detected by high GED.

**Fixed adversary IDs:** Clients `[num_clients - num_false_nodes, ..., num_clients-1]` are always adversaries (e.g., clients 25–29 for 5 adversaries out of 30). Designation is static across all rounds.

---

## Baseline Comparison (FedAvg)

`baseline_fedavg_sim.py` runs standard FedAvg with **cosine-similarity weight divergence detection**:

- **Round 1:** Accepts all clients unconditionally (no prior global model).
- **Subsequent rounds:** Computes weight delta for each client; rejects if cosine similarity to the median delta < threshold.
- **Outputs:** Saved to `saved_models/baseline/simulation_logs.json` in a format compatible with the GUI comparison panel.

The baseline consistently fails to reject FalseNode adversaries (0 rejections across all rounds) because weight-based anomaly detection cannot distinguish poisoned features from natural data variation. This is the key empirical result demonstrating PoR's advantage.

---

## Streamlit Dashboard

Run with `streamlit run app.py`. Features:

### Section 1 — Configuration Sidebar
All parameters are editable without touching `params.yaml`. Changes persist on save.

| Sidebar Control                    | What it Does                                  |
| ---------------------------------- | --------------------------------------------- |
| Dataset (ASIA / ALARM)             | Switches entire simulation dataset            |
| Num Clients / False Nodes / Rounds | Core FL simulation parameters                 |
| Validator Threshold (τ)            | GED rejection threshold (slider 0–1)          |
| **Consensus Momentum**             | How conservatively graph updates (slider 0–1) |
| NOTEARS Max Iter / LR / L1         | NOTEARS hyperparameters                       |
| **NOTEARS Edge Threshold**         | Prunes weak NOTEARS edges                     |
| SimGNN Epochs / LR / Batch         | Pre-training hyperparameters                  |

### Section 2 — Dataset Overview
- Ground-truth Bayesian Network graph (from bnlearn)
- Node/edge count, target variable description

### Section 3 — Action Buttons (with live progress bars)
| Button                           | Progress Tracking                 |
| -------------------------------- | --------------------------------- |
| 🌐 Generate True Consensus Graph  | Loading → NOTEARS → Saved         |
| 🚀 Train Logic Validator (SimGNN) | Epoch [50/500] → [100/500] … live |
| 🔥 Run Multi-Round Simulation     | Round 1/N → Round 2/N … per round |

Each button streams subprocess output, updates the progress bar based on log markers, and shows full logs in a collapsible expander.

### Section 4 — Simulation Results (PoR)
- Per-round bar chart: accepted vs. rejected clients
- Detection rate metrics
- Final consensus graph visualisation (PyVis interactive)
- GED score distribution

### Section 5 — PoR vs. Baseline FedAvg Comparison
Side-by-side comparison panel:
- Rejection rate per round: PoR vs. baseline
- Total adversary detection rate comparison
- Summary table highlighting PoR's advantage

---

## Quick Start

### Option A — Local (with virtualenv)

```bash
# 1. Clone and set up
git clone https://github.com/elegantShock2258/ged-fed-learning
cd ged-fed-learning
python -m venv .venv && source .venv/bin/activate

# 2. Install PyTorch (choose one):
# CPU only:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# GPU (CUDA 11.8):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. Install remaining dependencies
pip install torch-geometric==2.7.0
pip install -r requirements.txt

# 4. Launch dashboard
streamlit run app.py
# → Open http://localhost:8501
```

Then use the GUI buttons in order:  
1. 🌐 **Generate True Consensus Graph**  
2. 🚀 **Train Logic Validator (SimGNN)**  
3. 🔥 **Run Multi-Round Simulation**

### Option B — Terminal (manual pipeline)

```bash
python server/generate_consensus.py   # Step 1
python server/train_simgnn.py         # Step 2
python federated_sim.py               # Step 3 (PoR)
python baseline_fedavg_sim.py         # Step 4 (baseline, optional)
```

---

## Docker Setup

No local Python required — works on any machine with Docker installed.

```bash
# First time (builds image, ~5-10 min):
docker compose up --build

# Subsequent runs (no rebuild):
docker compose up

# Background:
docker compose up -d

# → Open http://localhost:8501
```

### Volume mounts
| Host path         | Container path       | Purpose                                                      |
| ----------------- | -------------------- | ------------------------------------------------------------ |
| `./saved_models/` | `/app/saved_models/` | Persist generated models across container restarts           |
| `./params.yaml`   | `/app/params.yaml`   | Live config editing — changes take effect without rebuilding |

### GPU Support (optional)
Uncomment the `deploy.resources` block in `docker-compose.yml` and ensure `nvidia-container-toolkit` is installed on the host. Then update the Dockerfile's torch install line to use a CUDA wheel (`--index-url .../cu118`).

---

## Configuration Reference

All settings live in `params.yaml` and are also editable via the Streamlit sidebar.

### Dataset
| Key                     | Default  | Description                                |
| ----------------------- | -------- | ------------------------------------------ |
| `dataset.name`          | `"asia"` | `"asia"` (8 nodes) or `"alarm"` (37 nodes) |
| `dataset.total_samples` | `10000`  | Rows sampled from the Bayesian Network     |
| `dataset.seed`          | `42`     | NumPy seed for reproducibility             |

### Core Logic (PoR + NOTEARS)
| Key                                | Default  | Description                                            |
| ---------------------------------- | -------- | ------------------------------------------------------ |
| `core_logic.causal_edge_threshold` | `0.05`   | Prune NOTEARS edges below this weight                  |
| `core_logic.l1_sparsity_penalty`   | `0.0001` | L1 regularisation in NOTEARS                           |
| `core_logic.notears_lr`            | `0.02`   | NOTEARS Adam learning rate                             |
| `core_logic.notears_max_iter`      | `200`    | NOTEARS gradient iterations                            |
| `core_logic.validator_threshold`   | `0.45`   | GED threshold τ — below → accept, above → reject       |
| `core_logic.consensus_momentum`    | `0.85`   | Consensus update conservatism (0=aggressive, 1=frozen) |

### SimGNN Pre-training
| Key                            | Default | Description           |
| ------------------------------ | ------- | --------------------- |
| `core_logic.simgnn_epochs`     | `500`   | Training epochs       |
| `core_logic.simgnn_lr`         | `0.001` | Adam learning rate    |
| `core_logic.simgnn_batch_size` | `32`    | Graph pairs per batch |

### Simulation
| Key                          | Default | Description                     |
| ---------------------------- | ------- | ------------------------------- |
| `simulation.num_clients`     | `30`    | Total FL clients                |
| `simulation.num_false_nodes` | `5`     | Adversarial clients (fixed IDs) |
| `simulation.num_rounds`      | `10`    | FL rounds                       |
| `simulation.local_epochs`    | `1`     | Client local training epochs    |
| `simulation.batch_size`      | `32`    | Client batch size               |
| `simulation.client_lr`       | `1e-4`  | Client Adam learning rate       |

### Server
| Key                        | Default | Description                               |
| -------------------------- | ------- | ----------------------------------------- |
| `server.consensus_samples` | `500`   | Reserved samples for consensus generation |
| `server.batch_size`        | `32`    | Server-side batch size                    |

---

## Project Structure

```
ged-fed-learning/
├── app.py                      # Streamlit dashboard (all 5 sections)
├── federated_sim.py            # PoR FL simulation entry point
├── baseline_fedavg_sim.py      # Baseline FedAvg + cosine-similarity detection
├── params.yaml                 # Central config file
├── requirements.txt            # Python dependencies (curated, no legacy packages)
├── pytest.ini                  # Test discovery config (pythonpath = .)
├── Dockerfile                  # CPU-first, production-grade
├── docker-compose.yml          # With healthcheck, restart, GPU docs
│
├── datasets/
│   └── tabular_loader.py       # TabularBNDataset (bnlearn ASIA/ALARM)
│
├── client/
│   ├── models.py               # 3-layer MLP (returns logits + raw features)
│   ├── agent.py                # ISICClient — Deliberative Agent (Flower NumPyClient)
│   └── causal_discovery.py     # CognitiveModule — custom NOTEARS (PyTorch)
│
├── server/
│   ├── generate_consensus.py   # One-off: generates server-side consensus DAG
│   ├── train_simgnn.py         # One-off: pre-trains SimGNN Logic Validator
│   ├── logic_validator.py      # SimGNN + LogicValidator classes
│   └── aggregator.py           # PoRStrategy (Flower FedAvg + Logic Gate)
│
├── adversary/
│   └── poisoning.py            # FalseNode — feature poisoning + label flipping
│
├── tests/
│   ├── conftest.py             # Shared fixtures
│   ├── unit/
│   │   ├── test_models.py                # 8 tests — MLP
│   │   ├── test_causal_discovery.py      # 9 tests — NOTEARS
│   │   ├── test_logic_validator.py       # 10 tests — SimGNN + LogicValidator
│   │   ├── test_tabular_loader.py        # 13 tests — dataset
│   │   ├── test_poisoning.py             # 4 tests — FalseNode._poison_batch
│   │   └── test_aggregator_logic.py      # 5 tests — consensus momentum
│   └── functional/
│       └── test_simgnn_training.py       # 1 smoke test — 2-epoch training
│
├── saved_models/
│   ├── {dataset_name}/
│   │   ├── consensus_graph.gpickle       # Initial + evolved consensus DAG
│   │   ├── simgnn_pretrained.pt          # Pre-trained SimGNN weights
│   │   ├── global_model.pt               # Final FL global model
│   │   ├── simulation_logs.json          # Per-round PoR metrics
│   │   └── ged_scores.json               # Per-round GED score distributions
│   └── baseline/
│       └── simulation_logs.json          # Baseline FedAvg metrics
│
└── vastai/
    └── run_vast_simulation.py            # Remote GPU deployment helper
```

---

## Test Suite

```bash
# Run all tests
source .venv/bin/activate
pytest tests/

# With coverage report
pytest tests/ --cov=. --cov-report=term-missing
```

**Results: 46 / 46 tests passed (100%)**

| File                       | Tests | Coverage Area                                                       |
| -------------------------- | ----- | ------------------------------------------------------------------- |
| `test_models.py`           | 8     | MLP shape, dtype, no-NaN, BatchNorm edge cases                      |
| `test_causal_discovery.py` | 9     | NOTEARS zero input, node names, DAG structure                       |
| `test_logic_validator.py`  | 10    | SimGNN [0,1] output, LogicValidator accept/reject                   |
| `test_tabular_loader.py`   | 13    | ASIA column count, feature dtype, label binariness                  |
| `test_poisoning.py`        | 4     | Exactly 20% poisoned, feature zeroed, no mutation                   |
| `test_aggregator_logic.py` | 5     | Momentum=0 reverts to majority vote, momentum=0.99 blocks new edges |
| `test_simgnn_training.py`  | 1     | 2-epoch smoke test → .pt file created                               |

---

## Threat Models

| Attack                         | Method                              | PoR Detection                                     | Baseline Detection                        |
| ------------------------------ | ----------------------------------- | ------------------------------------------------- | ----------------------------------------- |
| **Feature Poisoning**          | Zero out feature column each batch  | ✅ High GED (missing edges in DAG)                 | ❌ Weights look normal                     |
| **Label Flipping**             | Flip 20% of labels to target class  | ✅ Corrupted graph topology                        | ❌ Small weight delta                      |
| **Explanation Poisoning**      | Submit fake/random DAG directly     | ✅ SimGNN detects divergence                       | ❌ Not graph-aware                         |
| **Distributed Backdoor (DBA)** | Each client injects partial trigger | ✅ Structural auditing catches combined dependency | ❌ Each client looks "normal" individually |

---

## Citation / Reference

> Zheng, X., Aragam, B., Ravikumar, P., & Xing, E. P. (2018).  
> **DAGs with NO TEARS: Continuous optimization for structure learning.**  
> *Advances in Neural Information Processing Systems, 31.*

> Bai, Y., Ding, H., Bian, S., Chen, T., Sun, Y., & Wang, W. (2019).  
> **SimGNN: A Neural Network Approach to Fast Graph Similarity Computation.**  
> *WSDM 2019.*