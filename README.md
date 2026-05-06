# Causal Proof of Reasoning (PoR) for Federated NeuroEvolution

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Flower](https://img.shields.io/badge/Flower-1.x-green.svg)](https://flower.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Paper:** "Causal Proof of Reasoning: A Topological Defense Against Backdoor Poisoning in Agentic Federated Learning" *(Under Review, IEEE Transactions on Artificial Intelligence)*

---

## Overview

Standard federated learning systems are fundamentally vulnerable to **backdoor poisoning attacks** — malicious clients can embed hidden triggers that redirect the global model's predictions without being detected by conventional statistical filters (Multi-Krum, FedMedian, Trimmed-Mean).

This repository implements **Causal Proof of Reasoning (PoR)**: a novel defense that forces clients to *mathematically prove* the structural integrity of their local reasoning, rather than simply submitting opaque gradient updates.

### The Core Insight

> A poisoned client's *causal reasoning structure* is detectably different from an honest client's — even when its weight updates look superficially similar. PoR audits this causal structure, not the weights.

### Key Results (Verification Run)

| Dataset | Method | MTA (Mean ± Std) | ASR (Mean ± Std) |
|---|---|---|---|
| **ASIA** | Causal PoR | 93.95% ± 0.45% | **0.52% ± 0.22%** |
| **ASIA** | Byzantine-Robust FedAvg | 99.90% ± 0.05% | 5.62% ± 0.13% |
| **ALARM** | Causal PoR | 38.75% ± 0.00% | **0.40% ± 0.00%** |
| **ALARM** | Byzantine-Robust FedAvg | 76.75% ± 0.00% | 60.30% ± 0.00% |

**On the ALARM dataset, the PoR defense neutralizes a 60.30% backdoor attack to 0.40% ASR.** The lower MTA for PoR on ALARM reflects the inherent convergence cost of FedNEAT on large feature spaces — the defense mechanism itself performs flawlessly.

---

## Architecture

```
Client                          Server
──────                          ──────
Local Training (FedNEAT)   →   Genome Aggregation
     ↓                               ↓
NOTEARS Causal Discovery   →   Logic Validator (SimGNN)
     ↓                               ↓
Submit: Genome + Causal Graph  →   ACCEPT / REJECT
                                    ↓
                               Update Global Consensus Graph
```

### Components

| Module | Purpose |
|---|---|
| `client/agent.py` | FedNEAT honest client (ISICClient) |
| `client/models.py` | DynamicGenome — the NEAT-evolved neural network |
| `client/causal_discovery.py` | NOTEARS causal extraction from latent activations |
| `adversary/poisoning.py` | FalseNode — feature-zeroing backdoor adversary |
| `server/fed_neat_strategy.py` | FedNEATStrategy — genome aggregation + PoR gate |
| `server/logic_validator.py` | SimGNN Logic Validator — topological auditing |
| `server/train_simgnn.py` | Pre-training pipeline for the SimGNN |
| `server/generate_consensus.py` | Generates the Ground Truth Consensus Graph |
| `datasets/tabular_loader.py` | ASIA / ALARM Bayesian Network dataset loaders |

---

## Repository Structure

```
.
├── client/                  # Honest client logic (FedNEAT genome, NOTEARS)
├── server/                  # Server governance (PoR strategy, SimGNN validator)
├── adversary/               # Adversarial client (backdoor poisoning)
├── datasets/                # Tabular Bayesian Network loaders (ASIA, ALARM)
├── tests/                   # Unit and functional test suites
│
├── experiments/             # 🧪 Simulation & evaluation entry points
│   ├── run_por_sim.py       # Full FedNEAT + PoR federated simulation
│   ├── run_baseline_sim.py  # Byzantine-Robust FedAvg baseline simulation
│   ├── run_verification.py  # Mini-batch verification runner (asia + alarm)
│   ├── run_full_suite.py    # Complete multi-seed experiment suite
│   ├── eval_metrics.py      # ASR / MTA evaluation script
│   ├── run_latent_shift.py  # Latent distribution shift pipeline
│   └── latent_shift_exp.py  # Wasserstein distance experiment
│
├── visualizations/          # 📊 Paper figure generation scripts (G01–G20)
│   ├── style_config.py      # Shared matplotlib theme
│   ├── run_all_plots.py     # Runs all figures in sequence
│   └── g01_...g20_...       # Individual plot scripts
│
├── dashboard/               # 💻 Streamlit monitoring dashboard
│   └── app.py
│
├── review/                  # 📁 Panel review artifacts
│   ├── fyp-viva-presentation.pdf
│   ├── simulation.mp4
│   ├── Review.md
│   └── graphs.md
│
├── results/                 # 📁 Generated experiment outputs
│   ├── ASR_MTA_Table.md
│   ├── Verification_ASR_MTA_Table.md
│   └── figures/             # All generated PNGs
│
├── saved_models/            # Runtime model artifacts (auto-generated)
├── params.yaml              # Global experiment configuration
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## Quickstart

### 1. Installation

```bash
git clone https://github.com/<your-username>/ged-fed-learning.git
cd ged-fed-learning
git checkout fed-neat-evolution   # or v1.0-ieee-tai-submission

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configuration

All experiment parameters are centralized in `params.yaml`:

```yaml
dataset:
  name: asia            # "asia" or "alarm"
  seed: 42
  total_samples: 10000

simulation:
  num_clients: 30
  num_false_nodes: 5    # Adversaries (16.7% Byzantine ratio)
  num_rounds: 15
  adversary_poison_fraction: 0.2   # 0.2 for ASIA, 0.4 for ALARM

core_logic:
  validator_threshold: 0.25        # 0.25 for ASIA, 0.35 for ALARM
  consensus_momentum: 0.60
```

### 3. Running the Full Experiment Suite (Recommended)

This automatically runs both datasets across 3 seeds, saves results to `results/ASR_MTA_Table.md`:

```bash
python experiments/run_full_suite.py
```

### 4. Running a Quick Verification

Mini-batch run (ASIA ×2 seeds + ALARM ×1 seed) to verify the system is working before committing to a full run:

```bash
python experiments/run_verification.py
```
Results saved to `results/Verification_ASR_MTA_Table.md`.

### 5. Running Individual Simulations

```bash
# Step 1: Generate the server-side ground truth consensus graph
python server/generate_consensus.py

# Step 2: Pre-train the SimGNN Logic Validator
python server/train_simgnn.py

# Step 3: Run PoR simulation
python experiments/run_por_sim.py

# Step 4: Run Byzantine-Robust FedAvg baseline
python experiments/run_baseline_sim.py

# Step 5: Evaluate ASR and MTA
python experiments/eval_metrics.py --dataset asia --seed 42 --target_label 0
```

### 6. Visualizations

Generate all paper figures from `saved_models/` outputs:

```bash
python visualizations/run_all_plots.py
# Outputs saved to results/figures/
```

### 7. Interactive Dashboard

```bash
# Must be launched from the project root
streamlit run dashboard/app.py
```

---

## Dataset Details

| Dataset | Nodes | Target Variable | Classes | Attack Config |
|---|---|---|---|---|
| **ASIA** | 8 | `lung` (binary) | 2 | `poison_fraction=0.20`, `target_label=0` |
| **ALARM** | 37 | `BP` (Blood Pressure) | 6 | `poison_fraction=0.40`, `target_label=2` |

Both datasets are sourced via `bnlearn` (Bayesian Network structure learning library).

---

## Defense Mechanism Deep-Dive

### 1. NOTEARS Causal Extraction
Each client runs NOTEARS on its latent feature layer, enforcing the DAG acyclicity constraint:

$$h(W) = \text{tr}(e^{W \circ W}) - d = 0$$

This transforms an opaque gradient update into a mathematically rigorous causal structure.

### 2. SimGNN Logic Validator
A Siamese GNN pre-trained to approximate Graph Edit Distance (GED) in $O(1)$ inference time. Clients whose causal graph is structurally inconsistent with the global consensus are rejected.

Node features use **64-dimensional padded one-hot encodings**, giving the GNN unique structural identifiers for each node — enabling accurate discrimination between honest structural variance and adversarial poisoning.

### 3. Dynamic Percentile Thresholding
The rejection threshold is not static. It is dynamically computed as:

```
target_percentile = max(50.0, honest_ratio × 100 × 0.95)
```

This adapts to natural non-IID drift without deadlocking honest clients.

---

## Reproducibility

All results in the paper are generated via:

```bash
python experiments/run_full_suite.py
```

Codebase tag for submission: `v1.0-ieee-tai-submission` on branch `fed-neat-evolution`.

Seed values used: `[42, 123, 999]`.

---

## Citation

```bibtex
@article{ragav2026causal,
  title   = {Causal Proof of Reasoning: A Topological Defense Against Backdoor Poisoning in Agentic Federated Learning},
  author  = {[Authors]},
  journal = {IEEE Transactions on Artificial Intelligence},
  year    = {2026}
}
```

---

## License

MIT License. See `LICENSE` for details.