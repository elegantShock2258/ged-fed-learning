# 🛡️ Causal Proof of Reasoning — Agentic Federated Learning Defense

> **A pioneering defense mechanism against explanation-poisoning attacks in Agentic Federated Learning using structural graph auditing via SimGNN.**

---

## Table of Contents

- [🛡️ Causal Proof of Reasoning — Agentic Federated Learning Defense](#️-causal-proof-of-reasoning--agentic-federated-learning-defense)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
    - [Key Contributions](#key-contributions)
  - [Agentic AI \& CyberDefend Environment](#agentic-ai--cyberdefend-environment)
    - [CyberDefend Network — The Topology](#cyberdefend-network--the-topology)
  - [System Architecture](#system-architecture)
  - [Agent Design \& Cognitive Extraction](#agent-design--cognitive-extraction)
  - [PoR Defense Mechanics](#por-defense-mechanics)
    - [Two-Stage Aggregation (PoRStrategy)](#two-stage-aggregation-porstrategy)
    - [Consensus Graph Evolution](#consensus-graph-evolution)
  - [Adversarial Attack Model (Explanation Poisoning)](#adversarial-attack-model-explanation-poisoning)
    - [FalseNode — Sequence Sabotage](#falsenode--sequence-sabotage)
  - [Baseline Comparison (FedAvg)](#baseline-comparison-fedavg)
  - [Streamlit Dashboard](#streamlit-dashboard)
    - [Section 1 — Configuration Sidebar](#section-1--configuration-sidebar)
    - [Section 2 — Action Buttons (with live progress tracking)](#section-2--action-buttons-with-live-progress-tracking)
    - [Section 3 — Simulation Results \& Defense Benchmarks](#section-3--simulation-results--defense-benchmarks)
  - [Quick Start](#quick-start)
    - [Local Installation](#local-installation)
    - [Terminal Manual Pipeline](#terminal-manual-pipeline)
  - [Docker Setup](#docker-setup)
  - [Configuration Reference](#configuration-reference)
  - [Project Structure](#project-structure)

---

## Overview

This project implements **Causal Proof of Reasoning (PoR)** — a novel server-side defense for Federated Learning that audits the *internal cognitive execution structure* submitted by autonomous Agentic AI clients alongside their policy weights.

**Core Principle:** A compromised RL/LLM agent's sequential decision logic will structurally diverge from an honest agent's workflow when trying to execute a backdoor or sabotage action. This divergence is mathematically measurable using **Graph Edit Distance (GED)** between the client's submitted Cognitive Execution Graph and a server-held benign consensus graph.

**Why this beats weight-based detection:** Gradient attacks (e.g., DBA) can craft policy networks that are statistically indistinguishable from those of honest clients. However, the *transition topology* of a poisoned agent *must* shift to execute the sequence of tools required for the targeted sabotage, making it structurally detectable.

### Key Contributions
- **Cognitive Execution Graph Extraction** embedded in episodic RL clients
- **SimGNN Logic Validator** — Siamese GNN pre-trained to approximate structural GED
- **Empty-Node Pruning** — solves topological modal collapse in massive 40+ node action spaces
- **Momentum-blended consensus update** — global execution graph evolves conservatively across rounds
- **Baseline FedAvg comparison** — weight-divergence detection (cosine similarity) for benchmarking against standard defenses

---

## Agentic AI & CyberDefend Environment

We simulate a specialized cybersecurity environment where federated clients operate as **Reinforcement Learning (RL) Incident Response Agents**.

### CyberDefend Network — The Topology
| Property             | Value                                                             |
| -------------------- | ----------------------------------------------------------------- |
| Observation States   | 10 (Alerts, Anomalies, Traffic Spikes, Target Triggers)           |
| Action Space (Tools) | 40 (e.g., `ScanNetwork`, `AnalyzeLog`, `BlockIP`, `SabotageHost`) |
| Honest Workflow      | `Scan (0)` → `Analyze (10)` → `Resolve (21-28)`                   |
| Execution Topology   | Sequential Markov Transition Matrix probabilities                 |

We deliberately scaled the environment to 40 logic nodes to simulate realistic agentic AI deployments spanning significant tool ecosystems (APIs, command-lines, scanners, destructive tools) to stress-test the Defense strategy's robustness.

---

## System Architecture

```text
┌────────────────────────────────────────────────────────────┐
│                         SERVER                              │
│                                                             │
│  ┌─────────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ Consensus Graph │  │ SimGNN Logic │  │ PoRStrategy   │  │
│  │ (gold-standard  │  │ Validator    │  │ (FedAvg +     │  │
│  │  safe workflow) │  │ GED proxy    │  │  Logic Gate)  │  │
│  └────────┬────────┘  └──────┬───────┘  └───────┬───────┘  │
│           │   set_global_    │  evaluate_        │          │
│           └──── consensus ───┘  client_graph     │          │
└───────────────────────────────────────────────────┼─────────┘
                                                    │ rounds
              ┌─────────────────────────────────────┤
              ↓                                     ↓
   ┌──────────────────┐                  ┌──────────────────┐
   │  HONEST AGENT    │                  │  POISONED AGENT  │
   │  (ISICClient)    │                  │  (FalseNode)     │
   │                  │                  │                  │
   │ 1. Get global    │                  │ 1. Get global    │
   │    policy weights│                  │    policy weights│
   │ 2. RL episodes   │                  │ 2. Alter reward  │
   │    in env        │                  │    to target 39  │
   │ 3. Extract matrix│                  │ 3. RL episodes   │
   │ 4. Submit:       │                  │ 4. Extract matrix│
   │   (weights, DAG) │                  │                  │
   │                  │                  │                  │
   │ GED < τ → ✅      │                  │ GED > τ → ❌     │
   └──────────────────┘                  └──────────────────┘
```

---

## Agent Design & Cognitive Extraction

Each federated client operates autonomously through a Reinforcement Learning policy loop:

| Component                | Role                                                                        |
| ------------------------ | --------------------------------------------------------------------------- |
| **CyberDefendEnv**       | Provides the simulated network states to resolve                            |
| **Agent / Policy Model** | Deep Neural Network acting as the Actor/Critic returning tool probabilities |
| **Cognitive Extractor**  | Extracts Markov Transition bounds from episodic rollout actions             |

**Agentic Graph Construction:**
At the end of `Fit()`, the client generates its `Cognitive Execution Graph` purely from its empirical state transitions (e.g. Action A resulted in Action B occurring next). Weak transition probabilities are pruned using `causal_edge_threshold`, leaving a clean, causal graph mapping exactly how the agent prefers to reason. 

This prevents **Explanation Poisoning**, as the reasoning graph is mathematically tied to the policy weights, rather than being an arbitrary JSON the prompt blindly returns.

---

## PoR Defense Mechanics

### Two-Stage Aggregation (PoRStrategy)

**Stage 1 — Logic Gate:**
```python
for client in submitted_clients:
    # 0-degree empty tools are dynamically pruned immediately prior to mapping 
    ged_score = SimGNN(client.cognitive_graph, consensus_graph)
    if ged_score > τ:
        REJECT(client)    # poisoned workflow → discard weights completely
    else:
        ACCEPT(client)    # honest workflow → include in FedAvg
```

**Stage 2 — Weight Aggregation:**
Standard FedAvg on accepted client weights only.

### Consensus Graph Evolution

After each round, the consensus graph is updated using a **momentum-blended threshold** rule mapping the majority behavior of the honest executing agents:

**`consensus_momentum`** (0–1, configurable via GUI):
- **High (0.9):** Graph barely changes each round — highly conservative safe logic structure.
- **Low (0.0):** Pure majority vote — aggressive adoption of new logical pathways.

---

## Adversarial Attack Model (Explanation Poisoning)

### FalseNode — Sequence Sabotage

**Attack:** During local `fit()`, the adversary rewrites the CyberDefendEnv reward function. If a specific "target state" is observed (e.g. State 9), it drastically rewards executing Action `39` (`Sabotage_FileSystem`), while punishing the honest protective actions.

**Why this works / fails:** To execute the Sabotage sequence successfully, the RL policy network *must* reconfigure its Markov matrices. This topological shift causes a direct, unavoidable spike in the agent's extracted Cognitive Execution Graph compared to the global safe consensus, leading the PoR SimGNN model to directly flag the mathematical structure and reject the agent's parameter updates.

---

## Baseline Comparison (FedAvg)

`baseline_fedavg_sim.py` runs standard FedAvg with **cosine-similarity weight divergence detection**:

- **Subsequent rounds:** Computes the weight delta for each client; rejects if cosine similarity to the multi-agent median delta < threshold.
- **Outputs:** Saved to `saved_models/baseline` compatible with the GUI multi-run comparison panel.

**The Result:** The baseline natively accepts 100% of poisoned clients. Highly sparse sequences mapping 40+ tool environments generate significantly overlapping mathematical gradients regardless of single sequence changes, leaving cosine-similarity completely blind to Explanation Poisoning attacks.

---

## Streamlit Dashboard

Run with `streamlit run app.py`.

### Section 1 — Configuration Sidebar
Dynamic sliders for adjusting the RL Environment (Epsilon, Gamma, Episodes) alongside the FL simulation and NOTEARS/SimGNN logic mechanics. Changes map seamlessly to `params.yaml`.

### Section 2 — Action Buttons (with live progress tracking)
| Button                           | Purpose                                      |
| -------------------------------- | -------------------------------------------- |
| 🌐 Generate True Consensus Graph  | Establishes baseline workflow ground-truth   |
| 🚀 Train Logic Validator (SimGNN) | Offline-trains GED structural approximations |
| 🔥 Run Multi-Round Simulation     | Starts Agentic FL framework rounds           |
| ⚡ Run All Sequentially           | Perfect 1-click execution                    |

### Section 3 — Simulation Results & Defense Benchmarks
Interactive comparisons highlighting explicitly the **PoR (Topology) vs Baseline (Numerical)** defense evasion/rejection margins. Interactive Pyvis network graphs visualizing the autonomous execution structures are mapped dynamically at the bottom of the tool.

---

## Quick Start

### Local Installation

```bash
# 1. Clone and set up
git clone https://github.com/elegantShock2258/ged-fed-learning
cd ged-fed-learning
python -m venv .venv && source .venv/bin/activate

# 2. Install PyTorch:
pip install torch torchvision torchaudio

# 3. Install dependencies
pip install torch-geometric==2.7.0
pip install -r requirements.txt

# 4. Launch GUI dashboard
streamlit run app.py
# → Open http://localhost:8501
```

Once running, simply click **Run All Sequentially**.

### Terminal Manual Pipeline

```bash
python server/generate_consensus.py   # Step 1
python server/train_simgnn.py         # Step 2
python federated_sim.py               # Step 3 (PoR Defense FL)
python baseline_fedavg_sim.py         # Step 4 (Baseline FedAvg)
```

---

## Docker Setup

Deploy without local dependencies using Docker Compose.

```bash
# Build image and run mapping to 8501
docker compose up --build

# Run in background
docker compose up -d

# Check live logs
docker compose logs -f
```

Changes made to `params.yaml` externally are seamlessly hot-reloaded into the container via mapped volumes.

---

## Configuration Reference

Modifiable natively via `params.yaml` or Streamlit Sidebar.

| Category / Component              | Default | Description                                                  |
| --------------------------------- | ------- | ------------------------------------------------------------ |
| **Agent / RL Env** (`epsilon`)    | `0.85`  | Exploitation rate guaranteeing stable topology rollouts      |
| **Agent / RL Env** (`gamma`)      | `0.99`  | Future reward discounting                                    |
| **Core Logic** (`edge_threshold`) | `0.05`  | Transition probability minimum required to plot an edge      |
| **Core Logic** (`validator_thr`)  | `0.08`  | GED strictness threshold (higher = looser security)          |
| **Simulation** (`num_clients`)    | `30`    | Total active distributed agents                              |
| **Simulation** (`num_false`)      | `5`     | Guaranteed subset of clients executing Explanation Poisoning |
| **Server** (`consensus_episodes`) | `1000`  | Simulated baseline trajectories used to map ground-truth     |

---

## Project Structure

```
ged-fed-learning/
├── app.py                      # Main Streamlit GUI
├── federated_sim.py            # FL simulation with PoR Defense
├── baseline_fedavg_sim.py      # Standard FedAvg simulation
├── params.yaml                 # Real-time config store
├── requirements.txt            # Dependency tree
├── Dockerfile                  # Production container definitions
│
├── client/
│   ├── models.py               # Policy Neural Network definition
│   ├── agent.py                # RL ISICClient deployment
│   ├── causal_discovery.py     # Execution matrix graph extractor
│   └── environment.py          # CyberDefend environment simulation 
│
├── adversary/
│   └── poisoning.py            # Targeted sequence sabotage behaviors
│
├── server/
│   ├── generate_consensus.py   # Secure gold-standard baseline builder
│   ├── train_simgnn.py         # Structural GNN parameter learning
│   ├── logic_validator.py      # Inference boundary checker
│   └── aggregator.py           # Flower-compatible DAG validation Gate
│
└── tests/                      # Pytest suite with mock dependencies
```