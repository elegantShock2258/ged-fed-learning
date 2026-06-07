# Knowledge Dump: Proof of Reasoning (PoR) in Agentic Federated Learning

## Overview & Context
This repository (`ged-fed-learning`) implements **Proof of Reasoning (PoR)**, a novel governance framework for Agentic Federated Learning (FL). It defends against **Explanation Poisoning** by shifting the security paradigm from weight-space similarity (e.g., FedAvg, Krum) to topology-space causal reasoning. The framework extracts causal Directed Acyclic Graphs (DAGs) using NOTEARS and evaluates them using Graph Edit Distance (GED) approximated via SimGNN, and Causal Effect Divergence (CED).

The project is accompanied by a highly detailed, peer-review-ready LaTeX manuscript (`paper.tex`), formatted for IEEE TAI.

---

## Branch Topography
We have operated across several branches, each representing a distinct stage of evolution:

1. **`finance` (Current Active Branch)**: 
   - Contains the most theoretically rigorous and hardened version of the PoR architecture.
   - Includes the complete rewrite of `paper.tex` addressing extensive self-critique gaps.
   - Features the **Dual-Gate PoR**, **Graph Differential Privacy**, **Curriculum Grace Period**, and **SimGNN Calibration**.
   - Includes a full Hedge Fund / Quantitative Finance simulation via a Transformer Actor-Critic agent learning sequential observation curriculums.

2. **`fed-neat-evolution`**: 
   - Focuses heavily on the **Federated NeuroEvolution of Augmenting Topologies (NEAT)** strategy.
   - Contains UI/GUI visualizations, the End-to-End Docker Integration Test workflow, and test suite relaxations for NOTEARS acyclicity limits.
   - Handles the momentum-blended Byzantine consensus and genome merging.

3. **`agentic-impl`**: 
   - An earlier iteration focused on building the baseline aggregator & visualizer bridge.
   - Introduced the adaptive outlier thresholding, math GED validator, and debug telemetry logger.

---

## Core Architectural Implementations (Finance Branch)
The current state of the `finance` branch includes several critical mathematical and programmatic defenses implemented directly into the FL loop:

### 1. Dual-Gate PoR Algorithm
Implemented across `server/aggregator.py` and `client/finance_agent.py`.
*   **Gate 1 (Structural - GED)**: Uses a SimGNN model to approximate the Graph Edit Distance ($\tau$). Rejects topological deviations (DAG structure).
*   **Gate 2 (Causal Effect - CED)**: A newly implemented gate that calculates Causal Effect Divergence ($\theta$). It checks the continuous coefficient matrix $B_k$ against a server-maintained Exponential Moving Average (EMA) $\bar{B}$. This specifically combats "Weight-Only Backdoors" that achieve GED = 0.
*   **MAD Estimator Calibration**: Both gates utilize a robust Median Absolute Deviation (MAD) estimator to dynamically calculate rejection thresholds instead of naive percentiles.

### 2. Client-Side PoR Regularizer
*   Integrated into the PPO actor-critic loss function (`client/finance_agent.py`).
*   Uses a Dual-Frobenius penalty: $\lambda_s \|A_k - A_{\text{global}}\|_F^2 + \lambda_c \|B_k^\circ - \bar{B}\|_F^2$.
*   This acts as a differentiable structural regularizer guiding the agent's policy.

### 3. Liveness Mitigation (Curriculum Grace Period)
*   Solves the "Cold-Start Liveness Failure" (where 100% of honest clients are rejected during early rounds or curriculum expansions).
*   Implemented in `server/aggregator.py`: During the first few rounds, the aggregator gracefully falls back to **Coordinate-wise Median** aggregation instead of standard FedAvg, bypassing strict structural rejections while safely preventing malicious poisoning.

### 4. Graph Differential Privacy (Graph DP)
*   Implemented in `client/causal_discovery.py` to prevent "Logic Leakage".
*   Injects calibrated Laplace noise into the continuous coefficient matrix before DAG extraction to provide $(\epsilon, \delta)$ privacy guarantees.

### 5. Adversary Models (`adversary/`)
*   **`WeightOnlyAdversary`**: A critical adversary model that matches the global topology perfectly (GED=0) but amplifies hidden coefficients to test the CED gate.
*   **`AdaptiveRLAdversary`**, **`TemporalMimicry`**, **`TopologyInversion`**, and **`GradientMimicry`**: Models designed to bypass traditional Euclidean defenses but fail against PoR.

### 6. SimGNN Calibration
*   `calibrate_simgnn.py`: Script to empirically validate the approximation error (MAE) and Pearson correlation of the SimGNN model against exact NetworkX GED algorithms for the 40-node Finance domain.

---

## State of the Manuscript (`paper.tex`)
The manuscript is a **1,300+ line document** that has undergone brutal, multi-round critiques (addressing over 17 tracked gaps).

*   **Title**: *Defending Against Standard-Threat-Model Explanation Poisoning via Causal Proof of Reasoning*. (Scoped specifically to the Standard-Threat-Model, acknowledging NOTEARS faithfulness).
*   **Theorem Qualifications**: The central contradiction proof (Theorem 1) is formally scoped to the Finance domain and explicitly relies on Assumption A1 (NOTEARS Faithfulness). Theorem 2 utilizes a correct Selection Dominance formulation instead of a flawed Jensen's inequality bound.
*   **Empirical Logs**: The paper directly references simulation logs (`saved_models/finance/simulation_logs.json`), contrasting the 30/30 acceptance rate failure of baseline Euclidean defenses with the PoR's strict Cold-Start Conservatism.
*   **Future Work Roadmap**: Identifies unmitigated constraints like Cryptographic Bootstrap (ZKPs/MPC), Colluding Adversaries (Causal Laundering), and Adaptive/Delayed Injection attacks.

---

## Handoff: Next Steps for Claude Code
To continue exactly where Antigravity left off:

1.  **Quantitative Metric Insertion**: 
    - The final task in the previous session was to run `calibrate_simgnn.py` (updated for 40-feature Finance graphs) and `extract_ced_metrics.py`.
    - **Your first action**: Check the outputs of those scripts (or re-run them: `uv run python calibrate_simgnn.py` and `uv run python extract_ced_metrics.py`). 
    - Inject the exact numerical results (Finance MAE, Pearson correlation, and CED rejection rates of the `WeightOnlyAdversary`) directly into the empirical sections of `paper.tex`.
2.  **Verify Manuscript Alignment**: 
    - Ensure any new metrics don't contradict the claims in the text.
    - Check if the abstract or introduction needs minor numerical updates based on the exact Finance domain results.
3.  **Compile & Review**: 
    - The paper has not been compiled (`pdflatex`) since the last massive text injection. If requested by the user, ensure it compiles without breaking TikZ figures or equations.
