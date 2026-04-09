# FYP Evaluation Metrics: Causal PoR Federated Defense

> A comprehensive catalogue of every meaningful metric for your report graphs — grouped by system component with paper references, formulae, and data source notes.

---

## How to Read This Document

Each metric entry includes:
- **Formula / Definition** — what is computed
- **Paper Refs** — which baseline/SOTA papers use it
- **Data Available?** — whether your logs already have it, or whether a computation pass is needed
- **Graph Type** — what kind of chart best visualises it

---

## 1. FL Defense Core Metrics (PoR vs. Baseline)

These are the primary comparative metrics used by **every** FL security paper (DBA [Xie et al., 2020], Krum, FoolsGold, SignGuard, FLDETECTOR, etc.).

---

### 1.1 Adversary Detection Rate (ADR) — *Primary KPI*

**Definition:** Across all rounds, what fraction of adversarial client submissions were correctly rejected by the defense?

$$
\text{ADR} = \frac{\sum_{r=1}^{R} |\text{adversaries rejected in round } r|}{\text{total adversarial submissions}} \times 100\%
$$

**For your setup (30 clients, 5 adversaries, 10 rounds):**
$$\text{Total adversarial submissions} = 5 \times 10 = 50$$

**Paper refs:** DBA [Xie et al., 2020], FLDETECTOR [Zhang et al., 2022], FLAME [Nguyen et al., 2022], FedDF

**Data Available?** ✅ Computable from `simulation_logs.json` — the `rejected_clients` field counts rejections, but you need to know which rejected clients were adversaries. Since adversary IDs are **fixed** (last `num_false_nodes` clients), cross-reference with `ged_scores.json`.

**Graph Type:** Grouped bar chart — PoR ADR vs. Baseline ADR (expected: PoR ~90-100%, Baseline ~0%)

---

### 1.2 False Positive Rate (FPR) — *Honest Client Penalty*

**Definition:** Fraction of honest client submissions incorrectly rejected.

$$
\text{FPR} = \frac{FP}{FP + TN} = \frac{\text{honest clients rejected}}{\text{total honest submissions}}
$$

**Why critical:** The baseline DBA paper and SignGuard [Xu et al., 2022] both table FPR because aggressive defenses like Krum severely penalise honest clients (causing MTA drop from 78% → 42%). This is your PoR's differentiator — it should have low FPR.

**Paper refs:** SignGuard [Xu et al. NeurIPS 2022], FLAME [Nguyen et al. USENIX 2022], Multi-Krum

**Data Available?** ✅ Computable — accepted honest = total honest - honest rejected. Since adversaries are last 5 clients, honest submissions = 25/round.

**Graph Type:** Line chart across rounds — PoR FPR vs. Baseline FPR per round; also summary bar

---

### 1.3 False Negative Rate (FNR) — *Adversary Evasion Rate*

**Definition:** Fraction of adversarial submissions that slipped through undetected.

$$
\text{FNR} = 1 - \text{ADR} = \frac{FN}{TP + FN} = \frac{\text{adversaries accepted}}{\text{total adversarial submissions}}
$$

**Paper refs:** FLAME, FedDF, ByzantineSGD

**Data Available?** ✅ Computable from logs (adversaries that appear in `accepted` in `ged_scores.json`)

**Graph Type:** Line chart across rounds; also heatmap (round × client)

---

### 1.4 Precision, Recall, F1-Score of Detection

Treating detection as binary classification (adversarial=positive):

$$
\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}, \quad F_1 = \frac{2 \cdot P \cdot R}{P + R}
$$

**Paper refs:** Explicitly used in FedDetect [Zhao et al., 2023], FLTrust [Cao et al., 2020], FedInspector

**Data Available?** ✅ Fully computable from logs

**Graph Type:** Summary table + radar chart (PoR vs. Baseline on P / R / F1 / ADR / FPR axes)

---

### 1.5 Main Task Accuracy (MTA) — Global Model Quality

**Definition:** Classification accuracy of the global model on clean test data at each round.

$$\text{MTA}_r = \frac{\text{correct predictions on clean test set in round } r}{\text{total clean test samples}}$$

**Why critical:** Every FL security paper reports MTA alongside security metrics. The standard threat is that defenses crush MTA. PoR should maintain or improve MTA by filtering poisoned weights.

**Paper refs:** FedAvg [McMahan et al., 2017], DBA [Xie et al., 2020], Krum, Trimmed Mean, FedProx

**Data Available?** ⚠️ Partially — `accuracy` is logged in `fit_res.metrics` per round. Check `history.metrics_distributed`. May need another simulation run with explicit test-set logging.

**Graph Type:** Line chart — MTA vs. Round for PoR, Baseline FedAvg (no defense), Baseline+Cosine

---

### 1.6 Attack Success Rate (ASR)

**Definition:** Rate at which the poisoned global model misclassifies targeted inputs.

$$\text{ASR} = \frac{\text{poisoned inputs classified as target label}}{\text{total poisoned test inputs}}$$

**Why critical:** The #1 metric in every backdoor FL paper. Shows whether the attack actually "works" in the global model after aggregation.

**Paper refs:** DBA [Xie et al., 2020 — Table 2], BadNets, FLAME, NEUphi

**Data Available?** ⚠️ Requires generating a poisoned test set (20% feature-zeroed samples) and evaluating the final global model. Straightforward to add as a post-simulation eval script.

**Graph Type:** Bar chart PoR vs. Baseline (expected: PoR ~low, Baseline ~high); line chart ASR across rounds

---

### 1.7 Rejection Rate per Round

**Definition:** Per-round fraction of submitted clients rejected.

$$\text{RejRate}_r = \frac{\text{rejected in round } r}{\text{total submitted in round } r}$$

**Paper refs:** FedDetect, FLDETECTOR, PoRStrategy itself

**Data Available?** ✅ Direct from `simulation_logs.json` — `rejected_clients/accepted_clients` per round

**Graph Type:** Stacked bar chart (accepted vs. rejected per round) — your primary existing visualisation, but now with proper labels + adversary breakdown

---

### 1.8 Cumulative Adversary Suppression Curve

**Definition:** Running total of adversarial submissions blocked as rounds progress.

$$\text{CAC}(r) = \sum_{k=1}^{r} |\text{adversaries rejected in round } k|$$

**Paper refs:** FLAME [Nguyen et al., 2022], progressive defense papers

**Data Available?** ✅ Computable from logs

**Graph Type:** Cumulative line chart (PoR vs. Baseline) — dramatic visual showing PoR's asymptotic protection

---

## 2. GED Score / SimGNN Distribution Metrics

These metrics probe the quality and separability of your GED detector — directly from `ged_scores.json`.

---

### 2.1 GED Score Distribution by Client Type

**Definition:** Full distribution of SimGNN-predicted GED scores, separated by honest vs. adversarial clients.

**Computed values (from your `ged_scores.json`, last round):**
- Honest clients: scores range ~0.69–0.83 (accepted)
- Adversary clients: scores at ~0.84–0.85 (rejected)
- Threshold τ = 0.83 (approx, from context)

**Paper refs:** SimGNN [Bai et al., WSDM 2019], FLTrust, FLAME

**Data Available?** ✅ Direct from `ged_scores.json` — collect across all rounds

**Graph Type:** **Box plot** (honest vs. adversary GED score distributions); violin plot; overlapping KDE density plots

---

### 2.2 Honest-Adversary GED Separability (Cohen's d)

**Definition:** Standardised mean difference between honest and adversary GED score distributions.

$$d = \frac{\mu_{adv} - \mu_{honest}}{\sigma_{pooled}}$$

where $\sigma_{pooled} = \sqrt{\frac{(n_h - 1)\sigma_h^2 + (n_a - 1)\sigma_a^2}{n_h + n_a - 2}}$

**Why critical:** Quantifies how well the GED dimension alone separates the two populations. d > 0.8 = strong effect. Essential for justifying SimGNN as a detector.

**Paper refs:** Effect size reporting recommended in FedInspector, FLAME

**Data Available?** ✅ Computable from per-round `ged_scores.json`

**Graph Type:** Bar chart showing d across rounds; scatter of score vs. client type with decision boundary

---

### 2.3 ROC Curve & AUC of GED Detector

**Definition:** Sweep the GED threshold τ from 0 to 1 and record TPR/FPR at each setting. Plot as ROC curve.

$$\text{AUC} = \int_0^1 \text{TPR}(\tau) \, d\text{FPR}(\tau)$$

**Why critical:** AUC is threshold-independent — shows the intrinsic discriminative power of SimGNN GED scores regardless of chosen τ. AUC=1.0 = perfect separation, =0.5 = random.

**Paper refs:** FLDETECTOR [Zhang et al., 2022], FedDetect, anomaly detection FL literature

**Data Available?** ✅ Compute by treating known adversary IDs as ground truth and sweeping τ over all ged scores collected across rounds.

**Graph Type:** ROC curve (TPR vs. FPR) with AUC annotation; compare PoR vs. Cosine-Similarity Baseline

---

### 2.4 GED Score Stability (Round-to-Round Variance)

**Definition:** Standard deviation of per-client GED scores across rounds, measuring how consistent SimGNN is.

$$\sigma_{GED,cid} = \text{std}(\{s_{cid,r}\}_{r=1}^R)$$

**Why critical:** Low variance = stable, reliable detector. High variance = threshold sensitivity / SimGNN instability.

**Paper refs:** SimGNN [Bai et al., 2019] — stability discussion

**Data Available?** ✅ If ged_scores.json is collected per round (needs multi-round collection, not just last round)

**Graph Type:** Error bar chart — mean ± std GED score per client (split honest/adversary)

---

### 2.5 Threshold Sensitivity Analysis

**Definition:** Plot ADR and FPR as functions of τ to show the optimal operating point.

**Why critical:** Justifies your chosen τ = 0.45 (or whatever the adaptive threshold converges to). Shows the defense is robust to small τ perturbations.

**Paper refs:** FLTrust [Cao et al., 2020 — Figure 5 threshold sensitivity], FLDETECTOR

**Data Available?** ✅ Run post-hoc sweep over τ values using collected GED scores + known adversary IDs

**Graph Type:** Dual-Y line chart (ADR on left, FPR on right) vs. τ; shaded optimal region

---

### 2.6 SimGNN Inference Speedup vs. Exact GED

**Definition:** Ratio of exact A* GED runtime to SimGNN runtime, per graph pair.

$$\text{Speedup} = \frac{T_{A^*}}{T_{SimGNN}}$$

**Known values from paper [Bai et al., 2019]:**
- AIDS dataset: 2174s → 1.26s → **×1720 speedup**
- LINUX: 212s → 0.88s → **×241 speedup**
- IMDB: Timeout → 0.77s → **∞ speedup**

**Paper refs:** SimGNN [Bai et al., WSDM 2019] — Table 2

**Data Available?** ✅ Use published table directly; can benchmark your own ASIA/ALARM graphs vs. networkx `graph_edit_distance` for measured values

**Graph Type:** Log-scale bar chart comparison; Table with MSE and runtime

---

### 2.7 SimGNN MSE on Validation Pairs

**Definition:** Mean Squared Error between SimGNN-predicted normalised GED and ground-truth GED on validation graph pairs.

$$\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (\hat{g}_i - g_i)^2$$

**Paper refs:** SimGNN [Bai et al., 2019] — primary evaluation metric (Table 1: AIDS MSE=1.191×10⁻³, LINUX=0.735×10⁻³)

**Data Available?** ⚠️ Requires running SimGNN on a held-out set of ASIA/ALARM graph pairs with known GED labels. Generate via the perturbation function in `train_simgnn.py`.

**Graph Type:** Training loss curve (MSE vs. epoch) during SimGNN pre-training

---

## 3. Causal Graph / NOTEARS Metrics

These evaluate the quality of the causal DAGs discovered by NOTEARS — unique to your project since you have **ground-truth** Bayesian network structures.

---

### 3.1 Structural Hamming Distance (SHD)

**Definition:** Minimum number of edge additions, deletions, or reversals to transform the estimated DAG into the ground truth.

$$\text{SHD}(G_{est}, G_{true}) = |E_{est} \triangle E_{true}| + \text{orientation penalties}$$

**Why it's perfect for your project:** The ASIA and ALARM networks have **known ground-truth edges**. You can directly compute SHD between each client's NOTEARS output and the true BN structure. This is the gold standard metric in causal discovery.

**Paper refs:** NOTEARS [Zheng et al., NeurIPS 2018 — Table 1], PC algorithm, GES, LiNGAM benchmarks; Zheng (2018) reports SHD across all methods.

**Data Available?** ✅ Ground truth from `bnlearn` (e.g., ASIA=8 edges exactly known); client graphs available from `causal_graph_edges` metric in fit_res. Requires post-processing pass over simulation results.

**Graph Type:** Bar chart — SHD for honest vs. adversary clients (expected: adversary SHD >> honest); box plot per round; SHD vs. data partition size

---

### 3.2 Precision & Recall of Edge Recovery

**Definition:** Treating each directed edge as a binary classification:

$$\text{Precision}_{edge} = \frac{|E_{est} \cap E_{true}|}{|E_{est}|}, \quad \text{Recall}_{edge} = \frac{|E_{est} \cap E_{true}|}{|E_{true}|}$$

**Paper refs:** NOTEARS [Zheng et al., 2018], DGES [Bello et al., 2022], Covariance NOTEARS

**Data Available?** ✅ Computable from client graphs + known ground truth edge sets

**Graph Type:** Scatter plot (Precision vs. Recall) for each client — honest clients cluster near (1,1), adversaries far away; F1 bar chart

---

### 3.3 False Discovery Rate (FDR) of Edges

**Definition:** Proportion of predicted edges that are false positives (spurious edges).

$$\text{FDR} = \frac{|\text{edges in } G_{est} \setminus G_{true}|}{|E_{est}|}$$

**Paper refs:** NOTEARS [Zheng et al., 2018], DAG-GNN [Yu et al., 2019], GOLEM

**Data Available?** ✅ Computable

**Graph Type:** Bar chart FDR for adversary vs. honest (expected: adversary FDR=high, honest FDR=low); correlate with GED score

---

### 3.4 Consensus Graph Edge Overlap with Ground Truth (Jaccard)

**Definition:** Jaccard similarity between the server's consensus graph and the true BN structure across rounds.

$$J(\text{consensus}_r, G_{true}) = \frac{|E_{consensus_r} \cap E_{true}|}{|E_{consensus_r} \cup E_{true}|}$$

**Why critical:** Tracks whether the PoR defense is *converging* the consensus towards the truth or drifting away. This is a **novel metric** specific to your system.

**Paper refs:** Federated graph learning literature; Jaccard used in topology drift papers [MDPI FGL 2023]

**Data Available?** ✅ `consensus_graph.gpickle` available per round (need to save per-round snapshots); ground truth from bnlearn

**Graph Type:** Line chart — Consensus Jaccard vs. Round (PoR vs. Baseline); should show PoR's consensus converging to truth while baseline drifts

---

### 3.5 Consensus Graph Edge Stability (Round-to-Round Jaccard)

**Definition:** Self-Jaccard between consecutive consensus graphs measuring how much the consensus changes per round.

$$\text{Stability}_r = J(\text{consensus}_r, \text{consensus}_{r-1}) = \frac{|E_r \cap E_{r-1}|}{|E_r \cup E_{r-1}|}$$

**Paper refs:** Topology drift papers [MDPI FGL 2023], Link Change Ratio in distributed graph learning

**Data Available?** ⚠️ Requires saving per-round consensus snapshots (add one line to `fed_neat_strategy.py`)

**Graph Type:** Line chart — stability vs. round under different `consensus_momentum` values (0.5, 0.7, 0.85, 0.95); shows momentum hyperparameter effect

---

### 3.6 DAG-ness Violation (Acyclicity Score h(W))

**Definition:** NOTEARS enforces $h(W) = \text{tr}(e^{W \circ W}) - d = 0$. Plot the value of $h(W)$ at convergence per client.

$$h(W) = \text{tr}(\exp(W \circ W)) - d$$

**Why it matters:** Honest clients converge to $h \approx 0$ (true DAG). Adversarial clients with zeroed features may produce $h > 0$ (non-DAG) — another detection signal.

**Paper refs:** NOTEARS [Zheng et al., 2018 — core theorem], DAG-GNN, GOLEM

**Data Available?** ⚠️ Requires logging `h(W)` from the NOTEARS optimiser in `causal_discovery.py`. Single-line addition.

**Graph Type:** Bar chart — per-client h(W) at convergence; honest near 0, adversary potentially elevated; scatter h(W) vs. GED score (correlation plot)

---

### 3.7 NOTEARS Sparsity (L0 Edge Count) vs. Ground Truth

**Definition:** Number of edges in the client's learned DAG vs. number of edges in the true BN.

$$\Delta_{\text{edges}, c} = |E_{est,c}| - |E_{true}|$$

**Why it matters:** Feature poisoning (zero-ing feature_0) eliminates edges involving that feature → estimated graph is *sparser* than truth. Adversary has fewer edges. Quantifies the structural damage.

**Paper refs:** NOTEARS [Zheng et al., 2018], causal discovery benchmarks

**Data Available?** ✅ Computable from existing client graphs

**Graph Type:** Bar chart — edge count per client grouped by honest/adversary; line chart of mean edge count across rounds

---

## 4. FedNEAT Genome Topology Metrics

These are **novel metrics** with no prior FL paper equivalent — they demonstrate the unique FedNEAT contribution.

---

### 4.1 Genome Topological Complexity

**Definition:** Number of active hidden nodes + active connections in the merged global genome after each round.

$$\text{Complexity}_r = |H_r^{active}| + |C_r^{active}|$$

where $H$ = hidden nodes, $C$ = active connections.

**Paper refs:** NEAT [Stanley & Miikkulainen, 2002 — Figure 4], CoDeepNEAT

**Data Available?** ✅ Parse `saved_models/realtime_state.json` — the genome JSON contains `nodes` and `connections` with `active` flags. Can track across rounds if genome is saved per round.

**Graph Type:** Line chart — global genome hidden nodes and active connections vs. round (shows complexification trend)

---

### 4.2 Innovation Hash Diversity

**Definition:** Number of unique innovation hashes across all accepted client genomes per round — measures genetic diversity in the population.

$$\text{InnovDiv}_r = \left|\bigcup_{c \in accepted} \text{innovations}_c^{(r)}\right|$$

**Paper refs:** NEAT [Stanley & Miikkulainen, 2002 — speciation section], CoDeepNEAT, OpenNEAT

**Data Available?** ⚠️ Requires logging accepted genome innovation keys per round from `_crossover()` in `fed_neat_strategy.py`

**Graph Type:** Line chart — unique innovations vs. round; shows whether FedNEAT explores topology space across rounds

---

### 4.3 Crossover Inheritance Ratio

**Definition:** Fraction of the merged genome's connections that came from ≥2 accepted clients (shared heritage) vs. came from only 1 (novel mutations).

$$\text{InheritanceRatio}_r = \frac{|\{innov : |G_{innov}| \geq 2\}|}{|\text{total merged connections}|}$$

**Paper refs:** NEAT [Stanley & Miikkulainen, 2002], FedNEAT design rationale

**Data Available?** ⚠️ Requires minor instrumentation in `_crossover()` to log per-innovation source count

**Graph Type:** Stacked area chart — shared innovations vs. unique mutations per round

---

### 4.4 Compatibility Distance Distribution

**Definition:** NEAT compatibility distance between client genomes, measuring structural similarity:

$$\delta(g_i, g_j) = \frac{c_1 \cdot E + c_2 \cdot D}{N} + c_3 \cdot \overline{W}$$

where $E$=excess genes, $D$=disjoint genes, $\overline{W}$=mean weight difference, $N$=genome length normaliser.

**Why critical:** Shows FedNEAT's topology diversity — clients with very different δ from the global are structural outliers (another adversary signal).

**Paper refs:** NEAT [Stanley & Miikkulainen, 2002 — Equation 1], SAFE-NEAT

**Data Available?** ⚠️ Requires computing δ between each client genome and the global genome per round

**Graph Type:** Box plot — δ distribution for honest vs. adversary clients; show adversaries have higher δ

---

### 4.5 FedNEAT vs. FedAvg — Weight Parameter Count

**Definition:** Number of parameters in the global model under FedNEAT (variable topology) vs. FedAvg (fixed 3-layer MLP).

**Why critical:** FedNEAT can grow/shrink — this demonstrates the adaptive capacity. Show the model actually evolves in structure.

**Paper refs:** NEAT [Stanley & Miikkulainen, 2002], Neural Architecture Search literature

**Data Available?** ✅ Parse genome JSON — `in_features`, `num_classes`, count connections

**Graph Type:** Bar chart — total parameters vs. round for FedNEAT (evolving) and FedAvg (flat line)

---

## 5. Byzantine Robustness Convergence Metrics

---

### 5.1 Training Loss Convergence Curve

**Definition:** Global model cross-entropy loss per round — shows whether the model converges under attack.

$$\mathcal{L}_r = -\frac{1}{|D_{test}|}\sum_{(x,y)\in D_{test}} \log p_\theta(y|x)$$

**Paper refs:** FedAvg [McMahan et al., 2017], FedProx [Li et al., 2020], Krum [Blanchard et al., 2017]

**Data Available?** ✅ Available via `history.losses_distributed` in Flower; logged in baseline `simulation_logs.json`

**Graph Type:** Line chart — loss vs. round for PoR vs. Baseline; show PoR converges faster/lower

---

### 5.2 Round-to-Round Loss Variance (Stability Metric)

**Definition:** Variance of the loss reduction across rounds, measuring convergence stability.

$$\text{LossVar} = \text{Var}(\{\mathcal{L}_1, \mathcal{L}_2, \ldots, \mathcal{L}_R\})$$

**Why critical:** A defense that wildly oscillates loss (rejecting too many honest clients) is brittle. PoR should show lower variance than aggressive defenses like Krum.

**Paper refs:** FedProx [Li et al., 2020 — heterogeneity stability], Byzantine convergence theory

**Data Available?** ✅ Computable from loss history

**Graph Type:** Error bands on loss curves; box plot of round-to-round Δloss for PoR vs. Baseline

---

### 5.3 Honest Client Participation Rate

**Definition:** Mean fraction of honest clients accepted per round.

$$\text{HCPR} = \frac{1}{R} \sum_{r=1}^{R} \frac{|\text{honest accepted in round } r|}{|\text{honest submitted in round } r|}$$

**Why critical:** PoR targets adversaries, not honest clients. HCPR should be ≥ 90%. If it dips, the defense is over-rejecting.

**Paper refs:** SignGuard [Xu et al., 2022 — Table 3], FLAME, FedAvg

**Data Available?** ✅ Computable from logs (total accepted - adversaries accepted = honest accepted)

**Graph Type:** Line chart across rounds; compare PoR vs. Baseline (Baseline should accept ~100% as it never rejects adversaries)

---

### 5.4 Effective Aggregation Efficiency

**Definition:** Mean fraction of submitted updates that actually contribute to aggregation per round.

$$\text{EAE} = \frac{1}{R}\sum_r \frac{\text{accepted}_r}{\text{submitted}_r}$$

**Why it matters:** Shows PoR's trade-off — by rejecting adversaries, it reduces the aggregation pool. Too aggressive = EAE too low = slow convergence.

**Data Available?** ✅ Direct from logs

**Graph Type:** Comparison table: PoR EAE vs. Baseline EAE vs. Krum EAE (from literature)

---

### 5.5 Byzantine Tolerance Fraction (Theoretical vs. Empirical)

**Definition:** Theoretical breakdown point (max adversary fraction tolerable) vs. empirically observed.

$$f_{theoretical} = \frac{N_{adversaries}}{N_{total}} = \frac{5}{30} \approx 16.7\%$$

Compare to theoretical guarantees from PoR's BFT bounds (add_ratio ≤ 0.70, keep_ratio ≥ 0.30):
$$f_{max,PoR} \leq 30\% \text{ (hardcoded in } \texttt{fed\_neat\_strategy.py}\text{)}$$

**Paper refs:** Krum [Blanchard et al., NIPS 2017 — $f < (n-2)/2$], Byzantine SGD, Robust FL theory

**Data Available?** ✅ From code constants + published theorems

**Graph Type:** Comparison table PoR vs. Krum vs. Trimmed Mean vs. FedAvg; theorem box in paper

---

## 6. Comparison Summary Metrics (PoR vs. Baseline FedAvg+Cosine)

These are the headline numbers for your comparison section — distilled from all the above.

| Metric | PoR (FedNEAT) | Baseline (FedAvg+Cosine) | Source |
|---|---|---|---|
| ADR (%) | Compute from logs | ~0% (never rejects) | `simulation_logs.json` |
| FPR (%) | Compute from logs | ~0% (accepts all) | `simulation_logs.json` |
| F1 Detection | Compute | ~0 | Derived |
| MTA (final) | From accuracy logs | From accuracy logs | Flower history |
| ASR (final) | Eval script needed | Eval script needed | Post-sim eval |
| GED Separability (Cohen's d) | From ged_scores.json | N/A (no GED) | ged_scores.json |
| AUC ROC | From ged_scores.json + sweep | 0.5 (no detection) | Derived |
| Consensus Jaccard (to truth) | From gpickle files | N/A | gpickle + bnlearn |
| NOTEARS SHD (honest clients) | Compute from edges | N/A | Client metrics |
| NOTEARS SHD (adversary) | Compute from edges | N/A | Client metrics |
| Genome Complexity (final) | From realtime_state | N/A (fixed MLP) | JSON state |

---

## 7. Visualisation Blueprint (Suggested Graph List)

Priority-ordered — the most impactful graphs for a report first:

| # | Graph Title | Type | Axes / Data | Key Finding |
|---|---|---|---|---|
| G1 | Adversary Detection Rate: PoR vs. Baseline | Grouped Bar | Method vs. ADR % | PoR detects, Baseline misses |
| G2 | Per-Round Acceptance: Honest vs. Adversary | Stacked Bar | Round vs. # clients | Visual PoR gate over time |
| G3 | GED Score Distribution: Honest vs. Adversary | Box/Violin Plot | Client type vs. GED score | Score separability |
| G4 | ROC Curve of GED Detector | ROC Curve | TPR vs. FPR at varying τ | AUC near 1.0 for PoR |
| G5 | False Positive Rate vs. Round | Line Chart | Round vs. FPR | PoR doesn't harm honest clients |
| G6 | Global Model Accuracy vs. Round | Line Chart | Round vs. MTA | Both converge, PoR potentially faster |
| G7 | Global Model Loss vs. Round | Line Chart | Round vs. Loss | Convergence comparison |
| G8 | NOTEARS SHD: Honest vs. Adversary Clients | Bar/Violin | Client type vs. SHD | Adversary graph is structurally wrong |
| G9 | Consensus Graph Jaccard vs. Round | Line Chart | Round vs. J(consensus, truth) | Consensus converges to truth |
| G10 | Threshold (τ) Sensitivity Analysis | Dual-Y Line | τ vs. ADR and FPR | Robust operating region |
| G11 | Cumulative Adversary Suppression | Cumulative Line | Round vs. total blocked | PoR vs. Baseline asymptotic |
| G12 | GED Score Heatmap (clients × rounds) | Heatmap | Client ID vs. Round | Adversary IDs always high score |
| G13 | Genome Complexity vs. Round (FedNEAT) | Line Chart | Round vs. hidden nodes + connections | Complexification evidence |
| G14 | SimGNN Training Loss Curve | Line Chart | Epoch vs. MSE | Validates trained GED proxy |
| G15 | Edge Recovery Precision/Recall (NOTEARS) | Scatter Plot | Precision vs. Recall per client | Ground truth BN recovery quality |
| G16 | NOTEARS FDR: Honest vs. Adversary | Bar Chart | Client type vs. FDR | Adversary has spurious edges |
| G17 | Consensus Stability vs. Momentum | Multi-line | Round vs. Jaccard(r, r-1) | Hyperparameter effect |
| G18 | Attack Success Rate Comparison | Bar Chart | Method vs. ASR % | PoR suppresses backdoor |
| G19 | Precision/Recall/F1 Radar Chart | Radar | P / R / F1 / ADR / FPR | All-in-one defence profile |
| G20 | SimGNN Speedup vs. Exact A* | Log-bar Chart | Method vs. Runtime(s) | Justify SimGNN choice |

---

## 8. Data Collection Checklist

Before generating graphs, ensure the following data is collected:

### Already Available ✅
- Per-round accepted/rejected counts (`simulation_logs.json`)
- Last-round GED scores per client (`ged_scores.json`)
- Final consensus graph (`consensus_graph.gpickle`)
- Ground truth BN structure (via `bnlearn`)
- Client causal graph edges (parsed from `fit_res.metrics`)
- Genome topology (from `realtime_state.json`)
- Baseline loss history (`simulation_logs.json → losses_distributed`)

### Needs One Additional Simulation Pass ⚠️
- Per-round GED scores for all rounds (save `ged_scores.json` per round, not overwrite)
- Per-round consensus graph snapshots (save `consensus_r.gpickle` each round)
- h(W) acyclicity score per client (add logging to `causal_discovery.py`)
- Client accuracy per round (ensure `history.metrics_distributed_fit["accuracy"]` logged)

### Needs Post-simulation Eval Script ⚠️
- ASR — generate poisoned test set, evaluate final global model
- MTA on clean test set — evaluate final global model on full holdout
- SimGNN MSE on held-out graph pairs — generate from perturbation function

### Derived / Computable from Existing Data ✅
- ADR, FPR, FNR, Precision, Recall, F1 (from logs + known adversary IDs)
- ROC + AUC (from GED scores + sweep over τ)
- Cohen's d separability
- NOTEARS SHD, FDR, edge Precision/Recall (client graphs + bnlearn truth)
- Consensus Jaccard to ground truth
- Genome complexity (parse realtime_state.json)
- Cumulative suppression curve

---

## 9. Key Paper References for Each Metric Area

| Area | Primary Papers |
|---|---|
| FL Defense & ADR/ASR/FPR | DBA [Xie et al., ICLR 2020], FLAME [Nguyen et al., USENIX 2022], FLTrust [Cao et al., NDSS 2020], SignGuard [Xu et al., NeurIPS 2022] |
| Byzantine Tolerance | Krum [Blanchard et al., NIPS 2017], Trimmed Mean [Yin et al., ICML 2018], Byzantine-SGD |
| SimGNN / GED Metrics | SimGNN [Bai et al., WSDM 2019] — MSE, Spearman ρ, Kendall τ, runtime |
| Causal Discovery Metrics | NOTEARS [Zheng et al., NeurIPS 2018] — SHD, FDR, TPR; PC algorithm benchmarks |
| FedNEAT / Topology | NEAT [Stanley & Miikkulainen, Evol. Comp. 2002] — innovation, speciation, complexity |
| FL Convergence | FedAvg [McMahan et al., AISTATS 2017], FedProx [Li et al., ICLR 2020] |
| Graph Stability | Federated Graph Learning [MDPI 2023], Jaccard topology drift |
| Consensus / BN Ground Truth | ASIA [Lauritzen & Spiegelhalter, 1988], ALARM [Beinlich et al., 1989] |
