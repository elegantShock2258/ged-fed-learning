# FYP Panel Review — Deep-Dive Briefing
## Federated NeuroEvolution with Causal Proof of Reasoning

---

## Part 1: What is NeuroEvolution? (The Core Concept)

### The Fundamental Idea
Traditional neural networks learn by **gradient descent** — they take the derivative of a loss function and nudge weights slightly downward along the error curve.

NeuroEvolution is an entirely different paradigm. Instead of calculating gradients, it borrows ideas from **biological evolution**:

> "If you have 100 different neural networks and only let the best-performing ones reproduce and mutate, after many generations you get increasingly capable networks — without ever computing a single gradient."

This is called **Evolutionary Computation**, and the specific algorithm we implement is called **NEAT** — **NeuroEvolution of Augmenting Topologies**, published by Stanley & Miikkulainen in 2002.

### What makes NEAT special vs. plain Evolution?
Most evolutionary Neural Networks evolve only the **weights** of a fixed architecture. NEAT also evolves the **topology** — the actual structure of connections and neurons. It can:
- **Add a new neuron node** mid-training (a mutation)
- **Add a brand new connection** between existing neurons
- **Disable/re-enable** a connection across generations

This means NEAT doesn't just tune a static MLP — it creates entirely new, dynamically-shaped architectures that fit the problem.

### The Innovation Number (The Key Technical Idea)
The hardest problem with evolving topologies is: **how do you combine two networks when they have different shapes?**

If Network A has 12 connections and Network B has 15 connections after different mutations, how do you average them together? You can't just zip them up by index — the connections are in completely different positions.

NEAT solves this brilliantly via **Innovation Numbers (Innovation Hashes in our code)**:

> Every time a new connection or node type is created anywhere in the population, it gets a **globally unique ID** (the innovation number). Two networks that independently developed the "same" structural mutation will have the same innovation number for it.

This allows **topological crossover** — finding matching innovations across two parent genomes and combining only those, safely ignoring non-matching ones:

```
Parent A:  [1]→[2]  [1]→[3]  [3]→[4]   (innovations 1, 2, 3)
Parent B:  [1]→[2]  [2]→[4]  [4]→[5]   (innovations 1, 4, 5)

Shared innovation 1 is present in both → average its weight
Innovations 2,3 only in A → inherit from A
Innovations 4,5 only in B → inherit from B
```

### Why NeuroEvolution for Federated Learning?

In standard Federated Learning (FedAvg), the server takes weight tensors from all clients and averages them. This works fine when all clients have the **exact same network architecture**. But in our system, each client runs **NEAT** — meaning after a few rounds of evolution, client #5 might have 12 layers while client #12 might have 7, and they have completely different shapes.

**You absolutely cannot average tensors of different shapes.** `numpy` will crash.

FedNEAT solves this elegantly:
- Clients don't send tensors — they send **genome blueprints** (JSON dictionaries of nodes and connections with innovation IDs).
- The server reconstructs these blueprints, finds shared innovation IDs across all accepted clients, and averages only those matched connections.
- Novel mutations in a single accepted client get inherited directly (they don't average to zero).

---

## Part 2: How It's Implemented — `client/agent.py`

Before understanding the server, you need to understand what the client sends and receives.

### The `DynamicGenome` (the "brain" of each client)
Each client maintains a `DynamicGenome` object. This is not a standard PyTorch model — it is a **self-describing neural network blueprint** that knows its own topology.

It stores:
```python
genome.nodes        # dict: {node_id: node_type} e.g. {"0": "input", "7": "hidden"}
genome.connections  # dict: {innovation_hash: {in, out, weight, active}}
genome.hidden_nodes # list of extra hidden nodes that evolved this session
```

### `serialize_genome` / `deserialize_genome`
Because Flower (the FL framework) expects plain numpy arrays in its `Parameters` object, we have to convert the genome dictionary. We do this by:

```python
# Packing the genome blueprint into bytes
json_str = json.dumps(genome_dict)
byte_array = np.array(bytearray(json_str.encode("utf-8")))
```

When the server receives it, it converts back:
```python
byte_arr = parameters_to_ndarrays(fit_res.parameters)[0]
genome_data = json.loads(bytearray(byte_arr).decode("utf-8"))
```

This byte-packing trick lets us transport an arbitrary Python dictionary through Flower's strictly typed parameter channel.

### `fit()` — The Client Local Training Loop
During each FL round, a client runs a **mini evolutionary algorithm locally**:
1. Start with `population_size` clones of the genome received from the server.
2. Mutate all but the first (elite preservation).
3. Evaluate each genome's fitness (accuracy on local data).
4. Keep the best genome ("elite") as the next generation seed.
5. After all generations: extract the **causal graph** from the best genome's intermediate features using NOTEARS.
6. Return: serialized genome blueprint + causal edge list embedded in metrics dict.

---

## Part 2b: What is NOTEARS? (How the Causal Graph is Extracted)

NOTEARS stands for **"No Tears"** — **Non-combinatorial Optimization via Trace Exponential and Augmented lagRangian for Structure learning** (Zheng et al., 2018). It solves one of the hardest problems in statistics: learning which variables causally *cause* which other variables from observational data alone.

### The Problem It Solves

Given a table of observations — say, the intermediate feature activations from the neural network:

```
Sample | Feature_0 | Feature_1 | Feature_2 | Feature_3 |
------------------------------------------------------
  1    |   0.81    |   0.32    |   0.55    |   0.12    |
  2    |   0.44    |   0.71    |   0.21    |   0.96    |
 ...
```

NOTEARS figures out the **Directed Acyclic Graph (DAG)** that best explains the conditional dependencies between these features — i.e., which features cause which other features given the data, without cycles.

The traditional way to do this was an exhaustive search over all possible DAG structures — which is `NP-hard`, exponential in the number of variables.

### The Mathematical Breakthrough

NOTEARS reformulates the combinatorial graph search into a **continuous optimization** problem by reparameterising the structure as a weighted adjacency matrix `W`:

```
min_W   0.5/n · ‖X - X·W‖²   +   λ‖W‖₁

subject to:   h(W) = trace(exp(W ∘ W)) - d = 0
```

Breaking that down:
- **`X`** — the matrix of feature observations (shape `n_samples × d_features`)
- **`W`** — the weight matrix we're trying to learn (shape `d × d`). `W[i,j] ≠ 0` means "feature i causally influences feature j"
- **`‖X - X·W‖²`** — the least-squares reconstruction loss. Think of it as: "can I predict each feature as a linear combination of the others?"
- **`λ‖W‖₁`** — L1 regularisation to force sparsity (most entries of W should be zero — not every feature causes every other)
- **`h(W) = trace(exp(W∘W)) - d = 0`** — **the key constraint** that enforces acyclicity

### The Acyclicity Constraint `h(W)` — The Real Genius

This single equation is the entire reason NOTEARS works:

```python
# Our implementation (causal_discovery.py, lines 73-75):
W_sq = W * W          # elementwise square of weight matrix
E = torch.matrix_exp(W_sq)   # matrix exponential
h = torch.trace(E) - d       # scalar > 0 if graph has cycles, = 0 only for DAG
```

**Why does `trace(exp(W∘W)) - d = 0` mean "no cycles"?**

The matrix exponential `exp(A)` for a matrix A has the property that `trace(exp(A)) ≥ d` with equality if and only if all eigenvalues of A are zero. For `A = W∘W` (element-wise square), all diagonal entries of the matrix power series contain path-length information. If the graph has a cycle, there exists a non-trivial path back to the starting node, contributing positively to the trace. A DAG (no cycles) has zero path-count back to origin for all nodes → `h(W) = 0`.

In plain English: **`h(W) = 0` if and only if the graph encoded in `W` is cycle-free.**

### How the Optimisation Works (Our Implementation)

We use an **Augmented Lagrangian method** — an outer loop that progressively tightens the acyclicity constraint:

```python
rho = 1.0    # penalty coefficient (starts small, grows)
alpha = 0.0  # Lagrange multiplier (accumulates constraint violation)

for outer_iter in range(max_iter):     # Augmented Lagrangian outer loop
    for inner_iter in range(max_iter): # Adam inner optimisation
        loss = (
            0.5/n * ||X - X@W||²        # reconstruction accuracy
          + λ * ||W||₁                   # L1 sparsity
          + 0.5 * rho * h(W)²            # penalty for cycles
          + alpha * h(W)                 # Lagrange multiplier term
        )
        loss.backward()
        optimizer.step()                 # Adam gradient update on W

    if h(W) < 1e-8:                      # Acyclicity achieved — stop
        break
    
    rho *= 10    # Tighten the cycle penalty for the next outer iteration
    alpha += rho * h(W)   # Accumulate the Lagrange multiplier
```

The key insight: instead of hard-enforcing `h(W)=0` from the start (which would be too constrained for gradient descent), we start with a *soft* penalty and gradually increase `rho` until the optimiser is forced to find a DAG. It's the mathematical equivalent of slowly tightening a vice until cycles have nowhere to hide.

### After Optimisation — Thresholding and Naming

Once `W` converges:

```python
# Prune weak edges (noise) below threshold
W_est[np.abs(W_est) < self.threshold] = 0

# Convert to NetworkX DiGraph
G = nx.from_numpy_array(W_est, create_using=nx.DiGraph)

# Relabel nodes from integers to actual feature names (e.g., "smoke", "lung", "bronc")
mapping = {i: name for i, name in enumerate(self.feature_names)}
G = nx.relabel_nodes(G, mapping)
```

The threshold prunes statistically weak edges that NOTEARS found but that have very small weights — these are likely spurious correlations rather than genuine causal links.

### Why Features From the Neural Network, Not Raw Data?

Rather than running NOTEARS on the original dataset columns, we run it on the **intermediate latent features extracted by the MLP** during forward pass. This is deliberate:

1. **The MLP has already abstracted away raw measurement noise.** The latent space is cleaner than raw tabular columns.
2. **Adversaries poison raw features** (e.g., setting column 0 to zero). In the latent space, this poisoning propagates differently — it creates a distinctive structural distortion that NOTEARS reliably captures as a topological deviation from the honest consensus.
3. **This makes the causal graph a fingerprint of the client's reasoning process**, not just their raw data distribution.

### What an Adversary's Graph Looks Like vs. an Honest Graph

An honest client's latent features preserve the true Bayesian Network causal structure (e.g., `smoke → lung → either`). NOTEARS will recover edges that closely match the server's consensus graph → **low GED**.

A `FalseNode` adversary zeros out `feature_column_0` in 20% of their training batch. This feature has near-zero variance → NOTEARS finds no causal relationships to/from it → the edges involving it **completely disappear** from the submitted graph → **high GED** → **rejected**.

---


## Part 3: The Server Files — Deep-Dive

### `aggregator.py` — The Simpler PoR Strategy (FedAvg + Logic Gate)

This is the **conceptually simpler** version. It inherits from Flower's built-in `FedAvg` and simply inserts a filter step before calling the parent's aggregation.

#### Flow:
```
Receive all client results
      │
      ▼
┌─────────────────────────────────────────────────────┐
│  Stage 1: PoR Logic Gate                            │
│  For each client:                                   │
│    1. Deserialize causal_graph_edges from metrics   │
│    2. Rebuild nx.DiGraph from edge list             │
│    3. Call SimGNN(client_graph, consensus_graph)    │
│    4. If GED score > threshold τ → REJECT           │
│    5. Else → ACCEPT and pass to Stage 2             │
└─────────────────────────────────────────────────────┘
      │
      ▼ (only accepted client results)
┌─────────────────────────────────────────────────────┐
│  Stage 2: Weight Aggregation                        │
│  super().aggregate_fit(accepted_results)            │
│  → weighted average of tensors (FedAvg)             │
└─────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────┐
│  Stage 3: Logic Aggregation (_aggregate_logic)      │
│  Momentum-blended graph voting for consensus update │
└─────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────┐
│  Stage 4: SimGNN Fine-tuning                        │
│  _finetune_simgnn_on_consensus → re-anchors GED     │
└─────────────────────────────────────────────────────┘
```

#### `_aggregate_logic()` — The Consensus Graph Update
This is the most intellectually rich piece of `aggregator.py`.

The consensus graph evolves each round via a **momentum-blended dual-threshold voting rule**:

```python
keep_threshold = (1 - momentum) * 0.5 * n_clients  # easy bar to keep existing edges
add_threshold  = (0.5 + 0.5 * momentum) * n_clients  # hard bar to add new edges
```

**Why two different thresholds?** This is the key insight:

- **Keeping** an existing edge should be **easy** (you don't want the consensus to collapse because one round had noisy clients). An edge that exists already just needs to get at least a few votes.
- **Adding** a new edge should be **hard** (you don't want a single noisy client to corrupt the consensus with a spurious new causal link). A new edge only gets added if near-unanimity agrees.

When `momentum=0.85`:
- An existing edge survives as long as `≥ 7.5%` of clients vote for it (extremely forgiving)
- A new edge only appears if `≥ 92.5%` of clients vote for it (near-unanimous)

When `momentum=0.0` (pure 50% majority):
- Keep and add both require `≥ 50%` of clients (symmetric and equal)

#### `_finetune_simgnn_on_consensus()`
After the consensus updates, the SimGNN model has a problem: its training reference graph has changed. The embeddings it learned relative to the old consensus are now slightly wrong.

This function runs a **quick 10-step gradient update** on SimGNN using synthetic graph pairs generated from the new consensus:

- **Similar pairs** (small perturbation → low GED): teach SimGNN what "close to consensus" looks like now.
- **Dissimilar pairs** (large perturbation → high GED): teach SimGNN what "far from consensus" looks like now.

This prevents "stale embedding drift" where the validator starts making wrong decisions because its reference frame has shifted.

> **Note:** This is *disabled* in `fed_neat_strategy.py` to prevent **catastrophic forgetting** — where fine-tuning on only the new consensus destroys the general-purpose distance knowledge SimGNN built during full pre-training.

---

### `fed_neat_strategy.py` — The Full FedNEAT Strategy

This is the **architecturally complete** implementation. It doesn't inherit from `FedAvg` — it implements Flower's `Strategy` interface from scratch, because its aggregation logic is fundamentally incompatible with tensor-averaging.

#### The Key Difference from `aggregator.py`

| Feature | `aggregator.py` (PoRStrategy) | `fed_neat_strategy.py` (FedNEATStrategy) |
|---|---|---|
| Inheritance | `fl.server.strategy.FedAvg` | `fl.server.strategy.Strategy` |
| Model transport | Numpy weight tensors | JSON genome blueprints (bytes) |
| Aggregation | Weighted-average tensors | Innovation Hash Crossover |
| Genome evolution | ❌ No | ✅ Yes |
| Can handle different architectures | ❌ No | ✅ Yes |

#### Stage 1: PoR Logic Gate (identical concept, smarter threshold)

The key upgrade here is `update_dynamic_threshold()` in the LogicValidator:

Instead of using a fixed τ = 0.4 every round, after each round the LogicValidator computes a **percentile-based threshold** from the actual scores observed:

```python
honest_ratio = (num_clients - num_false_nodes) / num_clients  
# e.g., 25/30 = 83.3%
target_percentile = honest_ratio * 100 * 0.95  
# 79.2th percentile of observed GED scores
next_boundary = np.percentile(client_scores, target_percentile)
```

This means the threshold is **calibrated by the actual data**, not a guess. If all clients submit similar graphs, the threshold tightens. If there's large variation, it widens to avoid collateral rejection.

**Why Round 0 is always accepted:**
```python
if server_round <= 0:
    return True, 0.0
```
Round 0 is the initial calibration round. No prior consensus exists yet. The server needs genuine data from all clients (including honest ones) to build the first statistical distribution of GED scores, which it then uses to calibrate the `dynamic_threshold` for Round 1+.

#### Stage 2: Topological Crossover — `_crossover()`

This is the implementation of NEAT crossover:

```python
def _crossover(self, genomes: list) -> dict:
    # 1. Collect the union of ALL innovation hashes seen
    all_innovations = set()
    for g in genomes:
        all_innovations.update(g["connections"].keys())

    merged_connections = {}
    for innov in all_innovations:
        # Look up this innovation in every genome
        sources = [g["connections"][innov] for g in genomes if innov in g["connections"]]
        active_weights = [c["weight"] for c in sources if c["active"]]
        
        if active_weights:
            # Average the weights from all genomes where this innovation was active
            merged_connections[innov] = {
                "weight": sum(active_weights) / len(active_weights), 
                "active": True
            }
        else:
            # Innovation only existed in disabled state → inherit disabled
            merged_connections[innov] = {**sources[0], "active": False}
```

The key design philosophy: a structural innovation (new connection) only gets averaged if it was **active** in at least one accepted genome. Inactive (disabled) connections are inherited but stay disabled. This preserves structural diversity without averaging ghost connections.

#### Stage 3: Ground Truth Anchoring — The BFT Guard
In `_aggregate_logic()`, there is a special line:

```python
if hasattr(self, 'ground_truth_graph') and edge in self.ground_truth_graph.edges():
    new_consensus.add_edge(*edge)  # Always keep ground truth edges
```

This solves a subtle but devastating failure mode called **Ground Truth Graph Attrition**:

> If adversarial clients systematically avoid submitting certain edges (edges that expose their poisoning), over many rounds those edges might get so few votes that the momentum rule eventually drops them from the consensus. The adversary can **gradually erode the server's reference graph** until it no longer detects them.

By anchoring a copy of the initial consensus (`ground_truth_graph`) and unconditionally preserving all its edges, we make the consensus **monotonically stable** at its core. Adversaries cannot erode the server's baseline reference over time.

This is the implementation of the theoretical **Byzantine Fault Tolerance (BFT) bound** — assuming adversaries represent at most `f < 30%` of clients, the `keep_ratio` and `add_ratio` are clamped so honest clients always have sufficient votes to maintain the ground truth.

---

## Part 4: SimGNN — The Logic Validator (logic_validator.py)

### What is Graph Edit Distance (GED)?
Graph Edit Distance is the minimum number of edit operations (add node, delete node, add edge, delete edge, relabel) needed to transform Graph A into Graph B. Exact GED is **NP-hard** — for two 37-node graphs, exact computation is computationally impossible.

**SimGNN** approximates GED in milliseconds using a Siamese Graph Neural Network:

```
Graph A ──► GCN(128) ──► GCN(128) ──► GAT ──► MeanPool ──┐
                                                 MaxPool  ──► concat emb_A
                                                            │
Graph B ──► GCN(128) ──► GCN(128) ──► GAT ──► MeanPool ──┐ emb_B
                                                 MaxPool  ──┘
                                                            │
                              ┌─────────────────────────────┘
                              │ concat [emb_A, emb_B, |emb_A - emb_B|]
                              ▼
                        FC(768→128) → Dropout(0.2) → FC(128→64) → FC(64→1)
                              ▼
                          Sigmoid → GED ∈ [0, 1]
```

The crucial innovation in the forward pass is the **absolute difference term** `|emb_A - emb_B|`:

Without it, the network could produce the same combined embedding for two very different graphs if they happen to fall in opposite directions in embedding space, cancelling each other out. The `diff` term forces the network to explicitly measure *how different* the two embeddings are, not just what they are individually.

### `_nx_to_pyg_data()` — Deterministic Node Mapping
This was a critical bug-fixing moment. When converting a NetworkX graph to PyTorch Geometric format, node ordering matters for edge_index construction. The original naive conversion could produce non-deterministic orderings across different calls:

```python
# WRONG (non-deterministic):
for node in nx_graph.nodes():  # dict ordering is arbitrary
    ...

# CORRECT (alphabetically sorted = always the same order):
sorted_nodes = sorted(list(nx_graph.nodes()))
node_to_idx = {node: i for i, node in enumerate(sorted_nodes)}
```

If two different calls to `_nx_to_pyg_data` produced different node orderings, SimGNN would be comparing node 0 in graph A to node 0 in graph B even if they are completely different semantic nodes — making the GED score random noise.

---

## Part 5: The Hardest Parts to Figure Out and Write

### 1. 🔴 The Ground Truth Graph Attrition Problem
**What it is:** We discovered through testing that after several rounds, the consensus graph would lose edges — specifically the edges most relevant to detecting adversaries. Smart adversaries don't just submit random graphs; they submit graphs where they "conveniently" omit the edges around the features they are poisoning. Over 10+ rounds, those edges accumulate fewer and fewer votes until the momentum rule drops them.

**Why it was hard:** This failure mode is invisible in short simulations. It only manifests clearly after many rounds, which are slow to run. Understanding that adversaries could deliberately engineer graph attrition required thinking like an attacker across temporal dynamics — not just about a single round.

**How we solved it:** The `ground_truth_graph` immutable anchor in `_aggregate_logic()`. But figuring out the right line `if edge in self.ground_truth_graph.edges(): new_consensus.add_edge(*edge)` required us to first understand the problem through debugging logs in `debug_graphs_log.txt`.

### 2. 🔴 The Dynamic Threshold vs. Static Threshold Deadlock
**What it is:** With a fixed τ = 0.45, we observed that in Round 2, ALL clients were rejected — even completely honest ones. The problem: SimGNN's predicted GED scores shifted between rounds due to consensus graph changes. A score that was 0.3 (accepted) in Round 1 would become 0.52 (rejected) in Round 2 because the consensus graph had evolved.

**Why it was hard:** The static threshold creates a Goldilocks problem. Set it too low → reject honest clients (false positives). Set it too high → accept adversaries (false negatives). And the right value changes every round based on what the consensus looks like.

**How we solved it:** The percentile-based `update_dynamic_threshold()` function. Instead of guessing an absolute number, we observe the distribution of scores each round and set the threshold at the `honest_ratio`-th percentile. If 83% of clients are honest, we set the threshold at roughly the 79th percentile — mathematically guaranteeing we should only reject the top ~21% (the adversaries) unless the honest clients cluster very close to the adversary score distribution.

### 3. 🔴 Transporting Genome Structures Through Flower's API
**What it is:** Flower's FL framework was designed for tensor weights. `FitRes.parameters` is explicitly a list of numpy arrays. Our genome is a Python dictionary with nested structures (innovation hashes mapping to connection dicts). These are completely incompatible types.

**Why it was hard:** We couldn't use any Flower built-in aggregation because it only understands tensors. We had to implement the entire Strategy interface from scratch AND invent a serialization protocol that fits tensors but transports dictionaries.

**How we solved it:** The byte-packing trick:
```python
# JSON dict → UTF-8 string → raw bytes → numpy array (of uint8)
s = json.dumps(genome_dict)
byte_array = np.array(bytearray(s, "utf-8"))
```
Then on the receiving end, reverse the process. Numpy treats the genome blueprint as a single 1D array of byte values, which Flower happily transports without complaint.

### 4. 🟡 SimGNN Catastrophic Forgetting
**What it is:** After enabling on-the-fly SimGNN fine-tuning (the 10-step gradient updates after each consensus change), we saw that by Round 5, SimGNN was accepting every client — including obvious adversaries. The fine-tuning was destroying the general GED distance knowledge.

**Why it was hard:** Neural Networks suffer from **catastrophic forgetting** — when you fine-tune on new data, the network overwrites its old knowledge. Fine-tuning SimGNN on only the new consensus (which might be slightly different) destroys its ability to recognize distance patterns it learned during the full 500-epoch pre-training.

**How we solved it:** We **disabled** on-the-fly fine-tuning in `fed_neat_strategy.py` (you can see the commented-out block). Instead, SimGNN is pre-trained once thoroughly on a distribution of graph perturbations, and this general-purpose distance function is locked in place for inference. The file comment explicitly states: *"disabled to prevent Catastrophic Forgetting and lock the highly generalised distance heuristic structure cleanly."*

---

## Part 6: Quick Summary Points for Panel Q&A

**Q: Why not just use FedAvg directly?**
FedAvg averages all client weight tensors including adversaries. An adversarial client submitting poisoned weights directly corrupts the global model. FedAvg weight-based anomaly detection (cosine similarity) can't detect explanation-poisoning attacks where the weights look normal but the causal reasoning is corrupted.

**Q: Why not use exact GED instead of SimGNN?**
Exact GED is NP-hard and computationally intractable for graphs with more than ~10 nodes. The ALARM network has 37 nodes — exact GED would take seconds to minutes per client comparison per round. SimGNN approximates it in `<5ms` on CPU.

**Q: What does the `consensus_momentum` parameter really do?**
It controls the "stickiness" of the consensus graph. High momentum (0.9) means the graph is very slow to change — good for stability but could fail to update if honest clients legitimately evolve their causal structures. Low momentum (0.1) means the graph updates aggressively every round — good for adaptability but vulnerable to transient noisy rounds corrupting the reference graph.

**Q: How is this better than Krum / FLAME / FoolsGold?**
Those methods all operate in weight-space — they look at model parameters and try to identify outlier weights. They fundamentally cannot detect clients who submit correct-looking weights but poisoned causal reasoning. PoR operates in topology-space — it examines the logical structure of how a client's reasoning is organized, which is where explanation-poisoning attacks leave detectable traces.

**Q: What happens when ALL clients get rejected?**
In `fed_neat_strategy.py` line 228: the server falls back to returning the previous round's parameters unchanged. The global model doesn't update — it freezes. This is the correct behavior — it's better to have a temporarily stagnant model than to aggregate poisoned weights.
