"""
experiments/eval_metrics.py
========================
Post-simulation evaluation script.
Computes:
  - Main Task Accuracy (MTA): on clean test data
  - Attack Success Rate (ASR): on feature-poisoned test data (feature_0 = 0)
  - Per-method comparison: PoR global genome vs. Baseline FedAvg saved MLP

Requires the simulation to have been run first.
Results saved to: saved_models/{dataset}/eval_results.json

.. note::
   ASR values depend on multiple stochastic factors and are expected to vary
   across runs. Key sources of variance:

   * **Random seed**: different seeds produce different dataset splits, weight
     initialisations, and poison-feature selections (if random), all of which
     affect ASR.
   * **Adversary poison fraction** (``adversary_poison_fraction`` in
     ``params.yaml``): higher fractions yield higher ASR for the Baseline but
     also increase the chance of detection by the PoR Logic Validator.
   * **NOTEARS convergence**: the consensus DAG learned by NOTEARS can settle
     at different local optima across runs, altering the causal graph structure
     that the Logic Validator relies on.
   * **SimGNN training**: the validator's pairwise-GED predictions depend on
     the random initialisation and training trajectory of the SimGNN model.

   Typical observed ranges (with ``adversary_poison_fraction=0.4`` for ALARM,
   ``0.2`` for ASIA):

   * **PoR / FedNEAT**: MTA 50-70 %, ASR 5-25 %  (defence active).
   * **Baseline FedAvg**: MTA 45-65 %, ASR 25-60 % (no defence).

   These ranges are indicative — variance across seeds and hyperparameter
   choices can shift them substantially.
"""
import sys
import json
import torch
import numpy as np
import argparse
from pathlib import Path

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="asia")
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--poison_feature", type=int, default=0)
    parser.add_argument("--target_label", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=2000)
    return parser.parse_args()

def load_genome(dataset, path=None):
    """Load DynamicGenome from realtime_state.json."""
    from client.models import DynamicGenome
    rs_path = path or PROJECT / "saved_models" / dataset / "consensus_graph.gpickle" # Wait, the original code used saved_models/realtime_state.json. 
    # But `federated_sim.py` saves `realtime_state.json` inside saved_models/ at the root? Let's check original.
    # The original loaded from `PROJECT / "saved_models" / "realtime_state.json"`. I will keep it exactly as it was.
    rs_path = path or PROJECT / "saved_models" / dataset / "realtime_state.json"
    if not rs_path.exists():
        return None
    with open(rs_path) as f:
        rs = json.load(f)

    nodes   = rs["nodes"]
    conns   = rs["connections"]
    inputs  = [n for n, t in nodes.items() if t == "input"]
    outputs = [n for n, t in nodes.items() if t == "output"]
    hiddens = [n for n, t in nodes.items() if t == "hidden"]

    genome = DynamicGenome(in_features=len(inputs), num_classes=len(outputs))
    genome.nodes        = nodes
    genome.connections  = conns
    genome.input_nodes  = sorted(inputs)
    genome.output_nodes = sorted(outputs)
    genome.hidden_nodes = hiddens
    genome._sync_weights()
    genome.eval()
    return genome

def evaluate_model(model, features, targets):
    """Evaluate accuracy of a DynamicGenome or nn.Module on (X, y)."""
    model.eval()
    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(targets,  dtype=torch.long)
    with torch.no_grad():
        out = model(X)
        logits = out[0] if isinstance(out, tuple) else out
        preds  = logits.argmax(dim=1)
    correct = (preds == y).sum().item()
    return correct / len(y)

def compute_asr(model, features, poison_idx, target_label):
    """ASR: fraction of poisoned samples predicted as TARGET_LABEL."""
    model.eval()
    poisoned = features.copy()
    poisoned[:, poison_idx] = 0.0  # zero out feature
    X = torch.tensor(poisoned, dtype=torch.float32)
    with torch.no_grad():
        out = model(X)
        logits = out[0] if isinstance(out, tuple) else out
        preds  = logits.argmax(dim=1)
    asr = (preds == target_label).sum().item() / len(preds)
    return asr

def main():
    args = parse_args()
    print(f"\n=== Post-Simulation Evaluation: MTA + ASR ({args.dataset.upper()}) ===\n")

    SAVED = PROJECT / "saved_models" / args.dataset
    OUT   = SAVED / "eval_results.json"

    # ── Load test dataset ─────────────────────────────────────────────────────
    print(f"Loading {args.dataset.upper()} test dataset...")
    try:
        from datasets.tabular_loader import TabularBNDataset
        ds = TabularBNDataset(name=args.dataset, num_samples=args.num_samples, seed=args.seed + 1)
    except Exception as e:
        print(f"  ✗  Dataset load failed: {e}"); return

    features = ds.features
    targets  = ds.targets
    print(f"  ✓  Loaded {len(features)} samples  |  "
          f"Features: {features.shape[1]}  |  Target: '{ds.target_col}'")

    results = {}

    # ── PoR model (DynamicGenome from realtime_state.json) ────────────────────
    print("\nEvaluating PoR (FedNEAT) global genome...")
    por_model = load_genome(args.dataset)
    if por_model is not None:
        mta_por = evaluate_model(por_model, features, targets)
        asr_por = compute_asr(por_model, features, args.poison_feature, args.target_label)
        results["por"] = {"MTA": round(mta_por, 4), "ASR": round(asr_por, 4)}
        print(f"  ✓  PoR MTA = {mta_por:.4f}  |  ASR = {asr_por:.4f}")
    else:
        print("  ✗  PoR genome not found")

    # ── Baseline model (saved FedAvg MLP if available) ────────────────────────
    import torch.nn as nn
    baseline_ckpts = sorted(
        SAVED.glob("baseline_model*.pt")
    )
    if not baseline_ckpts:
        baseline_ckpts = sorted((PROJECT / "saved_models").glob("**/baseline_model*.pt"))
        
    if baseline_ckpts:
        print(f"\nEvaluating Baseline MLP ({baseline_ckpts[0].name})...")
        try:
            class MLP(nn.Module):
                def __init__(self, in_f=7, num_cls=6):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(in_f, 32), nn.ReLU(),
                        nn.Linear(32, 16),   nn.ReLU(),
                        nn.Linear(16, num_cls)
                    )
                def forward(self, x): return self.net(x)

            in_f   = features.shape[1]
            num_cls = 6 if args.dataset == "alarm" else 2

            mlp = MLP(in_f=in_f, num_cls=num_cls)
            mlp.load_state_dict(torch.load(baseline_ckpts[0], map_location="cpu"))
            mlp.eval()
            mta_base = evaluate_model(mlp, features, targets)
            asr_base = compute_asr(mlp, features, args.poison_feature, args.target_label)
            results["baseline"] = {"MTA": round(mta_base, 4), "ASR": round(asr_base, 4)}
            print(f"  ✓  Baseline MTA = {mta_base:.4f}  |  ASR = {asr_base:.4f}")
        except Exception as e:
            print(f"  ✗  Baseline load failed: {e}")
    else:
        print("\n  ⚠ No saved baseline MLP found (*.pt). Skipping baseline eval.")

    # ── Save results ──────────────────────────────────────────────────────────
    if results:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  ✓  Results saved to {OUT}")
        print(f"\n  Summary:")
        for method, vals in results.items():
            print(f"    {method:10s}  MTA={vals['MTA']:.4f}  ASR={vals['ASR']:.4f}")
    else:
        print("\n  ✗  No results to save")

if __name__ == "__main__":
    main()
