"""
GED Score Distribution Analysis — Causal PoR Finance Defense
=============================================================
Produces the empirical evidence that honest and adversary GED scores
are statistically separable — the key claim of the paper.

For each of 4 trajectory types (Honest / Temporal / Reversed / Gradient),
collects GED scores over `num_samples` episodes and:
  1. Saves raw distributions to eval/ged_distributions.json
  2. Generates a violin/box plot as eval/ged_distribution_plot.png
  3. Computes Cohen's d separability metric between honest and each adversary

Usage:
    uv run python eval/ged_distribution.py
"""

import os
import sys
import json
import numpy as np
import torch
import random
import logging
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client.finance_env import FinanceTradingEnv
from client.finance_agent import FinanceClient
from adversary.finance_poisoning import FalseTraderNode
from adversary.finance_adversary_pool import ReversedOrderNode, GradientMimicryNode
from client.causal_discovery import CognitiveModule
import networkx as nx

logging.basicConfig(level=logging.WARNING)

NUM_SAMPLES = 60   # GED samples per distribution (increase for paper)
DEVICE      = torch.device("cpu")


def _traj_to_graph(trajectories, n_nodes=36):
    cm = CognitiveModule(num_tools=n_nodes, threshold=0.05)
    edges_str = cm.extract_causal_graph(trajectories)
    g = nx.DiGraph()
    g.add_nodes_from(range(n_nodes))
    try:
        g.add_edges_from(eval(edges_str))
    except Exception:
        pass
    return g


def _consensus_graph():
    g = nx.DiGraph()
    g.add_nodes_from(range(36))
    for i in range(32):
        g.add_edge(i, i + 1)
    g.add_edge(32, 33)
    return g


def _jaccard_ged(g1, g2):
    e1 = set(g1.edges())
    e2 = set(g2.edges())
    if len(e1 | e2) == 0:
        return 0.0
    return 1.0 - len(e1 & e2) / len(e1 | e2)


def _collect_ged_samples(client_cls, name, consensus, num_samples, trigger=False):
    """Collect `num_samples` GED scores from a given client class."""
    scores = []
    for i in range(num_samples):
        random.seed(i)
        np.random.seed(i)
        torch.manual_seed(i)

        client = client_cls(str(i), DEVICE)
        if trigger:
            client.env.inject_trigger = True

        client.model.eval()
        env  = client.env
        trajs = []

        for _ in range(3):  # 3 episodes per sample
            obs  = env.reset()
            done = False
            traj = []
            while not done:
                obs_t  = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                logits, _ = client.model(obs_t)
                # Mix epsilon-greedy to get more diverse trajectories
                if random.random() < 0.7:
                    action = torch.argmax(logits, dim=-1).item()
                else:
                    action = random.randint(0, env.action_space_n - 1)
                obs, _, done, _ = env.step(action)
                traj.append(action)
            trajs.append(traj)

        g   = _traj_to_graph(trajs)
        ged = _jaccard_ged(g, consensus)
        scores.append(round(ged, 4))

        if (i + 1) % 20 == 0:
            print(f"  {name}: {i+1}/{num_samples} samples collected (mean GED={np.mean(scores):.4f})")

    return scores


def _cohens_d(a, b):
    """Effect size (Cohen's d) between two score distributions."""
    a, b = np.array(a), np.array(b)
    pooled_std = np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
    return float((np.mean(b) - np.mean(a)) / (pooled_std + 1e-9))


def run_ged_distribution():
    consensus = _consensus_graph()
    os.makedirs("eval", exist_ok=True)

    distributions = {}

    sources = [
        ("Honest",           FinanceClient,    False),
        ("Temporal Mimicry", FalseTraderNode,  True),
        ("Reversed Order",   ReversedOrderNode,True),
        ("Gradient Mimicry", GradientMimicryNode, True),
    ]

    for name, cls, trigger in sources:
        print(f"\nCollecting GED distribution for: {name}")
        scores = _collect_ged_samples(cls, name, consensus, NUM_SAMPLES, trigger)
        distributions[name] = {
            "scores":   scores,
            "mean":     round(float(np.mean(scores)), 4),
            "std":      round(float(np.std(scores)),  4),
            "median":   round(float(np.median(scores)),4),
            "p5":       round(float(np.percentile(scores, 5)),  4),
            "p95":      round(float(np.percentile(scores, 95)), 4),
        }
        print(f"  → mean={distributions[name]['mean']:.4f} ± {distributions[name]['std']:.4f}")

    # ── Cohen's d separability metrics ─────────────────────────────────────────
    hon_scores = distributions["Honest"]["scores"]
    separability = {}
    for name in ["Temporal Mimicry", "Reversed Order", "Gradient Mimicry"]:
        d = _cohens_d(hon_scores, distributions[name]["scores"])
        separability[name] = round(d, 3)
        print(f"  Cohen's d (Honest vs {name}): {d:.3f}")

    # ── Save raw data ────────────────────────────────────────────────────────────
    out = {
        "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_samples":  NUM_SAMPLES,
        "distributions": distributions,
        "cohens_d":     separability,
    }
    json_path = "eval/ged_distributions.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nDistribution data saved to {json_path}")

    # ── Generate matplotlib figure ───────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("GED Score Distributions: Honest vs Adversary Trajectories\n(Finance / Hedge Fund PoR Environment)", fontsize=13, fontweight="bold")

        # Violin plot
        labels = list(distributions.keys())
        data   = [distributions[k]["scores"] for k in labels]
        colors = ["#4CAF50", "#FF5722", "#2196F3", "#9C27B0"]

        parts = ax1.violinplot(data, positions=range(len(labels)), showmeans=True, showmedians=True)
        for i, (pc, color) in enumerate(zip(parts["bodies"], colors)):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        ax1.axhline(y=0.12, color="red", linestyle="--", linewidth=1.5, label="τ = 0.12 (GED threshold)")
        ax1.set_xticks(range(len(labels)))
        ax1.set_xticklabels(labels, rotation=15, ha="right")
        ax1.set_ylabel("GED Score (Jaccard Distance)")
        ax1.set_title("GED Score Violin Plot")
        ax1.legend()
        ax1.set_ylim(0, 1)
        ax1.grid(axis="y", alpha=0.3)

        # Cohen's d bar chart
        adv_names = list(separability.keys())
        d_values  = [separability[k] for k in adv_names]
        bar_colors = ["#FF5722", "#2196F3", "#9C27B0"]
        bars = ax2.bar(adv_names, d_values, color=bar_colors, alpha=0.8, edgecolor="black")
        for bar, d in zip(bars, d_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                     f"d={d:.2f}", ha="center", va="bottom", fontweight="bold")
        ax2.axhline(y=0.8, color="green", linestyle="--", label="Large effect (d=0.8)")
        ax2.set_ylabel("Cohen's d Effect Size")
        ax2.set_title("Separability: Honest vs Each Adversary Type")
        ax2.legend()
        ax2.grid(axis="y", alpha=0.3)
        ax2.set_ylim(0, max(d_values) * 1.3 if d_values else 2)

        plt.tight_layout()
        fig_path = "eval/ged_distribution_plot.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Figure saved to {fig_path}")

    except ImportError:
        print("matplotlib not available — skipping plot generation.")

    # ── Print summary table ──────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("GED Score Distribution Summary")
    print("="*65)
    print(f"{'Source':<25} {'Mean':>8} {'Std':>8} {'P5':>8} {'P95':>8}")
    print("-"*65)
    for name, stats in distributions.items():
        print(f"{name:<25} {stats['mean']:>8.4f} {stats['std']:>8.4f} {stats['p5']:>8.4f} {stats['p95']:>8.4f}")
    print("-"*65)
    print(f"\nSeparability (Cohen's d):")
    for name, d in separability.items():
        label = "Large" if d > 0.8 else "Medium" if d > 0.5 else "Small"
        print(f"  Honest vs {name:<25}: d = {d:.3f}  [{label} effect]")

    return out


if __name__ == "__main__":
    run_ged_distribution()
