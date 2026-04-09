"""
graphs/g19_consensus_jaccard_rounds.py
=========================================
G19 (Blueprint G9): Consensus Graph Jaccard vs. ASIA Ground Truth — per round.
Uses: saved_models/asia/consensus_round_N.gpickle (from collect_per_round_data.py + re-run)
      Falls back to single-point from consensus_graph.gpickle if per-round files absent.
"""
import sys
import glob
import pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
from style_config import (apply_style, save_fig, C,
                           load_consensus_graph, load_por_logs,
                           ASIA_GT_EDGES, note)

PROJECT = Path(__file__).parent.parent
SAVED   = PROJECT / "saved_models" / "asia"

def jaccard(est_edges, gt_edges):
    est, gt = set(est_edges), set(gt_edges)
    inter   = len(est & gt)
    union   = len(est | gt)
    return inter / union if union > 0 else 0.0

def edge_precision(est, gt):
    e, g = set(est), set(gt)
    return len(e & g) / len(e) if e else 0.0

def edge_recall(est, gt):
    e, g = set(est), set(gt)
    return len(e & g) / len(g) if g else 0.0

def main():
    apply_style()

    gt_edges = ASIA_GT_EDGES
    por_log  = load_por_logs()

    # ── Try to load per-round consensus snapshots ─────────────────────────────
    per_round_files = sorted(
        SAVED.glob("consensus_round_*.gpickle"),
        key=lambda p: int(p.stem.split('_')[-1])
    )
    has_per_round = len(per_round_files) > 0

    rounds_data = []   # list of (round_num, jaccard, precision, recall)

    if has_per_round:
        print(f"  ✓  Found {len(per_round_files)} per-round consensus files")
        for p in per_round_files:
            r = int(p.stem.split('_')[-1])
            with open(p, 'rb') as f:
                cg = pickle.load(f)
            cg_edges = list(cg.edges())
            rounds_data.append((
                r,
                jaccard(cg_edges, gt_edges),
                edge_precision(cg_edges, gt_edges),
                edge_recall(cg_edges, gt_edges),
                cg.number_of_edges()
            ))
    else:
        print("  ⚠  No per-round files — using final consensus only")
        print("      Run: python graphs/collect_per_round_data.py  then re-run the simulation")
        cg = load_consensus_graph()
        if cg:
            r = por_log['num_rounds'] if por_log else 15
            cg_edges = list(cg.edges())
            rounds_data.append((
                r,
                jaccard(cg_edges, gt_edges),
                edge_precision(cg_edges, gt_edges),
                edge_recall(cg_edges, gt_edges),
                cg.number_of_edges()
            ))

    if not rounds_data:
        print("  ✗  No data available"); return

    rs    = [d[0] for d in rounds_data]
    jacs  = [d[1] for d in rounds_data]
    precs = [d[2] for d in rounds_data]
    recs  = [d[3] for d in rounds_data]
    n_edges = [d[4] for d in rounds_data]

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 11), sharex=has_per_round)
    fig.suptitle(
        'Consensus Graph Convergence Towards ASIA Ground Truth\n'
        'Causal PoR Defense — Edge Recovery Quality Across Rounds',
        fontsize=13, fontweight='bold', y=1.02
    )

    # ── Top: Jaccard + Precision + Recall ─────────────────────────────────────
    if has_per_round:
        ax1.plot(rs, jacs,  color=C['por'],    lw=2.5, marker='o', markersize=7,
                 markeredgecolor='white', markeredgewidth=1.2,
                 label='Jaccard similarity  J(consensus, GT)', zorder=3)
        ax1.fill_between(rs, jacs, alpha=0.10, color=C['por'])
        ax1.plot(rs, precs, color=C['honest'], lw=2.2, ls='-.', marker='s',
                 markersize=6, markeredgecolor='white', markeredgewidth=1.2,
                 label='Edge Precision  (correct / predicted)', zorder=3)
        ax1.plot(rs, recs,  color=C['adv'],   lw=2.2, ls='--', marker='^',
                 markersize=6, markeredgecolor='white', markeredgewidth=1.2,
                 label='Edge Recall  (correct / GT total)', zorder=3)
    else:
        # Single-point scatter — annotated
        ax1.scatter(rs, jacs,  color=C['por'],    s=180, zorder=4,
                    edgecolors='white', linewidths=1.5,
                    label=f'Jaccard = {jacs[0]:.3f} (Round {rs[0]})')
        ax1.scatter(rs, precs, color=C['honest'], s=130, marker='s', zorder=4,
                    edgecolors='white', linewidths=1.5,
                    label=f'Precision = {precs[0]:.3f}')
        ax1.scatter(rs, recs,  color=C['adv'],    s=130, marker='^', zorder=4,
                    edgecolors='white', linewidths=1.5,
                    label=f'Recall = {recs[0]:.3f}')

        ax1.text(0.5, 0.5,
                 '⚠  Only final-round data available.\n'
                 'Run collect_per_round_data.py + re-simulation for full trajectory.',
                 transform=ax1.transAxes, fontsize=11, ha='center', va='center',
                 color=C['adv'], style='italic',
                 bbox=dict(boxstyle='round,pad=0.5', fc=C['adv_light'], alpha=0.85))

    ax1.axhline(1.0, color=C['accept'], lw=1.0, ls=':', alpha=0.4,
                label='Perfect = 1.0')
    ax1.set_ylabel('Edge Overlap Score  (0 → 1)', labelpad=8)
    ax1.set_title('Jaccard / Precision / Recall: Consensus ↔ ASIA Ground Truth\n'
                  '(Higher = consensus is structurally closer to the true BN)',
                  fontsize=11, pad=12)
    ax1.set_ylim(-0.05, 1.10)
    ax1.legend(fontsize=9.5, loc='lower right')
    note(ax1, 'ASIA GT: 8 nodes, 8 directed edges  |  [Lauritzen & Spiegelhalter, 1988]')

    # ── Bottom: edge count evolution ───────────────────────────────────────────
    if has_per_round:
        ax2.plot(rs, n_edges, color=C['por'], lw=2.5,
                 marker='o', markersize=7, markeredgecolor='white',
                 markeredgewidth=1.2, label='Consensus edge count', zorder=3)
        ax2.fill_between(rs, n_edges, alpha=0.08, color=C['por'])
    else:
        ax2.bar(rs, n_edges, width=0.5, color=C['por'], alpha=0.85,
                zorder=3, edgecolor='white', label='Consensus edges (final round)')

    ax2.axhline(len(gt_edges), color=C['threshold'], lw=2, ls='--',
                label=f'Ground truth edge count = {len(gt_edges)}')
    ax2.set_xlabel('Federated Round', labelpad=10)
    ax2.set_ylabel('Total Directed Edges in Consensus Graph', labelpad=8)
    ax2.set_title('Consensus Graph Edge Count vs. Round\n'
                  '(Divergence from GT=8 indicates spurious or missing edges)',
                  fontsize=11, pad=12)
    ax2.set_ylim(0, max(n_edges) * 1.30 + 2)
    ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax2.legend(fontsize=9.5)
    note(ax2, 'consensus_momentum controls smoothing of edge updates per round')

    plt.tight_layout()
    save_fig(fig, 'G19_consensus_jaccard_rounds.png')

if __name__ == '__main__':
    main()
    print("G19 done.")
