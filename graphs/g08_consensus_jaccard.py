"""
graphs/g08_consensus_jaccard.py
=================================
G08: Consensus Graph structural comparison against ASIA ground truth.
Bar chart of edge overlap metrics (Jaccard, Precision, Recall, F1).
Also shows missing/extra edges vs ground truth.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from style_config import (apply_style, save_fig, C,
                           load_consensus_graph,
                           load_rejected_edge_diff,
                           ASIA_GT_EDGES, ASIA_GT_NODES, note)

def edge_metrics(est_edges, gt_edges):
    est = set(est_edges)
    gt  = set(gt_edges)
    tp  = len(est & gt)
    fp  = len(est - gt)
    fn  = len(gt  - est)
    union = len(est | gt)
    jaccard   = tp / union   if union > 0 else 0.0
    precision = tp / len(est) if est    else 0.0
    recall    = tp / len(gt)  if gt     else 0.0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0.0
    return dict(jaccard=jaccard, precision=precision, recall=recall, f1=f1,
                tp=tp, fp=fp, fn=fn)

def main():
    apply_style()

    # ── load consensus graph ──────────────────────────────────────────────────
    cg = load_consensus_graph()
    if cg is None:
        print("  ✗  No consensus graph found"); return

    cg_edges = list(cg.edges())
    gt_edges  = ASIA_GT_EDGES
    gt_set    = set(gt_edges)

    m = edge_metrics(cg_edges, gt_edges)

    # ── figure: 1 row, 2 panels ───────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle(
        'Consensus Graph Quality vs. ASIA Bayesian Network Ground Truth\n'
        'Causal PoR Defense — Structural Edge Recovery Analysis',
        fontsize=13, fontweight='bold', y=1.02
    )

    # ── Left: metric bar chart ────────────────────────────────────────────────
    metrics_names  = ['Jaccard\nSimilarity', 'Edge\nPrecision',
                      'Edge\nRecall', 'F1 Score']
    metrics_vals   = [m['jaccard'], m['precision'], m['recall'], m['f1']]
    bar_colors     = [C['por'], C['honest'], C['adv'], C['gold']]

    bars = ax1.bar(metrics_names, metrics_vals, width=0.5,
                   color=bar_colors, alpha=0.85, zorder=3,
                   edgecolor='white', linewidth=1.0)

    for bar, val in zip(bars, metrics_vals):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.02,
                 f'{val:.3f}', ha='center', va='bottom',
                 fontsize=12, fontweight='bold', color=C['text'])

    # Reference lines
    ax1.axhline(1.0, color=C['accept'], lw=1.2, ls='--',
                alpha=0.6, label='Perfect score = 1.0')
    ax1.axhline(0.5, color=C['neutral'], lw=1.0, ls=':',
                alpha=0.5, label='Chance level = 0.5')

    ax1.set_ylim(0, 1.18)
    ax1.set_ylabel('Score  (0.0 → 1.0)', labelpad=8)
    ax1.set_title('Edge Overlap Metrics: Consensus ↔ Ground Truth', fontsize=11, pad=12)
    ax1.legend(fontsize=9, loc='upper right')

    # summary text box
    summary = (
        f"Consensus edges:  {len(cg_edges)}\n"
        f"Ground truth:        {len(gt_edges)}\n"
        f"Correctly recovered: {m['tp']}\n"
        f"Spurious (FP):       {m['fp']}\n"
        f"Missing (FN):        {m['fn']}"
    )
    ax1.text(0.02, 0.97, summary, transform=ax1.transAxes,
             fontsize=9.5, va='top', color=C['text'], family='monospace',
             bbox=dict(boxstyle='round,pad=0.45', fc=C['panel'],
                       ec=C['border'], alpha=0.92))
    note(ax1, 'ASIA BN [Lauritzen & Spiegelhalter, 1988]  |  NOTEARS [Zheng et al., 2018]')

    # ── Right: edge overlap visual ────────────────────────────────────────────
    # Three groups of edges shown as horizontal bands
    correct_edges = [e for e in gt_edges if e in set(cg_edges)]
    missing_edges = [e for e in gt_edges if e not in set(cg_edges)]
    spurious_edges = [e for e in cg_edges if e not in gt_set]

    groups = [
        ('Correctly\nRecovered\n(True Positive)', correct_edges,  C['honest']),
        ('Missing from\nConsensus\n(False Negative)', missing_edges, C['adv']),
        ('Spurious in\nConsensus\n(False Positive)', spurious_edges, C['base']),
    ]

    y_pos = 0
    yticks, ylabels = [], []
    EDGE_HEIGHT = 0.7
    GAP = 0.4

    for g_label, edges, color in groups:
        group_start = y_pos
        for e in edges:
            src, dst = str(e[0]), str(e[1])
            ax2.barh(y_pos, 1, EDGE_HEIGHT,
                     color=color, alpha=0.80, edgecolor='white',
                     linewidth=0.5, zorder=3)
            ax2.text(0.5, y_pos, f'{src}  →  {dst}',
                     ha='center', va='center', fontsize=9,
                     color='white', fontweight='bold', zorder=4)
            y_pos += EDGE_HEIGHT + 0.1

        if not edges:
            ax2.barh(y_pos, 1, EDGE_HEIGHT, color=color,
                     alpha=0.25, edgecolor=color, linewidth=1.0,
                     linestyle='--', zorder=3)
            ax2.text(0.5, y_pos, 'none', ha='center', va='center',
                     fontsize=9, color=color, style='italic', zorder=4)
            y_pos += EDGE_HEIGHT + 0.1

        group_mid = (group_start + y_pos - EDGE_HEIGHT/2) / 2
        yticks.append(group_mid)
        ylabels.append(f'{g_label}\n({len(edges)})')
        y_pos += GAP

        # divider line
        ax2.axhline(y_pos - GAP/2, color=C['border'], lw=0.8, ls='-')

    ax2.set_xlim(0, 1)
    ax2.set_ylim(-0.3, y_pos)
    ax2.set_yticks([])
    ax2.set_xticks([])
    ax2.set_xlabel('')
    ax2.set_title('Edge-by-Edge Recovery Analysis\n'
                  '(Consensus Graph vs. ASIA Ground Truth)',
                  fontsize=11, pad=12)
    ax2.spines['left'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.grid(False)

    # group labels on left
    for lbl, col, (_, edges, _col) in zip(
        [g[0] for g in groups], [g[2] for g in groups], groups
    ):
        pass  # already shown inline

    # legend
    patches = [
        mpatches.Patch(color=C['honest'], alpha=0.85, label=f'TP — Correct ({m["tp"]})'),
        mpatches.Patch(color=C['adv'],    alpha=0.85, label=f'FN — Missing ({m["fn"]})'),
        mpatches.Patch(color=C['base'],   alpha=0.85, label=f'FP — Spurious ({m["fp"]})'),
    ]
    ax2.legend(handles=patches, loc='lower right', fontsize=9.5, framealpha=0.92)
    note(ax2, f'SHD = {m["fp"] + m["fn"]}  (FP + FN edge errors from ground truth)')

    plt.tight_layout()
    save_fig(fig, 'G08_consensus_jaccard_groundtruth.png')

if __name__ == '__main__':
    main()
    print("G08 done.")
