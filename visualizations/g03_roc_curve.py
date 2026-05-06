"""
visualizations/g03_roc_curve
=========================
G03: ROC Curve of GED-based Adversary Detector.
Sweeps threshold τ over observed scores to plot TPR vs FPR, compute AUC.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from style_config import (apply_style, save_fig, C,
                           load_ged_scores, is_adversary, note)

def main():
    apply_style()
    ged = load_ged_scores()
    if ged is None:
        print("  ✗  No GED data"); return

    scores_map = ged['scores']
    items = [(float(info['score']), is_adversary(nid))
             for nid, info in scores_map.items()]
    y_true  = np.array([1 if adv else 0 for _, adv in items])
    y_score = np.array([s for s, _ in items])

    # ── sweep thresholds ──────────────────────────────────────────────────────
    thresholds = np.linspace(y_score.min() - 0.01, y_score.max() + 0.01, 500)
    tpr_list, fpr_list = [], []
    P = y_true.sum()
    N = (1 - y_true).sum()

    for t in thresholds:
        pred = (y_score >= t).astype(int)   # predict adversary if score >= t
        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        tpr_list.append(tp / P if P else 0)
        fpr_list.append(fp / N if N else 0)

    fpr_arr, tpr_arr = np.array(fpr_list), np.array(tpr_list)

    # sort by FPR for trapezoidal AUC
    idx = np.argsort(fpr_arr)
    fpr_s, tpr_s = fpr_arr[idx], tpr_arr[idx]
    auc = np.trapz(tpr_s, fpr_s)

    # ── optimal operating point (Youden's J) ─────────────────────────────────
    j_scores = tpr_arr - fpr_arr
    opt_idx  = np.argmax(j_scores)
    opt_fpr, opt_tpr = fpr_arr[opt_idx], tpr_arr[opt_idx]
    opt_tau          = thresholds[opt_idx]

    # ── figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.suptitle(
        'ROC Curve — GED-Based Adversary Detector\n'
        'Causal PoR Defense  (ASIA Dataset, Single Round)',
        fontsize=13, fontweight='bold', y=1.01
    )

    # shaded AUC region
    ax.fill_between(fpr_s, tpr_s, alpha=0.12, color=C['por'], zorder=1)

    # ROC curve
    ax.plot(fpr_s, tpr_s, color=C['por'], lw=2.5, zorder=3,
            label=f'PoR Causal GED Detector  (AUC = {auc:.3f})')

    # random baseline
    ax.plot([0, 1], [0, 1], color=C['subtext'], lw=1.4, ls='--',
            zorder=2, label='Random classifier  (AUC = 0.500)')

    # optimal point
    ax.scatter(opt_fpr, opt_tpr, s=120, color=C['threshold'],
               zorder=5, edgecolors='white', linewidths=1.5,
               label=f'Optimal τ = {opt_tau:.3f}  (J = {j_scores[opt_idx]:.3f})')
    ax.annotate(f'  τ* = {opt_tau:.3f}\n  TPR={opt_tpr:.2f}  FPR={opt_fpr:.2f}',
                xy=(opt_fpr, opt_tpr), fontsize=9, color=C['threshold'],
                xytext=(opt_fpr + 0.15, opt_tpr - 0.15),
                arrowprops=dict(arrowstyle='->', color=C['threshold'], lw=1.5))

    # Removed redundant perfect classifier star to prevent overlap with the optimal point

    ax.set_xlabel('False Positive Rate  (1 − Specificity)', labelpad=10)
    ax.set_ylabel('True Positive Rate  (Sensitivity / Recall)', labelpad=10)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.set_aspect('equal')
    ax.legend(loc='lower right', fontsize=9.5)

    info_text = (
        f'Total clients: {len(items)}\n'
        f'Adversaries:   {int(P)}\n'
        f'Honest:        {int(N)}'
    )
    ax.text(0.95, 0.50, info_text, transform=ax.transAxes,
            fontsize=9, va='center', ha='right', color=C['subtext'],
            bbox=dict(boxstyle='round,pad=0.4', fc=C['panel'],
                      ec=C['border'], alpha=0.92))

    note(ax, 'FLDETECTOR [Zhang et al., 2022]  |  SimGNN [Bai et al., 2019]')
    plt.tight_layout()
    save_fig(fig, 'G03_roc_curve.png')

if __name__ == '__main__':
    main()
    print("G03 done.")
