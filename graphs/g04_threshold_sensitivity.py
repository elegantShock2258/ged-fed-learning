"""
graphs/g04_threshold_sensitivity.py
=====================================
G04: Threshold Sensitivity Analysis — ADR and FPR vs. τ (dual-axis).
Shows optimal operating region for the GED-based decision gate.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from style_config import (apply_style, save_fig, C,
                           load_ged_scores, is_adversary,
                           NUM_FALSE_NODES, NUM_HONEST, note)

def main():
    apply_style()
    ged = load_ged_scores()
    if ged is None:
        print("  ✗  No GED data"); return

    scores_map = ged['scores']
    items = [(float(info['score']), is_adversary(nid))
             for nid, info in scores_map.items()]
    y_true  = np.array([1 if a else 0 for _, a in items])
    y_score = np.array([s for s, _ in items])
    P = y_true.sum()    # adversaries
    N = (1-y_true).sum() # honest

    # ── sweep ─────────────────────────────────────────────────────────────────
    thresholds = np.linspace(y_score.min()-0.005, y_score.max()+0.005, 400)
    adr_list, fpr_list = [], []
    for t in thresholds:
        pred = (y_score >= t).astype(int)
        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        adr_list.append(tp / P * 100 if P else 0)
        fpr_list.append(fp / N * 100 if N else 0)

    adr = np.array(adr_list)
    fpr = np.array(fpr_list)

    # actual threshold used (midpoint accepted/rejected)
    rej  = [s for s, a in items if y_score[list(y_score).index(s)] >= y_score.min()]
    acc_s = [info['score'] for info in scores_map.values() if info['status']=='accepted']
    rej_s = [info['score'] for info in scores_map.values() if info['status']=='rejected']
    actual_tau = (min(rej_s)+max(acc_s))/2 if rej_s and acc_s else thresholds[len(thresholds)//2]

    # crossover point
    diff = np.abs(adr - (100 - fpr))
    cross_idx = np.argmin(diff)

    # ── figure ────────────────────────────────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(11, 6.5))
    ax2 = ax1.twinx()

    fig.suptitle(
        'GED Threshold Sensitivity Analysis\n'
        'Adversary Detection Rate (ADR) & Honest Client False Positive Rate (FPR) vs. τ',
        fontsize=13, fontweight='bold', y=1.02
    )

    # ── ADR line (left axis = blue) ───────────────────────────────────────────
    l1, = ax1.plot(thresholds, adr, color=C['por'], lw=2.5,
                   label='Adversary Detection Rate (ADR)', zorder=3)
    ax1.fill_between(thresholds, adr, alpha=0.08, color=C['por'])

    # ── FPR line (right axis = red) ───────────────────────────────────────────
    l2, = ax2.plot(thresholds, fpr, color=C['base'], lw=2.5, ls='-.',
                   label='Honest Client FPR', zorder=3)
    ax2.fill_between(thresholds, fpr, alpha=0.06, color=C['base'])

    # ── actual threshold used ─────────────────────────────────────────────────
    ax1.axvline(actual_tau, color=C['threshold'], lw=2, ls='--', zorder=4,
                label=f'Deployed τ ≈ {actual_tau:.3f}')

    # ADR at actual tau
    tau_idx = np.argmin(np.abs(thresholds - actual_tau))
    ax1.scatter(actual_tau, adr[tau_idx], s=110, color=C['threshold'],
                zorder=6, edgecolors='white', linewidths=1.5)
    ax2.scatter(actual_tau, fpr[tau_idx], s=110, color=C['base'],
                zorder=6, edgecolors='white', linewidths=1.5)

    ax1.annotate(
        f'ADR = {adr[tau_idx]:.0f}%\nFPR = {fpr[tau_idx]:.0f}%',
        xy=(actual_tau, adr[tau_idx]),
        xytext=(actual_tau + 0.008, adr[tau_idx] - 18),
        fontsize=9.5, color=C['threshold'],
        arrowprops=dict(arrowstyle='->', color=C['threshold'], lw=1.4)
    )

    # ── optimal region shading ────────────────────────────────────────────────
    # region where ADR > 80 and FPR < 20
    good_mask = (adr > 70) & (fpr < 30)
    if good_mask.any():
        lo = thresholds[good_mask].min()
        hi = thresholds[good_mask].max()
        ax1.axvspan(lo, hi, alpha=0.07, color=C['accept'],
                    label=f'High-performance region  [τ ∈ {lo:.3f}, {hi:.3f}]', zorder=1)

    # ── axes config ───────────────────────────────────────────────────────────
    ax1.set_xlabel('GED Decision Threshold τ', labelpad=10)
    ax1.set_ylabel('Adversary Detection Rate (%)', color=C['por'], labelpad=8)
    ax2.set_ylabel('Honest Client False Positive Rate (%)', color=C['base'], labelpad=8)
    ax1.tick_params(axis='y', colors=C['por'])
    ax2.tick_params(axis='y', colors=C['base'])
    ax1.set_ylim(-5, 110)
    ax2.set_ylim(-5, 110)
    ax2.spines['right'].set_visible(True)
    ax2.spines['right'].set_color(C['border'])

    # combined legend
    lines = [l1, l2,
             plt.Line2D([0],[0], color=C['threshold'], lw=2, ls='--'),
             plt.Rectangle((0,0),1,1, fc=C['accept'], alpha=0.2)]
    labels = ['Adversary Detection Rate (ADR)',
              'Honest Client False Positive Rate (FPR)',
              f'Deployed threshold τ ≈ {actual_tau:.3f}',
              'High-performance operating region']
    ax1.legend(lines, labels, loc='center left', fontsize=9, framealpha=0.92)

    note(ax1, 'FLTrust [Cao et al., 2020]  |  FLDETECTOR [Zhang et al., 2022]')
    plt.tight_layout()
    save_fig(fig, 'G04_threshold_sensitivity.png')

if __name__ == '__main__':
    main()
    print("G04 done.")
