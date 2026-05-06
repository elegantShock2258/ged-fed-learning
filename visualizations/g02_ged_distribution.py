"""
visualizations/g02_ged_distribution
================================
G02: GED Score Distribution — Honest vs. Adversary Clients
Violin + boxplot showing score separability.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from style_config import (apply_style, save_fig, C,
                           load_ged_scores, is_adversary,
                           NUM_CLIENTS, NUM_FALSE_NODES, note)

def main():
    apply_style()
    ged = load_ged_scores()
    if ged is None:
        print("  ✗  No GED scores data found"); return

    round_num = ged.get('round', '?')
    scores_map = ged['scores']

    # ── classify clients ──────────────────────────────────────────────────────
    groups = {'Honest\nAccepted': [], 'Honest\nRejected': [],
              'Adversary\nAccepted': [], 'Adversary\nRejected': []}

    honest_all, adv_all = [], []
    for nid, info in scores_map.items():
        s, st = info['score'], info['status']
        if is_adversary(nid):
            adv_all.append(s)
            key = 'Adversary\nAccepted' if st == 'accepted' else 'Adversary\nRejected'
        else:
            honest_all.append(s)
            key = 'Honest\nAccepted' if st == 'accepted' else 'Honest\nRejected'
        groups[key].append(s)

    # remove empty groups
    groups = {k: v for k, v in groups.items() if v}

    # ── threshold estimate (midpoint between max-accepted and min-rejected) ───
    all_scores  = [info['score'] for info in scores_map.values()]
    rej_scores  = [info['score'] for info in scores_map.values()
                   if info['status'] == 'rejected']
    acc_scores  = [info['score'] for info in scores_map.values()
                   if info['status'] == 'accepted']
    tau = (min(rej_scores) + max(acc_scores)) / 2 if (rej_scores and acc_scores) else None

    # ── Cohen's d separability ────────────────────────────────────────────────
    if honest_all and adv_all:
        n_h, n_a   = len(honest_all), len(adv_all)
        m_h, m_a   = np.mean(honest_all), np.mean(adv_all)
        s_h, s_a   = np.std(honest_all, ddof=1), np.std(adv_all, ddof=1)
        pool_std   = np.sqrt(((n_h-1)*s_h**2 + (n_a-1)*s_a**2) / (n_h+n_a-2))
        cohens_d   = abs(m_a - m_h) / pool_std if pool_std > 0 else 0
    else:
        cohens_d = 0

    # ── figure ────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    fig.suptitle(
        f'SimGNN GED Score Distribution by Client Type  (Round {round_num})\n'
        f'Causal PoR Defense — ASIA Dataset  (N = {len(scores_map)} clients)',
        fontsize=13, fontweight='bold', y=1.02
    )

    # ── Left: 4-group violin ──────────────────────────────────────────────────
    labels   = list(groups.keys())
    data     = list(groups.values())
    colours  = [C['honest'], C['hon_light'], C['adv_light'], C['adv']]
    colours  = colours[:len(labels)]

    rng = np.random.default_rng(42)
    vp  = ax1.violinplot(data, positions=range(len(data)),
                         showmedians=True, showextrema=True, widths=0.68)

    for i, (body, col) in enumerate(zip(vp['bodies'], colours)):
        body.set_facecolor(col); body.set_alpha(0.60)
        body.set_edgecolor(C['border']); body.set_linewidth(1)

    vp['cmedians'].set_colors(C['text']); vp['cmedians'].set_linewidth(2.2)
    vp['cbars'].set_colors(C['subtext']);  vp['cmins'].set_colors(C['subtext'])
    vp['cmaxes'].set_colors(C['subtext'])

    for i, (d, col) in enumerate(zip(data, colours)):
        jitter = rng.uniform(-0.13, 0.13, size=len(d))
        ax1.scatter(i + jitter, d, color=col, s=42, alpha=0.9,
                    zorder=4, edgecolors=C['border'], linewidths=0.5)

    if tau is not None:
        ax1.axhline(tau, color=C['threshold'], lw=2, ls='--',
                    zorder=5, label=f'Decision threshold τ ≈ {tau:.3f}')
        ax1.legend(fontsize=9, framealpha=1.0,
                   loc='upper right',
                   bbox_to_anchor=(0.99, 0.99))

    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, fontsize=9.5)
    ax1.set_ylabel('SimGNN Predicted GED Score', labelpad=8)
    ax1.set_title('Score Distribution by Classification Outcome', fontsize=11, pad=12)
    note(ax1, 'TP=Adv. Rejected  FP=Honest Rejected  FN=Adv. Accepted')

    # ── Right: 2-group boxplot ────────────────────────────────────────────────
    bp = ax2.boxplot(
        [honest_all, adv_all],
        positions=[0, 1], patch_artist=True, widths=0.48,
        boxprops=dict(linewidth=1.5),
        whiskerprops=dict(linewidth=1.5, color=C['subtext']),
        capprops=dict(linewidth=2, color=C['subtext']),
        medianprops=dict(color='white', linewidth=2.5),
        flierprops=dict(marker='o', markersize=5, alpha=0.6,
                        markerfacecolor=C['neutral'], markeredgewidth=0)
    )
    bp['boxes'][0].set_facecolor(C['honest']); bp['boxes'][0].set_alpha(0.72)
    bp['boxes'][1].set_facecolor(C['adv']);    bp['boxes'][1].set_alpha(0.72)

    for i, (d, col) in enumerate([(honest_all, C['honest']), (adv_all, C['adv'])]):
        jitter = rng.uniform(-0.14, 0.14, size=len(d))
        ax2.scatter(i + jitter, d, color=col, s=38, alpha=0.65,
                    zorder=3, edgecolors='white', linewidths=0.4)
        # mean diamond
        ax2.scatter(i, np.mean(d), color='white', zorder=6,
                    s=65, edgecolors=C['text'], linewidths=1.5,
                    marker='D', label=f'Mean = {np.mean(d):.3f}' if i==0 else f'Mean = {np.mean(d):.3f}')

    if tau is not None:
        ax2.axhline(tau, color=C['threshold'], lw=2, ls='--',
                    label=f'τ ≈ {tau:.3f}')

    ax2.set_xticks([0, 1])
    ax2.set_xticklabels([
        f'Honest Clients\n(n = {len(honest_all)})',
        f'Adversary Clients\n(n = {len(adv_all)})'
    ], fontsize=10)
    ax2.set_ylabel('SimGNN Predicted GED Score', labelpad=8)
    ax2.set_title('Honest vs. Adversary Score Comparison', fontsize=11, pad=12)

    # Cohen's d annotation — placed top-LEFT where honest (low-GED) region is empty
    ax2.text(0.02, 0.97,
             f"Cohen's d = {cohens_d:.2f}\n"
             f"({'Large' if cohens_d>0.8 else 'Medium' if cohens_d>0.5 else 'Small'} effect)",
             transform=ax2.transAxes, fontsize=9,
             ha='left', va='top', color=C['text'],
             bbox=dict(boxstyle='round,pad=0.45', fc='white',
                       ec=C['border'], alpha=1.0))
    ax2.legend(fontsize=8.5, loc='lower right')
    note(ax2, 'SimGNN [Bai et al., WSDM 2019]')

    plt.tight_layout()
    save_fig(fig, 'G02_ged_score_distribution.png')

if __name__ == '__main__':
    main()
    print("G02 done.")
