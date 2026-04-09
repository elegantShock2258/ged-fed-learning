"""
graphs/g12_ged_score_heatmap.py
=================================
G12: GED Score Scatter by Client (last round).
Colour-codes each client by type and shows decision threshold.
Also produces a summary comparison bar (PoR vs Baseline effectiveness).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
from style_config import (apply_style, save_fig, C,
                           load_ged_scores, is_adversary,
                           NUM_CLIENTS, NUM_FALSE_NODES, NUM_HONEST, note)

def main():
    apply_style()
    ged = load_ged_scores()
    if ged is None:
        print("  ✗  No GED data"); return

    round_num  = ged.get('round', '?')
    scores_map = ged['scores']

    # ── classify and sort ─────────────────────────────────────────────────────
    records = []
    for nid, info in scores_map.items():
        cid_int = int(nid) % NUM_CLIENTS
        adv     = is_adversary(nid)
        records.append(dict(
            nid=nid, cid=cid_int, score=info['score'],
            status=info['status'], is_adv=adv
        ))
    records.sort(key=lambda r: r['cid'])

    scores = np.array([r['score'] for r in records])
    cids   = np.array([r['cid']   for r in records])
    is_adv = np.array([r['is_adv'] for r in records])
    statuses = np.array([r['status'] for r in records])

    # threshold
    rej_s = [r['score'] for r in records if r['status'] == 'rejected']
    acc_s = [r['score'] for r in records if r['status'] == 'accepted']
    tau   = (min(rej_s) + max(acc_s)) / 2 if (rej_s and acc_s) else 0.83

    # ── figure ────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 11),
                                    gridspec_kw={'height_ratios': [2.5, 1]})
    fig.suptitle(
        f'Per-Client GED Score Analysis  (Round {round_num})\n'
        'Causal PoR Defense — ASIA Dataset',
        fontsize=13, fontweight='bold', y=1.01
    )

    # ── Top: scatter plot ─────────────────────────────────────────────────────
    category_style = {
        # (is_adv, status): (colour, marker, zorder, label)
        (True,  'rejected'):  (C['base'],   'X',  5, 'Adversary  ·  Rejected (TP)'),
        (True,  'accepted'):  (C['adv'],    '^',  5, 'Adversary  ·  Accepted (FN)'),
        (False, 'accepted'):  (C['honest'], 'o',  3, 'Honest     ·  Accepted (TN)'),
        (False, 'rejected'):  (C['adv_light'], 's', 4, 'Honest     ·  Rejected (FP)'),
    }

    seen_labels = set()
    for r in records:
        key   = (r['is_adv'], r['status'])
        style = category_style.get(key, (C['neutral'], 'o', 3, ''))
        col, mk, zo, lbl = style

        ax1.scatter(r['cid'], r['score'], color=col, marker=mk,
                    s=130, zorder=zo, edgecolors='white', linewidths=1.2,
                    label=lbl if lbl not in seen_labels else '_nolegend_',
                    alpha=0.90)
        seen_labels.add(lbl)

        # label client id
        ax1.text(r['cid'], r['score'] + 0.004, str(r['cid']),
                 ha='center', va='bottom', fontsize=6.5,
                 color=col, fontweight='bold')

    # threshold line
    ax1.axhline(tau, color=C['threshold'], lw=2.2, ls='--', zorder=6,
                label=f'Decision threshold τ ≈ {tau:.3f}')

    # shade regions
    ax1.axhspan(tau, scores.max() + 0.01, alpha=0.05, color=C['reject'],
                label='Rejection zone (GED > τ)')
    ax1.axhspan(scores.min() - 0.01, tau, alpha=0.05, color=C['accept'],
                label='Acceptance zone (GED ≤ τ)')

    # adversary zone highlight
    adv_cids = [r['cid'] for r in records if r['is_adv']]
    if adv_cids:
        ax1.axvspan(min(adv_cids) - 0.5, max(adv_cids) + 0.5,
                    alpha=0.06, color=C['adv'],
                    label='Adversary client range (CID 25–29)')

    ax1.set_xlabel('Client ID  (0 = first honest  …  29 = last adversary)', labelpad=10)
    ax1.set_ylabel('SimGNN GED Score', labelpad=8)
    ax1.set_title('GED Score per Client — Colour by Classification Outcome',
                  fontsize=11, pad=12)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.legend(loc='lower left', fontsize=8.5, ncol=2, framealpha=0.92)
    note(ax1, 'CID is computed as: int(FlowerNodeID) mod 30')

    # ── Bottom: binary decision bar ───────────────────────────────────────────
    bar_data = {
        'Adversary\nRejected\n(True Positive)':  sum(1 for r in records if r['is_adv'] and r['status']=='rejected'),
        'Adversary\nAccepted\n(False Negative)': sum(1 for r in records if r['is_adv'] and r['status']=='accepted'),
        'Honest\nAccepted\n(True Negative)':     sum(1 for r in records if not r['is_adv'] and r['status']=='accepted'),
        'Honest\nRejected\n(False Positive)':    sum(1 for r in records if not r['is_adv'] and r['status']=='rejected'),
    }
    bar_cols = [C['base'], C['adv_light'], C['honest'], C['adv']]
    keys = list(bar_data.keys())
    vals = list(bar_data.values())

    bars = ax2.bar(keys, vals, width=0.5, color=bar_cols,
                   alpha=0.85, zorder=3, edgecolor='white', linewidth=1.0)
    for bar, val in zip(bars, vals):
        if val > 0:
            ax2.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.1,
                     str(int(val)), ha='center', va='bottom',
                     fontsize=13, fontweight='bold', color=C['text'])

    ax2.set_ylabel('Count', labelpad=8)
    ax2.set_title('Detection Outcome Summary — Single Round Confusion Matrix',
                  fontsize=11, pad=12)
    ax2.set_ylim(0, max(vals) + 2.5)
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))

    # computed metrics
    TP = bar_data['Adversary\nRejected\n(True Positive)']
    FN = bar_data['Adversary\nAccepted\n(False Negative)']
    FP = bar_data['Honest\nRejected\n(False Positive)']
    TN = bar_data['Honest\nAccepted\n(True Negative)']
    P  = TP/(TP+FP) if (TP+FP) > 0 else 0
    R  = TP/(TP+FN) if (TP+FN) > 0 else 0
    F1 = 2*P*R/(P+R) if (P+R) > 0 else 0
    metrics_txt = f'Precision={P:.2f}   Recall={R:.2f}   F1={F1:.2f}   ADR={R*100:.0f}%   FPR={FP/(FP+TN)*100:.1f}%'
    ax2.text(0.5, 0.97, metrics_txt, transform=ax2.transAxes,
             ha='center', va='top', fontsize=9.5, color=C['text'],
             bbox=dict(boxstyle='round,pad=0.4', fc=C['panel'],
                       ec=C['border'], alpha=0.92))
    note(ax2, 'DBA [Xie et al., ICLR 2020]  |  FLAME [Nguyen et al., USENIX 2022]')

    plt.tight_layout()
    save_fig(fig, 'G12_ged_score_per_client.png')

if __name__ == '__main__':
    main()
    print("G12 done.")
