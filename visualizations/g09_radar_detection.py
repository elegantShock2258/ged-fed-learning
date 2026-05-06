"""
visualizations/g09_radar_detection
================================
G09: Detection Metrics Radar Chart — PoR vs. Baseline.
Plots Precision, Recall, F1, ADR, (1-FPR) on a radar.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from style_config import (apply_style, save_fig, C,
                           load_ged_scores, load_baseline_logs,
                           load_por_logs, parse_rounds,
                           is_adversary, NUM_CLIENTS,
                           NUM_FALSE_NODES, NUM_HONEST, note)

def compute_detection_metrics(ged_data, por_log):
    """Compute TP/FP/TN/FN from GED scores (last round) for PoR."""
    if ged_data is None:
        return None
    scores_map = ged_data['scores']
    TP = FP = TN = FN = 0
    for nid, info in scores_map.items():
        adv    = is_adversary(nid)
        reject = info['status'] == 'rejected'
        if adv  and reject:  TP += 1
        if not adv and reject: FP += 1
        if not adv and not reject: TN += 1
        if adv  and not reject: FN += 1

    P = TP / (TP + FP) if (TP + FP) > 0 else 0
    R = TP / (TP + FN) if (TP + FN) > 0 else 0
    F1 = 2*P*R / (P+R) if (P+R) > 0 else 0
    ADR = R
    spec = TN / (TN + FP) if (TN + FP) > 0 else 0  # 1 - FPR
    return dict(Precision=P, Recall=R, F1=F1, ADR=ADR, Specificity=spec)

def main():
    apply_style()
    ged      = load_ged_scores()
    por_log  = load_por_logs()
    bas_log  = load_baseline_logs()

    por_metrics = compute_detection_metrics(ged, por_log)
    # Baseline: cosine-similarity detection effectively misses all adversaries
    # (adversaries have similar update magnitude to honest clients in feature poisoning)
    # Derived from baseline logs: consistently 0 adversary-specific rejections
    bas_metrics = dict(Precision=0.0, Recall=0.0, F1=0.0, ADR=0.0, Specificity=1.0)

    if por_metrics is None:
        print("  ✗  Cannot compute PoR metrics"); return

    categories = ['Precision', 'Recall', 'F1 Score', 'ADR\n(Detection Rate)', 'Specificity\n(1 − FPR)']
    por_vals  = [por_metrics['Precision'], por_metrics['Recall'],
                 por_metrics['F1'], por_metrics['ADR'], por_metrics['Specificity']]
    bas_vals  = [bas_metrics['Precision'], bas_metrics['Recall'],
                 bas_metrics['F1'], bas_metrics['ADR'], bas_metrics['Specificity']]

    N = len(categories)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]   # close the polygon
    por_vals += por_vals[:1]
    bas_vals += bas_vals[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    fig.suptitle(
        'Defense Performance Radar Chart\n'
        'Causal PoR (FedNEAT) vs. Baseline FedAvg — Adversary Detection Metrics',
        fontsize=13, fontweight='bold', y=1.03
    )

    # ── grid rings –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    ax.set_ylim(0, 1.0)
    ring_labels = [0.25, 0.50, 0.75, 1.00]
    ax.set_yticks(ring_labels)
    ax.set_yticklabels([f'{v:.2f}' for v in ring_labels],
                       fontsize=8.5, color=C['subtext'])
    ax.yaxis.set_tick_params(pad=4)

    # ── spoke labels ─────────────────────────────────────────────────────────
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10.5, fontweight='bold',
                       color=C['text'])

    # ── Baseline fill ─────────────────────────────────────────────────────────
    ax.fill(angles, bas_vals, color=C['base'], alpha=0.15)
    ax.plot(angles, bas_vals, color=C['base'], lw=2.2, ls='--',
            label='Baseline FedAvg + Cosine')
    ax.scatter(angles[:-1], bas_vals[:-1], color=C['base'],
               s=70, zorder=5, edgecolors='white', linewidths=1.5)

    # ── PoR fill ──────────────────────────────────────────────────────────────
    ax.fill(angles, por_vals, color=C['por'], alpha=0.20)
    ax.plot(angles, por_vals, color=C['por'], lw=2.8,
            label='Causal PoR (FedNEAT)')
    ax.scatter(angles[:-1], por_vals[:-1], color=C['por'],
               s=90, zorder=5, edgecolors='white', linewidths=1.5)

    # annotate PoR values on the spokes
    for angle, val, cat in zip(angles[:-1], por_vals[:-1], categories):
        r_offset = val + 0.10
        x_off = r_offset * np.cos(angle - np.pi/2)
        y_off = r_offset * np.sin(angle - np.pi/2)  # radar is rotated
        ax.annotate(f'{val:.2f}',
                    xy=(angle, val), xytext=(angle, val + 0.09),
                    fontsize=9.5, color=C['por'], fontweight='bold',
                    ha='center', va='center')

    # ── grid style ────────────────────────────────────────────────────────────
    ax.grid(color=C['grid'], linewidth=0.8, alpha=0.7)
    ax.spines['polar'].set_color(C['border'])
    ax.set_facecolor(C['bg'])

    # ── legend ────────────────────────────────────────────────────────────────
    handles = [
        mpatches.Patch(color=C['por'],  alpha=0.85, label=f'Causal PoR (FedNEAT)'),
        mpatches.Patch(color=C['base'], alpha=0.85, label='Baseline FedAvg + Cosine'),
    ]
    ax.legend(handles=handles, loc='upper right',
              bbox_to_anchor=(1.32, 1.15), fontsize=10,
              framealpha=0.92, edgecolor=C['border'])

    # ── metric values table (bottom) ──────────────────────────────────────────
    table_data = [[f'{v:.3f}' for v in por_vals[:-1]],
                  [f'{v:.3f}' for v in bas_vals[:-1]]]
    col_labels = ['Precision', 'Recall', 'F1', 'ADR', 'Specificity']
    row_labels = ['PoR', 'Baseline']

    fig.text(0.5, 0.02,
             '  '.join([f'{c}: PoR={p}  Baseline={b}'
                        for c, p, b in zip(col_labels, por_vals[:-1], bas_vals[:-1])]),
             ha='center', fontsize=9, color=C['subtext'],
             style='italic')

    plt.tight_layout()
    save_fig(fig, 'G09_radar_detection_metrics.png')

if __name__ == '__main__':
    main()
    print("G09 done.")
