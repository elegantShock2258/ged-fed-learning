"""
visualizations/g01_acceptance_bars
==============================
G01: Per-Round Client Acceptance/Rejection — PoR vs. Baseline FedAvg
Stacked bar chart with adversary reference line.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
from style_config import (apply_style, save_fig, C,
                           load_por_logs, load_baseline_logs,
                           parse_rounds, NUM_CLIENTS,
                           NUM_FALSE_NODES, NUM_HONEST, note)

def main():
    apply_style()
    por_log  = load_por_logs()
    bas_log  = load_baseline_logs()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
    fig.suptitle(
        'Per-Round Client Acceptance & Rejection\n'
        'Causal PoR (FedNEAT) vs. Baseline FedAvg + Cosine Similarity',
        fontsize=14, fontweight='bold', y=1.03
    )

    configs = [
        (por_log,  axes[0], 'Causal PoR Defense (FedNEAT)',           C['por'],  C['reject']),
        (bas_log,  axes[1], 'Baseline: FedAvg + Cosine-Sim Detection', C['base'], C['neutral']),
    ]

    for log, ax, title, c_acc, c_rej in configs:
        if log is None:
            ax.text(0.5, 0.5, 'Data not available',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=12, color=C['subtext'])
            ax.set_title(title, fontsize=12, pad=14)
            continue

        r_acc, v_acc = parse_rounds(log, 'accepted_clients')
        r_rej, v_rej = parse_rounds(log, 'rejected_clients')

        # ── align rounds ──────────────────────────────────────────────────
        rounds  = sorted(set(r_acc) | set(r_rej))
        acc_map = dict(zip(r_acc, v_acc))
        rej_map = dict(zip(r_rej, v_rej))
        acc = np.array([acc_map.get(r, 0) for r in rounds], dtype=float)
        rej = np.array([rej_map.get(r, 0) for r in rounds], dtype=float)

        x = np.arange(len(rounds))
        w = 0.62

        # ── bars ──────────────────────────────────────────────────────────
        b1 = ax.bar(x, rej, w, color=c_rej, alpha=0.72,
                    zorder=3, hatch='//', edgecolor='white', linewidth=0.5,
                    label='Rejected clients')
        b2 = ax.bar(x, acc, w, bottom=rej, color=c_acc, alpha=0.88, zorder=3,
                    label='Accepted clients')

        # ── adversary reference ───────────────────────────────────────────
        ax.axhline(NUM_FALSE_NODES, color=C['threshold'], lw=1.8,
                   ls='--', zorder=4,
                   label=f'{NUM_FALSE_NODES} adversarial clients present')

        # ── total line ────────────────────────────────────────────────────
        ax.axhline(NUM_CLIENTS, color=C['neutral'], lw=1.2,
                   ls=':', zorder=2, alpha=0.6,
                   label=f'Total = {NUM_CLIENTS} clients')

        # ── x-axis labels ─────────────────────────────────────────────────
        ax.set_xticks(x)
        ax.set_xticklabels([f'R{r}' for r in rounds],
                            rotation=45, ha='right', fontsize=9)

        ax.set_xlabel('Federated Round', labelpad=10)
        ax.set_ylabel('Number of Clients', labelpad=8)
        ax.set_title(title, fontsize=12, pad=14)
        ax.set_ylim(0, NUM_CLIENTS + 4)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        # ── annotation box ────────────────────────────────────────────────
        ax.annotate(
            f'{NUM_HONEST} Honest  +  {NUM_FALSE_NODES} Adversaries',
            xy=(0.02, 0.97), xycoords='axes fraction',
            fontsize=8.5, color=C['subtext'], va='top', style='italic',
            bbox=dict(boxstyle='round,pad=0.35', fc=C['panel'],
                      ec=C['border'], alpha=0.9)
        )
        ax.legend(loc='upper right', fontsize=9, framealpha=0.92)
        note(ax, 'ASIA dataset (30 clients, 5 adversaries)')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_fig(fig, 'G01_per_round_acceptance_bars.png')

if __name__ == '__main__':
    main()
    print("G01 done.")
