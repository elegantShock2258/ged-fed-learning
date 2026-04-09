"""
graphs/g07_loss_convergence.py
================================
G07: Training Loss Convergence — Baseline FedAvg across rounds.
Also shows PoR rejection-adjusted effective learning signal.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
from style_config import (apply_style, save_fig, C,
                           load_baseline_logs, load_por_logs,
                           parse_rounds, NUM_CLIENTS,
                           NUM_FALSE_NODES, note)

def main():
    apply_style()
    bas_log = load_baseline_logs()
    por_log = load_por_logs()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        'Training Loss Convergence & Aggregation Efficiency\n'
        'Baseline FedAvg vs. Causal PoR Defense  (ASIA Dataset)',
        fontsize=13, fontweight='bold', y=1.02
    )

    # ── Left: Loss curve (baseline has ground-truth loss; PoR estimated) ──────
    if bas_log:
        r_loss, v_loss = parse_rounds(bas_log, 'loss')
        rounds_b = np.array(r_loss)
        losses_b = np.array(v_loss)

        ax1.plot(rounds_b, losses_b, color=C['base'], lw=2.4,
                 marker='s', markersize=7, markeredgecolor='white',
                 markeredgewidth=1.2, label='Baseline FedAvg Loss', zorder=3)
        ax1.fill_between(rounds_b, losses_b, alpha=0.10, color=C['base'])

        # annotate convergence
        if len(losses_b) > 2:
            min_idx = np.argmin(losses_b)
            ax1.annotate(
                f'Min loss\n{losses_b[min_idx]:.2e}',
                xy=(rounds_b[min_idx], losses_b[min_idx]),
                xytext=(rounds_b[min_idx] + 1.2, losses_b[min_idx] * 12),
                fontsize=8.5, color=C['base'],
                arrowprops=dict(arrowstyle='->', color=C['base'], lw=1.3)
            )

    # PoR loss — not directly logged; show evidence from accepted client count
    # We use the number of accepted results as a proxy for learning signal quality
    if por_log:
        r_acc, v_acc = parse_rounds(por_log, 'accepted_clients')
        # Estimate: high acceptance ≈ more gradient signal ≈ lower effective loss
        # Shown as dashed placeholder with annotation
        eff_signal = [v / NUM_CLIENTS * 100 for v in v_acc]
        ax1_r = ax1.twinx()
        ax1_r.plot(r_acc, eff_signal, color=C['por'], lw=2.0, ls='-.',
                   marker='o', markersize=6, markeredgecolor='white',
                   markeredgewidth=1.2,
                   label='PoR: Effective gradient signal (%)', zorder=3, alpha=0.85)
        ax1_r.set_ylabel('PoR Effective Gradient Signal (%)', color=C['por'], labelpad=8)
        ax1_r.tick_params(axis='y', colors=C['por'])
        ax1_r.set_ylim(0, 115)
        ax1_r.spines['right'].set_visible(True)
        ax1_r.spines['right'].set_color(C['border'])
        ax1_r.legend(loc='center right', fontsize=9)

    ax1.set_xlabel('Federated Round', labelpad=10)
    ax1.set_ylabel('Cross-Entropy Loss', labelpad=8)
    ax1.set_title('Loss Convergence per Round', fontsize=11, pad=12)
    ax1.set_ylim(bottom=0)
    ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax1.legend(loc='upper right', fontsize=9)
    note(ax1, 'FedAvg [McMahan et al., AISTATS 2017]')

    # ── Right: Effective Aggregation Efficiency (EAE) ─────────────────────────
    for log, lbl, col, ls, mk in [
        (por_log, 'Causal PoR (FedNEAT)',       C['por'],  '-',  'o'),
        (bas_log, 'Baseline FedAvg + Cosine',   C['base'], '--', 's'),
    ]:
        if log is None:
            continue
        r_acc, v_acc = parse_rounds(log, 'accepted_clients')
        r_rej, v_rej = parse_rounds(log, 'rejected_clients')

        rounds  = sorted(set(r_acc) | set(r_rej))
        acc_m   = dict(zip(r_acc, v_acc))
        rej_m   = dict(zip(r_rej, v_rej))

        eae = []
        plot_r = []
        for r in rounds:
            if r == 0:
                continue
            a = acc_m.get(r, 0)
            rj = rej_m.get(r, 0)
            tot = a + rj
            eae.append(a / tot * 100 if tot > 0 else 0)
            plot_r.append(r)

        ax2.plot(plot_r, eae, color=col, lw=2.4, ls=ls,
                 marker=mk, markersize=7, markeredgecolor='white',
                 markeredgewidth=1.2, label=lbl, zorder=3)
        ax2.fill_between(plot_r, eae, alpha=0.08, color=col)

        mean_eae = np.mean(eae)
        ax2.axhline(mean_eae, color=col, lw=1.0, ls=':', alpha=0.6,
                    label=f'{lbl[:3]}... mean EAE = {mean_eae:.1f}%')

    # Ideal reference
    ax2.axhline(100, color=C['accept'], lw=1.2, ls=':', alpha=0.5,
                label='100% — all clients accepted')
    # Adversary fraction line
    adv_pct = NUM_FALSE_NODES / NUM_CLIENTS * 100
    ax2.axhline(100 - adv_pct, color=C['neutral'], lw=1.2, ls='--', alpha=0.5,
                label=f'If only adversaries rejected = {100-adv_pct:.0f}%')

    ax2.set_xlabel('Federated Round', labelpad=10)
    ax2.set_ylabel('Effective Aggregation Efficiency (%)\n'
                   'accepted / total submitted × 100', labelpad=8)
    ax2.set_title('Effective Aggregation Efficiency per Round\n'
                  '(Higher = more clients contributed to global model)', fontsize=11, pad=12)
    ax2.set_ylim(-5, 108)
    ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax2.legend(fontsize=8.5, loc='lower left')
    note(ax2, 'FedProx [Li et al., ICLR 2020]  |  Byzantine-SGD [Blanchard et al., 2017]')

    plt.tight_layout()
    save_fig(fig, 'G07_loss_convergence_efficiency.png')

if __name__ == '__main__':
    main()
    print("G07 done.")
