"""
graphs/g05_cumulative_suppression.py
======================================
G05: Cumulative Adversary Suppression Curve — PoR vs. Baseline.
Shows running total of adversarial submissions blocked across rounds.
Also shows Honest Client Participation Rate (HCPR) as a second axis.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
from style_config import (apply_style, save_fig, C,
                           load_por_logs, load_baseline_logs,
                           parse_rounds, NUM_CLIENTS,
                           NUM_FALSE_NODES, NUM_HONEST, note)

def main():
    apply_style()
    por_log = load_por_logs()
    bas_log = load_baseline_logs()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 11), sharex=False)
    fig.suptitle(
        'Cumulative Adversary Suppression & Honest Client Participation\n'
        'Causal PoR Defense vs. Baseline FedAvg  (ASIA Dataset)',
        fontsize=13, fontweight='bold', y=1.01
    )

    # ─── Top panel: Cumulative rejection comparison ─────────────────────────
    for log, lbl, col, ls, marker in [
        (por_log, 'Causal PoR (FedNEAT)', C['por'],  '-',  'o'),
        (bas_log, 'Baseline FedAvg',       C['base'], '--', 's'),
    ]:
        if log is None:
            continue
        r_rej, v_rej = parse_rounds(log, 'rejected_clients')
        # skip round 0 (warm-up)
        pairs = [(r, v) for r, v in zip(r_rej, v_rej) if r > 0]
        if not pairs:
            continue
        rounds = [p[0] for p in pairs]
        cumul  = np.cumsum([p[1] for p in pairs])

        ax1.plot(rounds, cumul, color=col, lw=2.5, ls=ls,
                 marker=marker, markersize=8, markeredgecolor='white',
                 markeredgewidth=1.2, label=lbl, zorder=3)
        ax1.fill_between(rounds, cumul, alpha=0.08, color=col)

        # annotate final value
        ax1.annotate(f'  {int(cumul[-1])} total',
                     xy=(rounds[-1], cumul[-1]),
                     xytext=(rounds[-1] - 0.5, cumul[-1] + 1.5),
                     fontsize=9.5, color=col, fontweight='bold')

    # Maximum possible adversarial submissions (if all rounds participated)
    if por_log:
        nr = por_log['num_rounds']
        total_adv = NUM_FALSE_NODES * nr
        ax1.axhline(total_adv, color=C['neutral'], lw=1.2, ls=':',
                    label=f'Max. possible adversary submissions = {total_adv}')

    ax1.set_xlabel('Federated Round', labelpad=8)
    ax1.set_ylabel('Cumulative Clients Rejected', labelpad=8)
    ax1.set_title('Cumulative Total Rejections Across Rounds', fontsize=11, pad=12)
    ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax1.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax1.legend(fontsize=9.5, loc='upper left')
    note(ax1, 'Note: includes both TP (adversary) and FP (honest) rejections')

    # ─── Bottom panel: Honest Client Participation Rate ─────────────────────
    for log, lbl, col, ls, marker in [
        (por_log, 'PoR: Honest Participation Rate', C['por'],  '-',  'o'),
        (bas_log, 'Baseline: Honest Participation Rate', C['base'], '--', 's'),
    ]:
        if log is None:
            continue
        r_acc, v_acc = parse_rounds(log, 'accepted_clients')
        r_rej, v_rej = parse_rounds(log, 'rejected_clients')

        rounds = sorted(set(r_acc) | set(r_rej))
        acc_m  = dict(zip(r_acc, v_acc))
        rej_m  = dict(zip(r_rej, v_rej))
        hcpr   = []
        plot_r = []
        for r in rounds:
            if r == 0:
                continue
            accepted = acc_m.get(r, 0)
            # lower bound honest accepted = max(0, accepted - NUM_FALSE_NODES)
            # (assumes all adversaries were accepted before honest were rejected)
            hon_acc = max(0, accepted - NUM_FALSE_NODES)
            rate    = hon_acc / NUM_HONEST * 100
            hcpr.append(rate)
            plot_r.append(r)

        ax2.plot(plot_r, hcpr, color=col, lw=2.5, ls=ls,
                 marker=marker, markersize=8, markeredgecolor='white',
                 markeredgewidth=1.2, label=lbl, zorder=3)
        ax2.fill_between(plot_r, hcpr, alpha=0.07, color=col)

    ax2.axhline(100, color=C['accept'], lw=1.4, ls=':',
                label='100% participation (ideal)')
    ax2.axhline(70,  color=C['adv'], lw=1.2, ls='--', alpha=0.5,
                label='70% — minimum acceptable threshold')

    ax2.set_xlabel('Federated Round', labelpad=8)
    ax2.set_ylabel('Honest Client Participation Rate (%)', labelpad=8)
    ax2.set_title('Honest Client Participation Rate per Round\n'
                  '(Lower bound estimate based on total acceptance counts)',
                  fontsize=11, pad=12)
    ax2.set_ylim(-5, 108)
    ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax2.legend(fontsize=9.5, loc='lower left')
    note(ax2, 'SignGuard [Xu et al., NeurIPS 2022]  |  FLAME [Nguyen et al., USENIX 2022]')

    plt.tight_layout()
    save_fig(fig, 'G05_cumulative_suppression.png')

if __name__ == '__main__':
    main()
    print("G05 done.")
