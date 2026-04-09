"""
graphs/g06_simgnn_speedup.py
==============================
G06: SimGNN Speedup vs. Exact GED Methods (published benchmark values).
Log-scale bar chart comparing runtime and MSE across datasets.
Source: Bai et al., WSDM 2019 (Table 1 & Table 2).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from style_config import apply_style, save_fig, C, note

def main():
    apply_style()

    # ── Published benchmark data (Bai et al., WSDM 2019) ─────────────────────
    datasets = ['AIDS\n(Chemical)', 'LINUX\n(Dependency)', 'IMDB\n(Social)']

    # Runtime in seconds (IMDB A* = timeout → representd as 10000s)
    runtime = {
        'A* Exact':    [2174.0, 212.0, 10000.0],  # 10000 = Timeout
        'Beam Search': [15.6,   12.5,  None],
        'SimGNN':      [1.264,  0.878, 0.770],
    }
    mse = {  # ×10⁻³
        'A* Exact':    [0.0,   0.0,   None],
        'Beam Search': [12.0,  25.0,  None],
        'SimGNN':      [1.191, 0.735, 0.759],
    }
    speedups = [2174/1.264, 212/0.878, None]  # None for IMDB (timeout)

    method_colors = {
        'A* Exact':    C['neutral'],
        'Beam Search': C['adv_light'],
        'SimGNN':      C['por'],
    }
    hatches = {'A* Exact': '//', 'Beam Search': '\\\\', 'SimGNN': ''}

    n_datasets = len(datasets)
    n_methods  = 3
    x = np.arange(n_datasets)
    bar_w = 0.24

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        'SimGNN Performance Benchmarks vs. Exact GED Methods\n'
        'Source: Bai et al., WSDM 2019  —  Justification for SimGNN in Causal PoR',
        fontsize=13, fontweight='bold', y=1.02
    )

    # ── Left panel: Runtime (log scale) ──────────────────────────────────────
    offsets = [-bar_w, 0, bar_w]
    for i, (method, col) in enumerate([
        ('A* Exact', method_colors['A* Exact']),
        ('Beam Search', method_colors['Beam Search']),
        ('SimGNN', method_colors['SimGNN']),
    ]):
        vals = []
        for v in runtime[method]:
            vals.append(v if v is not None else 10000.0)

        bars = ax1.bar(x + offsets[i], vals, bar_w,
                       color=col, alpha=0.85, zorder=3,
                       hatch=hatches[method],
                       edgecolor='white' if method == 'SimGNN' else C['border'],
                       linewidth=0.6, label=method)

        for j, (bar, v) in enumerate(zip(bars, runtime[method])):
            if v is None:
                ax1.text(bar.get_x() + bar.get_width()/2, 12000,
                         'Timeout', ha='center', va='bottom',
                         fontsize=8, color=C['reject'], fontweight='bold', rotation=0)
            else:
                label = f'{v:.2f}s' if v < 1000 else f'{v:.0f}s'
                ax1.text(bar.get_x() + bar.get_width()/2, v * 1.15,
                         label, ha='center', va='bottom',
                         fontsize=8, color=C['text'], fontweight='bold')

    ax1.set_yscale('log')
    ax1.set_ylim(0.3, 50000)
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, fontsize=10)
    ax1.set_xlabel('Graph Dataset', labelpad=10)
    ax1.set_ylabel('Runtime (seconds, log scale)', labelpad=8)
    ax1.set_title('Computation Time Comparison\n(Lower is better)', fontsize=11, pad=12)
    ax1.legend(fontsize=9.5, loc='upper right')
    ax1.axhline(1.0, color=C['accept'], lw=1.2, ls='--', alpha=0.5,
                label='1 second reference')

    # speedup callouts for SimGNN
    for j, (sp, ds) in enumerate(zip(speedups, datasets)):
        if sp is not None:
            ax1.annotate(
                f'×{sp:,.0f}\nfaster',
                xy=(x[j] + bar_w, runtime['SimGNN'][j]),
                xytext=(x[j] + bar_w + 0.04, runtime['SimGNN'][j] * 8),
                fontsize=8.5, color=C['por'], fontweight='bold',
                ha='center',
                arrowprops=dict(arrowstyle='->', color=C['por'], lw=1.3)
            )
        else:
            ax1.text(x[j] + bar_w, runtime['SimGNN'][j] * 12,
                     '∞ faster\n(A* timed out)',
                     ha='center', fontsize=8.5,
                     color=C['por'], fontweight='bold')

    note(ax1, 'Bai et al., "SimGNN: A Neural Network Approach to Fast GED", WSDM 2019')

    # ── Right panel: MSE (×10⁻³) ─────────────────────────────────────────────
    for i, (method, col) in enumerate([
        ('A* Exact', method_colors['A* Exact']),
        ('Beam Search', method_colors['Beam Search']),
        ('SimGNN', method_colors['SimGNN']),
    ]):
        vals_raw = mse[method]
        vals = []
        mask = []
        for v in vals_raw:
            vals.append(v if v is not None else 0.0)
            mask.append(v is not None)

        bars = ax2.bar(x + offsets[i], vals, bar_w,
                       color=col, alpha=0.85, zorder=3,
                       hatch=hatches[method],
                       edgecolor='white' if method == 'SimGNN' else C['border'],
                       linewidth=0.6, label=method)

        for bar, v, m in zip(bars, vals_raw, mask):
            if not m:
                ax2.text(bar.get_x() + bar.get_width()/2, 0.5,
                         'N/A', ha='center', va='bottom',
                         fontsize=8, color=C['subtext'], rotation=0)
            elif v == 0.0:
                ax2.text(bar.get_x() + bar.get_width()/2, 0.3,
                         'Exact\n(0.000)', ha='center', va='bottom',
                         fontsize=7.5, color=C['text'])
            else:
                ax2.text(bar.get_x() + bar.get_width()/2, v + 0.3,
                         f'{v:.3f}', ha='center', va='bottom',
                         fontsize=8.5, color=C['text'], fontweight='bold')

    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets, fontsize=10)
    ax2.set_xlabel('Graph Dataset', labelpad=10)
    ax2.set_ylabel('Mean Squared Error  (×10⁻³)', labelpad=8)
    ax2.set_title('Approximation Error (MSE)\n(Lower is better — 0.0 = exact)', fontsize=11, pad=12)
    ax2.legend(fontsize=9.5, loc='upper right')
    note(ax2, 'MSE between predicted and true normalised GED scores')

    plt.tight_layout()
    save_fig(fig, 'G06_simgnn_speedup_benchmarks.png')

if __name__ == '__main__':
    main()
    print("G06 done.")
