"""
visualizations/g13_rejected_edge_diff
==================================
G13: Rejected Client Graph Edge Difference Analysis.
Shows exactly which edges the adversarial client is missing or adding
compared to the consensus, making the poisoning attack structurally visible.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from style_config import (apply_style, save_fig, C,
                           load_rejected_edge_diff,
                           load_consensus_graph,
                           ASIA_GT_EDGES, ASIA_GT_NODES, note)
try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False

ASIA_POS = {
    'asia':   (0.0, 1.0), 'smoke': (1.2, 1.0),
    'tub':    (0.0, 0.6), 'lung':  (1.0, 0.6),
    'bronc':  (1.4, 0.6), 'either':(0.5, 0.3),
    'xray':   (0.2, 0.0), 'dysp':  (0.9, 0.0),
}

def draw_diff_graph(ax, consensus_edges, missing, extra, title):
    if not HAS_NX:
        ax.text(0.5, 0.5, 'networkx not available',
                ha='center', va='center', transform=ax.transAxes)
        return

    G = nx.DiGraph()
    G.add_nodes_from(ASIA_GT_NODES)
    for e in consensus_edges:
        G.add_edge(*e, kind='consensus')
    for e in missing:
        if len(e) == 2:
            G.add_edge(e[0], e[1], kind='missing')

    ax.set_facecolor(C['panel'])
    ax.axis('off')
    ax.set_title(title, fontsize=10, pad=10, fontweight='bold')

    xs = [p[0] for p in ASIA_POS.values()]
    ys = [p[1] for p in ASIA_POS.values()]
    ax.set_xlim(min(xs)-0.2, max(xs)+0.2)
    ax.set_ylim(min(ys)-0.2, max(ys)+0.3)

    # draw all consensus edges (present in rejected client too)
    missing_set = set(tuple(e) for e in missing)
    extra_set   = set(tuple(e) for e in extra)
    cons_set    = set(tuple(e) for e in consensus_edges)

    for u, v in cons_set:
        if (u, v) in missing_set or u not in ASIA_POS or v not in ASIA_POS:
            continue
        x0, y0 = ASIA_POS[u]; x1, y1 = ASIA_POS[v]
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='->', color=C['honest'],
                                    lw=2.0, alpha=0.7,
                                    connectionstyle='arc3,rad=0.05'))

    # draw missing edges (red dashed — adversary lost these)
    for e in missing_set:
        if len(e) != 2 or e[0] not in ASIA_POS or e[1] not in ASIA_POS:
            continue
        x0, y0 = ASIA_POS[e[0]]; x1, y1 = ASIA_POS[e[1]]
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='->', color=C['reject'],
                                    lw=2.2, alpha=0.85, linestyle='dashed',
                                    connectionstyle='arc3,rad=0.05'))

    # draw extra edges (orange — adversary added these spuriously)
    for e in extra_set:
        if len(e) != 2 or e[0] not in ASIA_POS or e[1] not in ASIA_POS:
            continue
        x0, y0 = ASIA_POS.get(e[0], (0.5,0.5))
        x1, y1 = ASIA_POS.get(e[1], (0.5,0.5))
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='->', color=C['adv'],
                                    lw=2.2, alpha=0.85,
                                    connectionstyle='arc3,rad=-0.05'))

    # nodes
    for n, (x, y) in ASIA_POS.items():
        circle = plt.Circle((x, y), 0.10, color=C['por'],
                             ec='white', lw=1.5, zorder=4)
        ax.add_patch(circle)
        ax.text(x, y, n, ha='center', va='center', fontsize=8,
                fontweight='bold', color='white', zorder=5)

def main():
    apply_style()
    diff = load_rejected_edge_diff()
    if diff is None:
        print("  ✗  No rejected edge diff data"); return

    ged_score      = diff.get('ged_score', '?')
    # Note: 'threshold' in rejected_edge_diff.json is the consensus add_ratio
    # parameter (controls momentum update), NOT the GED detection threshold τ.
    # The detection threshold τ is computed adaptively from score percentiles.
    add_ratio      = diff.get('threshold', '?')   # e.g. 0.4 = consensus_add_ratio
    missing_raw    = diff.get('missing_edges', [])
    extra_raw      = diff.get('extra_edges', [])

    # parse edge lists: stored as ["(u, v)", ...] or [["u","v"], ...]
    def parse_edges(raw):
        edges = []
        for e in raw:
            if isinstance(e, (list, tuple)) and len(e) == 2:
                edges.append((str(e[0]), str(e[1])))
            elif isinstance(e, str):
                e = e.strip("()' \"")
                parts = [p.strip().strip("'\"") for p in e.split(',')]
                if len(parts) == 2:
                    edges.append((parts[0], parts[1]))
        return edges

    missing = parse_edges(missing_raw)
    extra   = parse_edges(extra_raw)

    cg = load_consensus_graph()
    cons_edges = list(cg.edges()) if cg else ASIA_GT_EDGES

    # ── figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 8))
    fig.suptitle(
        f'Adversarial Client Graph Structural Deviation\n'
        f'SimGNN GED Score = {ged_score}  (exceeds adaptive percentile threshold τ)  →  CLIENT REJECTED',
        fontsize=13, fontweight='bold', y=1.02
    )

    ax_graph = fig.add_subplot(1, 2, 1)
    ax_stats  = fig.add_subplot(1, 2, 2)

    # ── Left: graph overlay ────────────────────────────────────────────────────
    draw_diff_graph(ax_graph, cons_edges, missing, extra,
                    'Structural Diff: Adversary vs. Consensus Graph')

    patches = [
        mpatches.Patch(color=C['honest'], label='Correct edges (present in adversary)'),
        mpatches.Patch(color=C['reject'], label=f'Missing edges ({len(missing)}) — feature poisoning removed these'),
        mpatches.Patch(color=C['adv'],    label=f'Spurious edges ({len(extra)}) — adversary injected these'),
    ]
    ax_graph.legend(handles=patches, loc='upper right', fontsize=8.5, framealpha=0.92)
    note(ax_graph, 'FalseNode: feature_0 zeroed → NOTEARS finds no causal links from/to feature_0')

    # ── Right: textual stats + impact ─────────────────────────────────────────
    ax_stats.set_facecolor(C['panel'])
    ax_stats.axis('off')
    ax_stats.set_title('Edge-Level Deviation Statistics', fontsize=11, pad=12, fontweight='bold')

    lines = [
        ('SimGNN GED Score',       f'{ged_score}', C['reject']),
        ('Detection',              'Adaptive percentile τ', C['threshold']),
        ('Consensus add_ratio',    f'{add_ratio}  (momentum param)', C['subtext']),
        ('Decision',               'REJECTED  ✗', C['reject']),
        ('', '', C['text']),
        ('Consensus edges',    f'{len(cons_edges)}', C['honest']),
        ('Missing from adversary', f'{len(missing)}', C['reject']),
        ('Extra in adversary', f'{len(extra)}', C['adv']),
        ('', '', C['text']),
        ('Attack mechanism',   'Feature poisoning', C['text']),
        ('Effect',             'feature[0] = 0 → no variance\n→ NOTEARS drops all edges\ninvolving feature[0]', C['text']),
        ('Why PoR detects it', 'Structural diff > τ\n→ topology audit fails', C['por']),
        ('Baseline detection', 'MISSED — weights look normal\nto cosine-similarity filter', C['base']),
    ]

    y = 0.93
    for label, val, col in lines:
        if label == '':
            y -= 0.03
            continue
        ax_stats.text(0.04, y, f'{label}:', transform=ax_stats.transAxes,
                      fontsize=10, va='top', color=C['subtext'], fontweight='bold')
        ax_stats.text(0.48, y, val, transform=ax_stats.transAxes,
                      fontsize=10, va='top', color=col, fontweight='bold')
        y -= 0.09

    # missing edges list
    if missing:
        y -= 0.01
        ax_stats.text(0.04, y, 'Missing edge details:', transform=ax_stats.transAxes,
                      fontsize=9, color=C['reject'], fontweight='bold', va='top')
        y -= 0.06
        for e in missing[:6]:
            ax_stats.text(0.08, y, f'• {e[0]}  →  {e[1]}',
                          transform=ax_stats.transAxes,
                          fontsize=9, color=C['reject'], va='top')
            y -= 0.055
        if len(missing) > 6:
            ax_stats.text(0.08, y, f'  + {len(missing)-6} more...',
                          transform=ax_stats.transAxes,
                          fontsize=9, color=C['subtext'], va='top', style='italic')

    note(ax_stats, 'DBA [Xie et al., ICLR 2020]  |  NOTEARS [Zheng et al., NeurIPS 2018]')

    plt.tight_layout()
    save_fig(fig, 'G13_rejected_edge_diff.png')

if __name__ == '__main__':
    main()
    print("G13 done.")
