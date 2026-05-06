"""
visualizations/g11_asia_ground_truth
==================================
G11: ASIA Bayesian Network Ground Truth Visualization.
Shows the 8-node causal DAG used as the consensus reference,
alongside the PoR consensus graph for side-by-side comparison.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from style_config import (apply_style, save_fig, C,
                           load_consensus_graph,
                           ASIA_GT_EDGES, ASIA_GT_NODES, note)
try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False

# Fixed positions for ASIA BN (domain-meaningful layout)
ASIA_POS = {
    'asia':   (0.0,  1.0),
    'smoke':  (1.0,  1.0),
    'tub':    (0.0,  0.6),
    'lung':   (0.8,  0.6),
    'bronc':  (1.2,  0.6),
    'either': (0.4,  0.3),
    'xray':   (0.2,  0.0),
    'dysp':   (0.8,  0.0),
}

NODE_ROLES = {
    'asia':   ('Exogenous\n(Asia visit)',    C['neutral']),
    'smoke':  ('Exogenous\n(Smoker)',         C['neutral']),
    'tub':    ('Tuberculosis',                C['adv']),
    'lung':   ('Lung Cancer',                 C['base']),
    'bronc':  ('Bronchitis',                  C['gold']),
    'either': ('Tub or Cancer\n(v-structure)',C['por']),
    'xray':   ('X-Ray Result',               C['honest']),
    'dysp':   ('Dyspnoea',                   C['honest']),
}

def draw_dag(ax, G, pos, node_roles, title, show_labels=True, alpha=1.0):
    ax.set_facecolor(C['panel'])
    ax.axis('off')
    ax.set_title(title, fontsize=11, pad=12, fontweight='bold')

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    margin = 0.15
    ax.set_xlim(min(xs)-margin, max(xs)+margin)
    ax.set_ylim(min(ys)-margin, max(ys)+margin)

    # edges
    for u, v in G.edges():
        if u not in pos or v not in pos:
            continue
        x0, y0 = pos[u]; x1, y1 = pos[v]
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='->', color=C['text'],
                                    lw=1.8, alpha=0.75*alpha,
                                    connectionstyle='arc3,rad=0.05'))

    # nodes
    for n, (x, y) in pos.items():
        if n not in G.nodes():
            col = C['subtext']
            label_col = C['subtext']
        else:
            col = node_roles.get(n, ('', C['neutral']))[1]
            label_col = 'white'

        circle = plt.Circle((x, y), 0.09, color=col, ec='white',
                             lw=1.5, zorder=4, alpha=alpha)
        ax.add_patch(circle)
        ax.text(x, y, n, ha='center', va='center',
                fontsize=8.5, fontweight='bold', color=label_col,
                zorder=5, alpha=alpha)

        if show_labels and n in node_roles:
            role_text = node_roles[n][0]
            offset_y = -0.14 if y > 0.5 else 0.14
            ax.text(x, y + offset_y, role_text, ha='center',
                    va='top' if offset_y < 0 else 'bottom',
                    fontsize=7, color=C['subtext'],
                    style='italic', alpha=alpha)

def main():
    apply_style()
    if not HAS_NX:
        print("  ✗  networkx not available"); return

    G_gt = nx.DiGraph()
    G_gt.add_nodes_from(ASIA_GT_NODES)
    G_gt.add_edges_from(ASIA_GT_EDGES)

    G_cg = load_consensus_graph()

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(
        'ASIA Bayesian Network — Ground Truth vs. PoR Consensus Graph\n'
        'Causal Proof of Reasoning: Reference Structure for GED Validation',
        fontsize=13, fontweight='bold', y=1.02
    )

    # ── Left: Ground truth ────────────────────────────────────────────────────
    draw_dag(axes[0], G_gt, ASIA_POS, NODE_ROLES,
             'ASIA BN Ground Truth\n(Lauritzen & Spiegelhalter, 1988)\n'
             f'{G_gt.number_of_nodes()} nodes  ·  {G_gt.number_of_edges()} directed edges')

    # edge count + v-structure note
    axes[0].text(0.02, 0.02,
                 'Key v-structure: tub → either ← lung\n'
                 '(Collider — tests NOTEARS orientation)',
                 transform=axes[0].transAxes, fontsize=8.5,
                 color=C['por'], style='italic', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.4', fc=C['panel'],
                           ec=C['por'], alpha=0.9))
    note(axes[0], 'Used as server-side consensus anchor for GED validation')

    # ── Right: PoR Consensus ──────────────────────────────────────────────────
    if G_cg is not None:
        # use same node positions, add any missing nodes
        cg_pos = {n: ASIA_POS.get(n, (0.5, 0.5)) for n in G_cg.nodes()}
        draw_dag(axes[1], G_cg, cg_pos, NODE_ROLES,
                 'PoR Evolved Consensus Graph\n(After federated rounds)\n'
                 f'{G_cg.number_of_nodes()} nodes  ·  {G_cg.number_of_edges()} directed edges')

        # overlay: mark edges present in gt vs not
        cg_edge_set = set(G_cg.edges())
        gt_edge_set = set(ASIA_GT_EDGES)
        matched   = cg_edge_set & gt_edge_set
        spurious  = cg_edge_set - gt_edge_set
        missing   = gt_edge_set - cg_edge_set

        summary = (f'✓ Correct edges:  {len(matched)}/{len(gt_edge_set)}\n'
                   f'✗ Spurious edges: {len(spurious)}\n'
                   f'△ Missing edges:  {len(missing)}')
        axes[1].text(0.02, 0.02, summary,
                     transform=axes[1].transAxes, fontsize=8.5,
                     color=C['text'], family='monospace', va='bottom',
                     bbox=dict(boxstyle='round,pad=0.4', fc=C['panel'],
                               ec=C['border'], alpha=0.92))
        note(axes[1], 'Updated every round via momentum-blended majority vote')
    else:
        axes[1].text(0.5, 0.5, 'Consensus graph\nnot available',
                     ha='center', va='center', transform=axes[1].transAxes,
                     fontsize=12, color=C['subtext'])
        axes[1].set_title('PoR Consensus Graph', fontsize=11, pad=12)

    # shared node-type legend
    patches = [mpatches.Patch(color=NODE_ROLES[n][1], label=f'{n}: {NODE_ROLES[n][0].replace(chr(10)," ")}')
               for n in ASIA_GT_NODES]
    fig.legend(handles=patches, loc='lower center', ncol=4,
               fontsize=8.5, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    save_fig(fig, 'G11_asia_ground_truth_vs_consensus.png')

if __name__ == '__main__':
    main()
    print("G11 done.")
