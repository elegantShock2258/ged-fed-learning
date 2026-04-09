"""
graphs/g10_genome_architecture.py
====================================
G10: FedNEAT Evolved Genome Architecture — Network Graph Visualization.
Left: evolved genome topology (inputs → hidden → outputs with connections).
Right: parameter count comparison FedNEAT vs. fixed FedAvg MLP.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from style_config import apply_style, save_fig, C, load_genome, note

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False

def build_genome_graph(genome):
    """Build a directed graph from genome connection data."""
    G = nx.DiGraph()
    nodes = genome.get('nodes', {})
    conns = genome.get('connections', {})

    for nid, ntype in nodes.items():
        G.add_node(nid, ntype=ntype)

    for innov, c in conns.items():
        if c.get('active', True):
            G.add_edge(c['in'], c['out'],
                       weight=abs(float(c['weight'])),
                       raw_weight=float(c['weight']))
    return G

def layered_positions(nodes_dict):
    """Assign x,y positions by layer: input | hidden | output."""
    inputs  = [n for n, t in nodes_dict.items() if t == 'input']
    outputs = [n for n, t in nodes_dict.items() if t == 'output']
    hiddens = [n for n, t in nodes_dict.items() if t == 'hidden']

    pos = {}
    # inputs at x=0
    for i, n in enumerate(sorted(inputs)):
        pos[n] = (0.0, i / max(len(inputs)-1, 1))
    # outputs at x=1
    for i, n in enumerate(sorted(outputs)):
        pos[n] = (1.0, 0.2 + i * 0.6)
    # hiddens spread across x=0.2..0.8 in a grid
    cols = 4
    for i, n in enumerate(sorted(hiddens)):
        col = i % cols
        row = i // cols
        x = 0.20 + col * (0.60 / max(cols-1, 1))
        n_rows = (len(hiddens) + cols - 1) // cols
        y = row / max(n_rows - 1, 1)
        pos[n] = (x, y)
    return pos

def main():
    apply_style()

    if not HAS_NX:
        print("  ✗  networkx not available"); return

    genome = load_genome()
    if genome is None:
        print("  ✗  No genome data"); return

    G   = build_genome_graph(genome)
    pos = layered_positions(genome.get('nodes', {}))

    nodes  = genome.get('nodes', {})
    inputs  = [n for n, t in nodes.items() if t == 'input']
    outputs = [n for n, t in nodes.items() if t == 'output']
    hiddens = [n for n, t in nodes.items() if t == 'hidden']

    # active connections
    conns = genome.get('connections', {})
    active_conns = [c for c in conns.values() if c.get('active', True)]
    inactive_conns = [c for c in conns.values() if not c.get('active', True)]
    n_params_neat = len(active_conns)

    # FedAvg MLP params: Linear(7,32)+bias + Linear(32,16)+bias + Linear(16,2)+bias
    mlp_layers = [(7,32), (32,16), (16,2)]
    n_params_mlp = sum(i*o + o for i, o in mlp_layers)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9),
                                    gridspec_kw={'width_ratios': [2.5, 1]})
    fig.suptitle(
        'FedNEAT Evolved Global Genome Architecture\n'
        'Topological Crossover Result — Causal PoR Defense',
        fontsize=13, fontweight='bold', y=1.02
    )

    # ── Left: network graph ───────────────────────────────────────────────────
    ax1.set_facecolor(C['panel'])
    ax1.set_xlim(-0.08, 1.08)
    ax1.set_ylim(-0.08, 1.10)
    ax1.axis('off')
    ax1.set_title(
        f'Evolved Genome Topology\n'
        f'{len(inputs)} inputs  ·  {len(hiddens)} hidden nodes  ·  '
        f'{len(outputs)} outputs  ·  {len(active_conns)} active connections',
        fontsize=11, pad=12
    )

    # Draw edges first (behind nodes)
    weights = [abs(c['weight']) for c in active_conns]
    w_max   = max(weights) if weights else 1.0

    for c in active_conns:
        src, dst = c['in'], c['out']
        if src not in pos or dst not in pos:
            continue
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        raw_w = float(c.get('weight', 0.0))
        w     = abs(raw_w)
        alpha = 0.25 + 0.55 * (w / w_max)
        col   = C['por'] if raw_w >= 0 else C['base']
        lw    = 0.4 + 1.2 * (w / w_max)
        ax1.annotate('', xy=(x1, y1), xytext=(x0, y0),
                     arrowprops=dict(arrowstyle='->', color=col,
                                     alpha=alpha, lw=lw,
                                     connectionstyle='arc3,rad=0.05'))

    # Draw nodes
    node_radius = 0.022
    for n, (x, y) in pos.items():
        ntype = nodes.get(n, 'hidden')
        if ntype == 'input':
            colour, ec, zord = C['honest'], C['text'], 5
        elif ntype == 'output':
            colour, ec, zord = C['por'], C['text'], 5
        else:
            colour, ec, zord = C['adv_light'], C['adv'], 4

        circle = plt.Circle((x, y), node_radius, color=colour,
                             ec=ec, lw=1.0, zorder=zord)
        ax1.add_patch(circle)
        # label
        short = n.replace('hid_', 'h').replace('in_', 'i').replace('out_', 'o')
        ax1.text(x, y, short, ha='center', va='center',
                 fontsize=5.5, fontweight='bold',
                 color='white' if ntype != 'hidden' else C['text'],
                 zorder=zord+1)

    # Layer labels
    for lbl, xv in [('Input Layer', 0.0), ('Hidden Layer', 0.5), ('Output Layer', 1.0)]:
        ax1.text(xv, 1.07, lbl, ha='center', va='bottom',
                 fontsize=9.5, fontweight='bold', color=C['subtext'],
                 transform=ax1.transData)

    # Legend for edges
    leg_items = [
        mpatches.Patch(color=C['por'],  alpha=0.7, label='Positive weight'),
        mpatches.Patch(color=C['base'], alpha=0.7, label='Negative weight'),
        mpatches.Patch(color=C['honest'],    label='Input node'),
        mpatches.Patch(color=C['adv_light'], label='Hidden node'),
        mpatches.Patch(color=C['por'],       label='Output node'),
    ]
    ax1.legend(handles=leg_items, loc='lower right',
               fontsize=8.5, framealpha=0.92, edgecolor=C['border'])
    note(ax1, 'NEAT [Stanley & Miikkulainen, Evol. Comp. 2002]  |  Innovation Hash Crossover')

    # ── Right: parameter comparison bar ──────────────────────────────────────
    methods = ['FedAvg\nFixed MLP\n(3 layers)', 'FedNEAT\nEvolved Genome\n(variable)']
    param_counts = [n_params_mlp, n_params_neat]
    colors = [C['base'], C['por']]

    bars = ax2.bar(methods, param_counts, width=0.45,
                   color=colors, alpha=0.85, zorder=3,
                   edgecolor='white', linewidth=1.0)

    for bar, val in zip(bars, param_counts):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 5,
                 f'{val}\nparams', ha='center', va='bottom',
                 fontsize=11, fontweight='bold', color=C['text'])

    # Architecture description
    ax2.text(0, n_params_mlp / 2,
             f'7→32→16→2\n(fixed)',
             ha='center', va='center', fontsize=9,
             color='white', fontweight='bold')
    ax2.text(1, n_params_neat / 2,
             f'{len(inputs)}i+{len(hiddens)}h+{len(outputs)}o\n(evolved)',
             ha='center', va='center', fontsize=9,
             color='white', fontweight='bold')

    ax2.set_ylabel('Learnable Parameter Count', labelpad=8)
    ax2.set_title('Parameter Efficiency\nFedNEAT vs. Fixed Architecture',
                  fontsize=11, pad=12)
    ax2.set_ylim(0, max(param_counts) * 1.25)
    ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # difference annotation
    diff_pct = (n_params_mlp - n_params_neat) / n_params_mlp * 100
    ax2.annotate(
        f'{diff_pct:.0f}% fewer\nparameters',
        xy=(1, n_params_neat), xytext=(0.6, (n_params_mlp + n_params_neat)/2),
        fontsize=9.5, color=C['por'], fontweight='bold',
        arrowprops=dict(arrowstyle='->', color=C['por'], lw=1.4)
    )

    note(ax2, f'FedNEAT: {len(active_conns)} active + {len(inactive_conns)} inactive connections')

    plt.tight_layout()
    save_fig(fig, 'G10_genome_architecture.png')

if __name__ == '__main__':
    main()
    print("G10 done.")
