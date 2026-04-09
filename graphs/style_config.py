"""
graphs/style_config.py
======================
Shared visual style, constants, data loaders, and utilities
for all FYP graph generation scripts.
"""
import json
import pickle
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')

# ── Directory Setup ────────────────────────────────────────────────────────────
GRAPHS_DIR   = Path(__file__).parent
PROJECT_DIR  = GRAPHS_DIR.parent
SAVED_MODELS = PROJECT_DIR / "saved_models"

# ── Simulation Constants ───────────────────────────────────────────────────────
NUM_CLIENTS     = 30
NUM_FALSE_NODES = 5
NUM_HONEST      = NUM_CLIENTS - NUM_FALSE_NODES   # 25
ADV_THRESHOLD   = NUM_CLIENTS - NUM_FALSE_NODES   # cid >= 25 → adversary

# ASIA Bayesian Network ground-truth edges (Lauritzen & Spiegelhalter, 1988)
ASIA_GT_EDGES = [
    ('asia', 'tub'), ('tub', 'either'), ('smoke', 'lung'),
    ('smoke', 'bronc'), ('lung', 'either'), ('either', 'xray'),
    ('either', 'dysp'), ('bronc', 'dysp')
]
ASIA_GT_NODES = ['asia', 'tub', 'smoke', 'lung', 'bronc', 'either', 'xray', 'dysp']

# ── Color Palette ──────────────────────────────────────────────────────────────
C = {
    'por':        '#1D4ED8',   # deep blue  – PoR / FedNEAT
    'por_light':  '#BFDBFE',   # light blue
    'base':       '#B91C1C',   # deep red   – Baseline
    'base_light': '#FECACA',   # light red
    'honest':     '#166534',   # dark green – honest clients
    'hon_light':  '#A7F3D0',   # light green
    'adv':        '#92400E',   # dark amber – adversary clients
    'adv_light':  '#FDE68A',   # light amber
    'accept':     '#15803D',   # green (accepted)
    'reject':     '#DC2626',   # red   (rejected)
    'threshold':  '#7C3AED',   # purple (decision threshold)
    'neutral':    '#4B5563',   # dark-grey
    'grid':       '#E5E7EB',   # light grid
    'border':     '#D1D5DB',
    'bg':         '#FFFFFF',
    'panel':      '#F9FAFB',
    'text':       '#111827',
    'subtext':    '#6B7280',
    'gold':       '#B45309',
}

# ── Global Matplotlib Settings ─────────────────────────────────────────────────
def apply_style():
    plt.rcParams.update({
        'figure.dpi':              150,
        'savefig.dpi':             300,
        'figure.facecolor':        C['bg'],
        'axes.facecolor':          C['bg'],
        'axes.grid':               True,
        'grid.alpha':              0.45,
        'grid.color':              C['grid'],
        'grid.linewidth':          0.65,
        'axes.spines.top':         False,
        'axes.spines.right':       False,
        'axes.spines.left':        True,
        'axes.spines.bottom':      True,
        'axes.edgecolor':          C['border'],
        'axes.linewidth':          0.9,
        'axes.labelcolor':         C['text'],
        'axes.titlecolor':         C['text'],
        'text.color':              C['text'],
        'xtick.color':             C['subtext'],
        'ytick.color':             C['subtext'],
        'xtick.major.size':        4,
        'ytick.major.size':        4,
        'xtick.major.width':       0.8,
        'ytick.major.width':       0.8,
        'font.family':             'DejaVu Sans',
        'font.size':               11,
        'axes.titlesize':          13,
        'axes.titleweight':        'bold',
        'axes.titlepad':           14,
        'axes.labelsize':          11,
        'axes.labelpad':           8,
        'xtick.labelsize':         10,
        'ytick.labelsize':         10,
        'legend.fontsize':         10,
        'legend.framealpha':       0.92,
        'legend.edgecolor':        C['border'],
        'legend.fancybox':         True,
        'legend.borderpad':        0.6,
        'legend.labelspacing':     0.4,
        'lines.linewidth':         2.2,
        'lines.markersize':        7,
        'patch.linewidth':         0.8,
    })

# ── Save Helper ────────────────────────────────────────────────────────────────
def save_fig(fig, filename):
    path = GRAPHS_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches='tight',
                pad_inches=0.3, facecolor=C['bg'], edgecolor='none')
    plt.close(fig)
    print(f"  ✓  Saved: {filename}")
    return path

# ── Data Loaders ───────────────────────────────────────────────────────────────
def load_por_logs():
    path = SAVED_MODELS / "asia" / "simulation_logs.json"
    if not path.exists():
        return None
    with open(path) as f:
        runs = json.load(f)
    return max(runs, key=lambda x: x['num_rounds'])

def load_baseline_logs():
    path = SAVED_MODELS / "baseline" / "simulation_logs.json"
    if not path.exists():
        return None
    with open(path) as f:
        runs = json.load(f)
    # Pick the most-recent 15-round run with losses
    candidates = [r for r in runs if r.get('num_rounds', 0) == 15]
    return candidates[-1] if candidates else runs[-1]

def load_ged_scores():
    path = SAVED_MODELS / "asia" / "ged_scores.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)

def load_consensus_graph():
    path = SAVED_MODELS / "asia" / "consensus_graph.gpickle"
    if not path.exists():
        return None
    try:
        import networkx as nx
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None

def load_genome():
    path = SAVED_MODELS / "realtime_state.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)

def load_rejected_edge_diff():
    path = SAVED_MODELS / "asia" / "rejected_edge_diff.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)

# ── Utilities ──────────────────────────────────────────────────────────────────
def is_adversary(node_id_str):
    """Return True if this Flower node ID belongs to an adversarial client."""
    try:
        return int(node_id_str) % NUM_CLIENTS >= ADV_THRESHOLD
    except Exception:
        return False

def parse_rounds(log, key):
    """Extract (rounds_list, values_list) from a simulation log dict."""
    if log is None:
        return [], []
    items = log.get('metrics', {}).get(key, [])
    return [i['round'] for i in items], [i['value'] for i in items]

def note(ax, text, loc='lower right'):
    """Add a small italic source note to an axis."""
    x, y, ha = (0.99, 0.02, 'right') if 'right' in loc else (0.01, 0.02, 'left')
    ax.text(x, y, text, transform=ax.transAxes, fontsize=7.5,
            color=C['subtext'], ha=ha, va='bottom', style='italic')
