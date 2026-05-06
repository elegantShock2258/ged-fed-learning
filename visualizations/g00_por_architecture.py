"""
visualizations/g00_por_architecture.py
===================================
Generates the architectural diagram for the Causal Proof of Reasoning system.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from style_config import apply_style, save_fig, C

def main():
    apply_style()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    # Diagram drawing logic here
    # Background panel (shifted up to remove negative space)
    rect = mpatches.FancyBboxPatch((0, 3.5), 11, 6.5, facecolor=C['panel'], edgecolor=C['border'], alpha=0.5, zorder=0, boxstyle="round,pad=0.2")
    ax.add_patch(rect)
    
    # Title
    ax.text(5.5, 9.5, "Agentic Federated Learning: Causal Proof of Reasoning (PoR)", 
            ha='center', va='center', fontsize=16, fontweight='bold', color=C['por'])
            
    # Client Side
    ax.text(2, 8.5, "Client Side (FedNEAT Agents)", ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Draw Clients
    for i, y in enumerate([7.5, 6.0, 4.5]):
        color = C['honest'] if i < 2 else C['adv']
        label = f"Honest Client {i+1}" if i < 2 else "Adversary Client (Poisoned)"
        box = mpatches.FancyBboxPatch((0.5, y-0.4), 3, 0.8, facecolor=C['bg'], edgecolor=color, lw=2, zorder=2, boxstyle="round,pad=0.1")
        ax.add_patch(box)
        ax.text(2, y, label, ha='center', va='center', fontsize=10, color=color, fontweight='bold')
        
        # Local Latent Space
        ax.text(2, y-0.2, "Local Latent Forward Pass", ha='center', va='center', fontsize=8, color=C['subtext'])
        
        # Arrow to DAG
        ax.annotate("", xy=(3.8, y), xytext=(3.5, y), arrowprops=dict(arrowstyle="->", color=C['neutral'], lw=1.5))
        
        # DAG Extraction
        dag_box = mpatches.Rectangle((3.8, y-0.3), 1.5, 0.6, facecolor=C['bg'], edgecolor=C['neutral'], lw=1, zorder=2)
        ax.add_patch(dag_box)
        ax.text(4.55, y, "NOTEARS\nDAG Extraction", ha='center', va='center', fontsize=8)
        
        # Arrow to Server
        ax.annotate("", xy=(6.0, y), xytext=(5.3, y), arrowprops=dict(arrowstyle="->", color=C['neutral'], lw=1.5))

    # Server Side
    ax.text(8.5, 8.5, "Server Side (PoR Strategy)", ha='center', va='center', fontsize=14, fontweight='bold')
    
    server_box = mpatches.FancyBboxPatch((6.0, 4.0), 4.5, 4.0, facecolor=C['por_light'], edgecolor=C['por'], lw=2, zorder=1, alpha=0.3, boxstyle="round,pad=0.1")
    ax.add_patch(server_box)
    
    # Consensus Graph
    cons_box = mpatches.Rectangle((6.5, 7.0), 2.8, 0.8, facecolor=C['bg'], edgecolor=C['por'], lw=1.5, zorder=2)
    ax.add_patch(cons_box)
    ax.text(7.9, 7.4, "Global Consensus Graph\n(Momentum Updated)", ha='center', va='center', fontsize=10, fontweight='bold', color=C['por'])
    
    # Logic Validator
    val_box = mpatches.Rectangle((6.5, 5.5), 2.8, 1.0, facecolor=C['bg'], edgecolor=C['gold'], lw=2, zorder=2)
    ax.add_patch(val_box)
    ax.text(7.9, 6.0, "SimGNN Logic Validator\n(GED > τ threshold)", ha='center', va='center', fontsize=10, fontweight='bold', color=C['gold'])
    
    # Arrows inside server
    ax.annotate("", xy=(7.9, 6.5), xytext=(7.9, 7.0), arrowprops=dict(arrowstyle="<->", color=C['por'], lw=1.5))
    
    # Decisions (shifted left to fit inside the wider server box)
    ax.annotate("", xy=(8.7, 6.1), xytext=(9.2, 6.1), arrowprops=dict(arrowstyle="<-", color=C['accept'], lw=2))
    ax.text(9.8, 6.1, "Accept\n(Honest)", ha='center', va='center', fontsize=9, color=C['accept'], fontweight='bold')
    
    ax.annotate("", xy=(8.7, 5.7), xytext=(9.2, 5.7), arrowprops=dict(arrowstyle="<-", color=C['reject'], lw=2))
    ax.text(9.8, 5.7, "Reject\n(Adversary)", ha='center', va='center', fontsize=9, color=C['reject'], fontweight='bold')
    
    ax.set_xlim(-0.2, 11.2)
    ax.set_ylim(3.3, 10.2)
    
    save_fig(fig, "por_architecture.png")

if __name__ == '__main__':
    main()
    print("G00 done.")
