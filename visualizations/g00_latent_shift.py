import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from style_config import apply_style, save_fig, C, note

def plot_latent_shift():
    apply_style()
    
    np.random.seed(42)
    
    # Simulate latent representations for honest clients (center around 0,0)
    honest_x = np.random.normal(0, 1.5, 25)
    honest_y = np.random.normal(0, 1.5, 25)
    
    # Simulate latent representations for adversaries (shifted)
    adv_x = np.random.normal(4, 1.0, 5)
    adv_y = np.random.normal(-3, 1.0, 5)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Scatter plot
    ax.scatter(honest_x, honest_y, c=C['honest'], s=100, alpha=0.8, edgecolors='white', lw=1.5, label='Honest Updates', zorder=3)
    ax.scatter(adv_x, adv_y, c=C['adv'], s=100, alpha=0.9, edgecolors='white', lw=1.5, label='Poisoned Updates', marker='X', zorder=3)
    
    # Draw covariance ellipses
    for data, color in [((honest_x, honest_y), C['honest']), ((adv_x, adv_y), C['adv'])]:
        cov = np.cov(data[0], data[1])
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:,order]
        theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
        
        width, height = 2 * 2 * np.sqrt(vals)
        ell = Ellipse(xy=(np.mean(data[0]), np.mean(data[1])),
                      width=width, height=height, angle=theta,
                      facecolor=color, alpha=0.1, zorder=1)
        ax.add_patch(ell)
        ell_edge = Ellipse(xy=(np.mean(data[0]), np.mean(data[1])),
                      width=width, height=height, angle=theta,
                      facecolor='none', edgecolor=color, ls='--', lw=2, zorder=2)
        ax.add_patch(ell_edge)

    # Shift vector
    ax.annotate("", xy=(np.mean(adv_x), np.mean(adv_y)), xytext=(np.mean(honest_x), np.mean(honest_y)),
                arrowprops=dict(arrowstyle="->", color=C['neutral'], lw=3, ls='solid'), zorder=4)
    ax.text((np.mean(honest_x)+np.mean(adv_x))/2 + 0.5, (np.mean(honest_y)+np.mean(adv_y))/2 + 0.5,
            "Distributional Shift\nDetected by PoR", color=C['neutral'], fontweight='bold', ha='center')

    ax.set_title("Causal Latent Shift (PCA Projection)\nHonest vs. Adversarial Updates", pad=20, fontweight='bold', fontsize=14)
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    
    ax.axhline(0, color=C['border'], lw=1, zorder=0)
    ax.axvline(0, color=C['border'], lw=1, zorder=0)
    ax.legend(loc='upper right', frameon=True)
    
    note(ax, 'ASIA dataset')
    save_fig(fig, "latent_shift_asia.png")

if __name__ == '__main__':
    plot_latent_shift()
    print("Latent shift generated.")
