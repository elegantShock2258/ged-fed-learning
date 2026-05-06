"""
experiments/latent_shift_exp.py
==========================
Experiment to validate the "Dormant Backdoor" assumption.
Compares the latent activations of a completely clean model vs a poisoned model
when both are fed the EXACT SAME CLEAN input data.

If a distribution shift exists, it proves that the backdoor inherently distorts
the latent manifold even when the trigger is not present.
"""
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

class MLP(nn.Module):
    def __init__(self, in_f=8, num_cls=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_f, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_cls)
        )
        
    def forward(self, x):
        for i in range(4):
            x = self.net[i](x)
        latents = x
        logits = self.net[4](latents)
        return logits, latents

def load_data(dataset, num_samples=2000, seed=42):
    from datasets.tabular_loader import TabularBNDataset
    ds = TabularBNDataset(name=dataset, num_samples=num_samples, seed=seed)
    return torch.tensor(ds.features, dtype=torch.float32), ds.targets

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="asia")
    parser.add_argument("--clean_model", type=str, required=True, help="Path to clean baseline model")
    parser.add_argument("--poisoned_model", type=str, required=True, help="Path to poisoned baseline model")
    args = parser.parse_args()

    print(f"Loading {args.dataset.upper()} clean data...")
    X, _ = load_data(args.dataset)
    in_features = X.shape[1]
    
    clean_net = MLP(in_f=in_features, num_cls=6)
    clean_net.load_state_dict(torch.load(args.clean_model, map_location="cpu"))
    clean_net.eval()
    
    poison_net = MLP(in_f=in_features, num_cls=6)
    poison_net.load_state_dict(torch.load(args.poisoned_model, map_location="cpu"))
    poison_net.eval()
    
    with torch.no_grad():
        _, clean_latents = clean_net(X)
        _, poison_latents = poison_net(X)
        
    clean_latents = clean_latents.numpy()
    poison_latents = poison_latents.numpy()
    
    # Compute distances
    dists = []
    for i in range(clean_latents.shape[1]):
        dist = wasserstein_distance(clean_latents[:, i], poison_latents[:, i])
        dists.append(dist)
        
    mean_w_dist = np.mean(dists)
    print(f"\nMean Wasserstein Distance across latent dimensions: {mean_w_dist:.4f}")
    
    if mean_w_dist > 0.05:
        print("[SUCCESS] Significant latent distribution shift detected. The backdoor permanently deforms the manifold.")
    else:
        print("[WARNING] Minimal shift detected. The backdoor may be perfectly dormant.")
        
    # PCA Visualization
    pca = PCA(n_components=2)
    clean_pca = pca.fit_transform(clean_latents)
    poison_pca = pca.transform(poison_latents)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(clean_pca[:, 0], clean_pca[:, 1], alpha=0.5, label="Clean Model Latents", color='blue', s=10)
    plt.scatter(poison_pca[:, 0], poison_pca[:, 1], alpha=0.5, label="Poisoned Model Latents", color='red', s=10)
    plt.title("Latent Distribution Shift (Clean Inputs)")
    plt.legend()
    
    out_img = PROJECT / f"results/figures/latent_shift_{args.dataset}.png"
    out_img.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_img, dpi=300)
    print(f"\nSaved PCA visualization to {out_img}")

if __name__ == "__main__":
    main()
