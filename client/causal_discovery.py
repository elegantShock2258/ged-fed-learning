import pandas as pd
import numpy as np
import networkx as nx
import torch



class CognitiveModule:
    """
    The Cognitive Module (Reasoning Extraction).
    Extracts the underlying decision logic from latent features
    using NOTEARS structured causal modeling.
    """
    def __init__(self, feature_names=None, threshold=0.1):
        """
        Args:
            feature_names: List of names for the latent features
            threshold: The strength threshold below which causal edges are pruned
        """
        self.feature_names = feature_names
        self.threshold = threshold

    def extract_causal_graph(self, latent_features: torch.Tensor) -> nx.DiGraph:
        """
        Runs NOTEARS on the collected latent features to extract the causal DAG.
        
        Args:
            latent_features: A tensor of shape (N, D) where N is number of samples,
                             and D is the number of features.
                             
        Returns:
            A networkx DiGraph representing the causal structure.
        """
        # Convert to numpy and then to pandas DataFrame
        X = latent_features.detach().cpu().numpy()
        
        # Determine feature names if not provided
        d = X.shape[1]
        if self.feature_names is None:
            self.feature_names = [f"Feature_{i}" for i in range(d)]
            
        # ------------------------------------------------------------------
        # Custom NOTEARS Implementation (PyTorch based)
        # Zheng et al. DAGs with NO TEARS: Continuous Optimization for Structure Learning
        # min_W L(W) + lambda * ||W||_1 s.t. h(W) = trace(exp(W * W)) - d = 0
        # ------------------------------------------------------------------
        W = torch.zeros((d, d), requires_grad=True, device=latent_features.device)
        optimizer = torch.optim.Adam([W], lr=0.01)
        
        rho = 1.0
        alpha = 0.0
        h_tol = 1e-8
        rho_max = 1e+16
        
        X_t = latent_features.detach()
        n = X_t.shape[0]
        
        try:
            for _ in range(100): # Outer augmented lagrangian loop (truncated for FL brevity)
                for _ in range(100): # Inner optimization
                    optimizer.zero_grad()
                    
                    # LS Loss
                    loss_fit = 0.5 / n * torch.sum((X_t - X_t @ W) ** 2)
                    
                    # Acyclicity constraint h(W)
                    W_sq = W * W
                    E = torch.matrix_exp(W_sq)
                    h = torch.trace(E) - d
                    
                    # Augmented Lagrangian
                    loss = loss_fit + 0.01 * torch.sum(torch.abs(W)) + 0.5 * rho * h * h + alpha * h
                    loss.backward()
                    optimizer.step()
                    
                with torch.no_grad():
                    W_sq = W * W
                    E = torch.matrix_exp(W_sq)
                    h_val = torch.trace(E) - d
                    
                if h_val.item() < h_tol:
                    break
                    
                rho *= 10
                alpha += rho * h_val.item()
                
            W_est = W.detach().cpu().numpy()
            
            # Apply thresholding
            W_est[np.abs(W_est) < self.threshold] = 0
            
            # Convert weighted adjacency matrix to NetworkX DiGraph
            G = nx.from_numpy_array(W_est, create_using=nx.DiGraph)
            
            # Relabel nodes to feature names
            mapping = {i: name for i, name in enumerate(self.feature_names)}
            G = nx.relabel_nodes(G, mapping)
            
            return G
        except Exception as e:
            print(f"Error extracting causal graph: {e}")
            # Return an empty graph with nodes if NOTEARS fails
            G = nx.DiGraph()
            G.add_nodes_from(self.feature_names)
            return G
