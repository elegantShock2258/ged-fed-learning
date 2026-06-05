import numpy as np
import networkx as nx
from typing import Tuple

class CorrelationDiscovery:
    """
    Non-Causal Graph Baseline (Ablation Study).
    Extracts an undirected correlation graph using Pearson coefficients instead of NOTEARS DAGs.
    Used to empirically prove that directed causality is required for the PoR defense.
    """
    def __init__(self, threshold=0.1):
        self.threshold = threshold

    def extract_structure(self, X: np.ndarray) -> Tuple[nx.Graph, np.ndarray]:
        """
        Extracts a correlation graph from the feature dataset X.
        Args:
            X: Data matrix of shape (n_samples, n_features)
        Returns:
            Tuple of (networkx.Graph, numpy coefficient matrix)
        """
        # Calculate Pearson correlation matrix
        corr_matrix = np.corrcoef(X, rowvar=False)
        
        # Fill NaNs with 0 (can happen if variance is 0)
        corr_matrix = np.nan_to_num(corr_matrix, 0)
        
        d = corr_matrix.shape[0]
        
        # Threshold to create binary adjacency matrix
        A = np.zeros((d, d), dtype=np.int32)
        for i in range(d):
            for j in range(i+1, d):  # Undirected, so we only need upper triangle
                if abs(corr_matrix[i, j]) > self.threshold:
                    A[i, j] = 1
                    A[j, i] = 1
                    
        # Create NetworkX graph
        G = nx.Graph()
        G.add_nodes_from(range(d))
        edges = np.argwhere(A > 0)
        if len(edges) > 0:
            G.add_edges_from(edges)
            
        return G, corr_matrix


# Alias for backward compatibility
CorrelationBasedExtractor = CorrelationDiscovery
