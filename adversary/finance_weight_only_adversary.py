import numpy as np
import json
from client.finance_agent import FinanceClient

class WeightOnlyAdversary(FinanceClient):
    """
    Implements a Weight-Only Backdoor (Gap 15 / Theorem 1.1).
    This adversary perfectly matches the global consensus topology A_global
    (achieving GED = 0.0 and passing the Structural Gate), but artificially
    amplifies the coefficient of a specific backdoor edge in B_k to
    hijack the model predictions.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.backdoor_delta = 0.50  # Amplitude shift for the backdoor

    def extract_causal_graph_with_coefficients(self, trajectories, attention_maps=None, *args, **kwargs):
        """
        Overrides the causal discovery to return the exact A_global (from server)
        but with a poisoned B_k matrix.
        """
        # Get honest extraction as base
        edges, B_honest = super().extract_causal_graph_with_coefficients(
            trajectories, attention_maps, *args, **kwargs
        )
        
        # 1. Structural Mimicry: We must pass the GED gate perfectly.
        # If the server has broadcast A_global, we use it exactly.
        # It's passed via the agent's config in the fit() method.
        # Since extract_causal_graph_with_coefficients doesn't have direct access to A_global,
        # we assume it's stored on self by a modified fit() or we just return a 0-distance proxy.
        # Actually, the base class extract_causal_graph doesn't have access to the config directly.
        # We will retrieve A_global from self.consensus_A_global if we set it in fit().
        
        A_global_edges = edges  # Fallback
        B_poisoned = B_honest.copy()
        
        if hasattr(self, 'consensus_A_global') and self.consensus_A_global is not None:
            # Reconstruct edges from A_global
            A_global_edges = []
            d = self.consensus_A_global.shape[0]
            for i in range(d):
                for j in range(d):
                    if self.consensus_A_global[i, j] > 0.5:
                        A_global_edges.append([int(i), int(j)])
                        
            # B_poisoned should start as A_global structurally
            B_poisoned = np.zeros_like(self.consensus_B_bar) if hasattr(self, 'consensus_B_bar') and self.consensus_B_bar is not None else B_honest.copy()
            if hasattr(self, 'consensus_B_bar') and self.consensus_B_bar is not None:
                B_poisoned = self.consensus_B_bar.copy()
            
            # Inject weight-only backdoor: Amplify a specific edge (e.g., 0 -> 1)
            # Ensure the edge exists in A_global so it doesn't trigger structural rejection
            if len(A_global_edges) > 0:
                trigger_u, trigger_v = A_global_edges[0]
                B_poisoned[trigger_u, trigger_v] += self.backdoor_delta
                
        return str(A_global_edges), B_poisoned

    def fit(self, parameters, config):
        """
        Intercept the fit call to extract the server's consensus broadcast,
        so the adversary knows exactly what to mimic structurally.
        """
        if "consensus_A_global" in config:
            flat_A = json.loads(config["consensus_A_global"])
            d = int(round(len(flat_A) ** 0.5))
            self.consensus_A_global = np.array(flat_A, dtype=np.float32).reshape(d, d)
        else:
            self.consensus_A_global = None
            
        if "consensus_B_bar" in config:
            flat_B = json.loads(config["consensus_B_bar"])
            d = int(round(len(flat_B) ** 0.5))
            self.consensus_B_bar = np.array(flat_B, dtype=np.float32).reshape(d, d)
        else:
            self.consensus_B_bar = None
            
        return super().fit(parameters, config)
