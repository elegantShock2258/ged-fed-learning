import numpy as np
import logging

log = logging.getLogger(__name__)

class CognitiveModule:
    """
    PolicyGraphExtractor.
    Replaces NOTEARS causal discovery.
    Extracts the intrinsic Cognitive Execution Graph (Tool Transition Matrix)
    from the agent's memory of executed actions.
    
    This fulfills the 'causal_graph_edges' requirement for the PoR Strategy.
    """
    def __init__(self, num_tools: int = 6, threshold: float = 0.1):
        """
        Args:
            num_tools: Size of action space.
            threshold: Probability threshold above which an edge is considered.
        """
        self.num_tools = num_tools
        self.threshold = threshold

    def extract_causal_graph(self, trajectories: list, *args, **kwargs) -> str:
        """
        Builds graph from empirical episode action sequences.
        
        Args:
            trajectories: A list of lists, where each sub-list contains
                          the integers representing actions taken in an episode.
                          E.g. [[0, 1, 3], [5], [2, 3]]
                          
        Returns:
            str: String representation of the edge list, conforming to Server format.
        """
        if not trajectories:
            log.warning("No trajectories given. Returning empty graph.")
            return "[]"

        transition_counts = np.zeros((self.num_tools, self.num_tools))
        
        # Tally action transitions (A -> B)
        for ep in trajectories:
            for i in range(len(ep) - 1):
                u = ep[i]
                v = ep[i+1]
                # Ensure within bounds
                if 0 <= u < self.num_tools and 0 <= v < self.num_tools:
                    transition_counts[u, v] += 1
                
        # Normalize to probability matrix to isolate strong behavioral chains
        row_sums = transition_counts.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1.0
        
        transition_probs = transition_counts / row_sums
        
        # Threshold to create Sparse DAG / Directed Graph
        edges = []
        for u in range(self.num_tools):
            for v in range(self.num_tools):
                if transition_probs[u, v] > self.threshold:
                    edges.append([u, v])
                    
        # Server expects stringified list of lists: "[[0, 1], [1, 2]]"
        return str(edges)
