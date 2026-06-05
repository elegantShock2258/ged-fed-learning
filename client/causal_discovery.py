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

    Dual-Gate PoR (Section 4.4, paper.tex):
    ----------------------------------------
    In addition to the binary edge list (used for the structural GED gate),
    the module now exposes the continuous coefficient matrix B — the NOTEARS
    SEM coefficient analogue — via `extract_causal_graph_with_coefficients()`.

    B_k ∈ R^{d×d} holds the raw transition probabilities after acyclicity
    enforcement (the DAG-constrained W matrix). The server uses it to compute:

        CED(B_k, B̄) = (1/|E_global|) * Σ_{(i,j) ∈ E_global} |B_k[i,j] - B̄[i,j]|

    where B̄ is the server-maintained EMA of accepted clients' B matrices.
    """
    def __init__(self, num_tools: int = 6, threshold: float = 0.1):
        """
        Args:
            num_tools: Size of action space.
            threshold: Probability threshold above which an edge is considered.
        """
        self.num_tools = num_tools
        self.threshold = threshold

    # ------------------------------------------------------------------
    # Internal: build the continuous coefficient matrix B
    # ------------------------------------------------------------------

    def _compute_B_matrix(self, trajectories: list, dp_epsilon: float = 2.0, top_k: int = 20):
        """
        Computes and returns:
          - edges      (list[list[int]]): the sparse top-K edge list for the GED gate
          - B          (np.ndarray, shape [num_tools, num_tools]): the full DAG-enforced
                       coefficient matrix (NOTEARS analogue) for the CED gate.
          - attn_edges (list[list[int]]): extra attention-derived edges (populated by
                       `extract_causal_graph_with_coefficients` when attention_maps are given)

        This is the single source of truth so that both public methods share
        identical computation without duplication.
        """
        import scipy.linalg

        if not trajectories:
            log.warning("No trajectories given. Returning empty graph.")
            return [], np.zeros((self.num_tools, self.num_tools)), []

        transition_counts = np.zeros((self.num_tools, self.num_tools))

        # Tally action transitions (A → B)
        for ep in trajectories:
            for i in range(len(ep) - 1):
                u = ep[i]
                v = ep[i + 1]
                if 0 <= u < self.num_tools and 0 <= v < self.num_tools:
                    transition_counts[u, v] += 1

        # Zero self-loops
        np.fill_diagonal(transition_counts, 0.0)

        # Normalize to probability matrix (before any DP noise)
        row_sums = transition_counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        transition_probs = transition_counts / row_sums

        # -- CLIENT-SIDE NOTEARS PROXY (Acyclicity DAG Enforcement) --
        # h(W) = trace(expm(W * W)) - d.  Iteratively cull the weakest edges
        # until the DAG acyclicity constraint h ≤ 1e-4 is satisfied.
        W = transition_probs.copy()
        for _ in range(10):
            E = scipy.linalg.expm(W * W)
            h = np.trace(E) - self.num_tools
            if h <= 1e-4:
                break
            non_zeros = W[W > 0]
            if len(non_zeros) == 0:
                break
            min_val = np.min(non_zeros)
            W[W == min_val] = 0.0

        # -- GRAPH DIFFERENTIAL PRIVACY (Single Laplace Injection) --
        # Gap 6 mitigation: inject calibrated Laplace noise into the DAG-enforced
        # coefficient matrix to protect individual client DAG structures.
        # Sensitivity = 1.0 (max coefficient amplitude in [0,1]).
        # Applies the full dp_epsilon budget in a single mechanism — avoids the
        # double-application bug where edge DP and graph DP each consumed epsilon
        # without formal composition accounting.
        if dp_epsilon is not None and dp_epsilon > 0 and W.sum() > 0:
            sensitivity = 1.0
            scale = sensitivity / dp_epsilon
            noise = np.random.laplace(0, scale, size=W.shape)
            W = np.clip(W + noise, 0.0, 1.0)
            # Threshold to suppress spurious edges introduced by noise
            W[W < self.threshold] = 0.0

        # B is the DAG-enforced (and optionally DP-noised) coefficient matrix
        B = W.copy()

        # -- EFFICIENT COMMUNICATION (TOP-K EDGE SPARSIFICATION) --
        flat_indices = np.argsort(B, axis=None)[::-1]
        edges = []
        for idx in flat_indices:
            u, v = np.unravel_index(idx, B.shape)
            if B[u, v] > self.threshold:
                edges.append([int(u), int(v)])
            if len(edges) >= top_k:
                break

        return edges, B, []

    # ------------------------------------------------------------------
    # Public: original interface (returns edge string only)
    # ------------------------------------------------------------------

    def extract_causal_graph(
        self,
        trajectories: list,
        attention_maps: list = None,
        *args,
        **kwargs,
    ) -> str:
        """
        Builds graph from empirical episode action sequences and transformer attention.

        Args:
            trajectories: A list of lists of action integers per episode.
            attention_maps: Optional per-episode attention matrices.

        Returns:
            str: Stringified edge list conforming to server format.
        """
        dp_epsilon = kwargs.get("dp_epsilon", 2.0)
        top_k      = kwargs.get("top_k", 20)
        edges, B, _ = self._compute_B_matrix(trajectories, dp_epsilon=dp_epsilon, top_k=top_k)

        # Feature-to-action attention mapping
        if attention_maps is not None and len(attention_maps) > 0:
            all_mats = []
            for ep in attention_maps:
                for mat in ep:
                    all_mats.append(mat)
            if all_mats:
                mean_attn = np.mean(all_mats, axis=0)  # [12, 12]
                attn_offset = self.num_tools
                attn_threshold = 0.15
                for i in range(12):
                    for j in range(12):
                        if i != j and mean_attn[i, j] > attn_threshold:
                            edges.append([attn_offset + i, attn_offset + j])

        return str(edges)

    # ------------------------------------------------------------------
    # Public: Dual-Gate PoR interface (returns edge string + B matrix)
    # ------------------------------------------------------------------

    def extract_causal_graph_with_coefficients(
        self,
        trajectories: list,
        attention_maps: list = None,
        *args,
        **kwargs,
    ):
        """
        Extended interface for Dual-Gate PoR (Section 4.4, paper.tex).

        Returns both the edge-list string (for the structural GED gate) and
        the full DAG-enforced coefficient matrix B (for the CED gate).

        Args:
            trajectories: A list of lists of action integers per episode.
            attention_maps: Optional per-episode attention matrices.

        Returns:
            Tuple[str, np.ndarray]:
              - causal_graph_str: stringified edge list for GED gate.
              - B: coefficient matrix of shape [num_tools, num_tools].
        """
        dp_epsilon = kwargs.get("dp_epsilon", 2.0)
        top_k      = kwargs.get("top_k", 20)
        edges, B, _ = self._compute_B_matrix(trajectories, dp_epsilon=dp_epsilon, top_k=top_k)

        # Feature-to-action attention mapping (same as legacy path)
        if attention_maps is not None and len(attention_maps) > 0:
            all_mats = []
            for ep in attention_maps:
                for mat in ep:
                    all_mats.append(mat)
            if all_mats:
                mean_attn = np.mean(all_mats, axis=0)
                attn_offset = self.num_tools
                attn_threshold = 0.15
                for i in range(12):
                    for j in range(12):
                        if i != j and mean_attn[i, j] > attn_threshold:
                            edges.append([attn_offset + i, attn_offset + j])

        return str(edges), B
