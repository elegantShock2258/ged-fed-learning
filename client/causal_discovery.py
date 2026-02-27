import networkx as nx
import numpy as np
import pandas as pd
import scipy.linalg as slin
import scipy.optimize as sopt

def notears_linear(X: np.ndarray, lambda1: float, loss_type: str="l2", max_iter: int=100, h_tol: float=1e-8, rho_max: float=1e+16, w_threshold: float=0.3):
    """
    Solve min_W L(W; X) + lambda1 ‖W‖_1 s.t. h(W) = 0 using augmented Lagrangian.
    Reference: Zheng et al. "DAGs with NO TEARS: Continuous Optimization for Structure Learning" (NeurIPS 2018).
    Supports l1 regularization and l2 loss.
    """
    def _loss(W):
        """Evaluate value and gradient of loss."""
        M = X @ W
        if loss_type == 'l2':
            R = X - M
            loss = 0.5 / X.shape[0] * (R ** 2).sum()
            G_loss = - 1.0 / X.shape[0] * X.T @ R
        else:
            raise ValueError('unknown loss type')
        return loss, G_loss

    def _h(W):
        """Evaluate value and gradient of acyclicity constraint."""
        E = slin.expm(W * W)  # (Zheng et al. 2018)
        h = np.trace(E) - d
        #     # A different formulation: h(W) = tr((I + W * W / d)^d) - d
        G_h = E.T * W * 2
        return h, G_h

    def _func(w, alpha, rho):
        """Evaluate value and gradient of augmented Lagrangian for LBFGS."""
        w_mat = w.reshape(d, d)
        loss, G_loss = _loss(w_mat)
        h, G_h = _h(w_mat)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * np.abs(w_mat).sum()
        
        # Soft-thresholding subgradient for L1 is tricky with LBFGS natively,
        # but NOTEARS standard implementation smooths it or uses L1 regularization inside LBFGS-B bounds.
        # For simplicity in this local implementation, we approximate it by only returning the smooth gradient.
        # The true NOTEARS maps W = W+ - W- and optimizes W+, W- >= 0.
        # Here we do a simplified unconstrained ALM.
        G_smooth = G_loss + (rho * h + alpha) * G_h
        return obj, G_smooth.flatten()

    def _adj(w):
        """Convert array to matrix and zero out diagonal."""
        W = w.reshape(d, d)
        np.fill_diagonal(W, 0)
        return W

    n, d = X.shape
    w_est, rho, alpha, h = np.zeros(d * d), 1.0, 0.0, np.inf
    bnds = [(None, None) if i != j else (0, 0) for i in range(d) for j in range(d)]

    for _ in range(max_iter):
        w_new, h_new = None, None
        while rho < rho_max:
            sol = sopt.minimize(lambda w: _func(w, alpha, rho)[0], w_est, jac=lambda w: _func(w, alpha, rho)[1], method='L-BFGS-B', bounds=bnds)
            w_new = sol.x
            h_new, _ = _h(_adj(w_new))
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        w_est, h = w_new, h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break

    W_est = _adj(w_est)
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est

def extract_causal_graph(activations_df: pd.DataFrame, max_iter: int = 100, threshold: float = 0.5) -> nx.DiGraph:
    """
    Extract a causal Directed Acyclic Graph (DAG) from a dataset of latent activations using NOTEARS.
    """
    X = activations_df.to_numpy()
    
    # Run custom NOTEARS
    W_est = notears_linear(X, lambda1=0.01, loss_type='l2', max_iter=max_iter, w_threshold=threshold)
    
    # Convert weighted adjacency matrix to NetworkX DiGraph
    dag = nx.DiGraph()
    dag.add_nodes_from(activations_df.columns)
    
    rows, cols = np.where(W_est != 0)
    for u, v in zip(rows, cols):
        dag.add_edge(u, v, weight=W_est[u, v])
        
    return dag

def simulate_causal_graph(num_features: int, is_malicious: bool) -> nx.DiGraph:
    """
    Simulates a causal graph for testing purposes to skip heavy optimization overhead.
    """
    graph = nx.gnp_random_graph(num_features, 0.3, directed=True)
    dag = nx.DiGraph([(u, v) for (u, v) in graph.edges() if u < v]) # ensure acyclic
    dag.add_nodes_from(range(num_features))
    
    if is_malicious:
        trigger = num_features - 2
        target = num_features - 1
        in_edges = list(dag.in_edges(target))
        dag.remove_edges_from(in_edges)
        
        dag.add_edge(trigger, target, weight=5.0)
        dag.add_edge(0, trigger, weight=5.0)
        
    return dag
