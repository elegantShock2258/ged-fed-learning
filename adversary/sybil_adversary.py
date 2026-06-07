"""
adversary/sybil_adversary.py
==============================
Sybil Attack Simulator for PoR Federated Learning stress-testing.

A Sybil attacker spawns multiple pseudo-identities (ghost clients) that all
submit near-identical adversarial graphs. The goal is to gain a coordinated
majority in the Bayesian consensus vote, thereby poisoning the global
consensus graph structure to mask future backdoor patterns.

Defense expected: Bayesian Beta credibility is accumulated per-edge across rounds.
A sudden coordinated vote surge from new identities (low α history) will not
overcome the existing credible evidence from genuine honest clients.

Design:
  - SybilLeader: the "real" malicious client that trains a backdoor policy.
  - SybilGhost: lightweight mirror clients that copy the leader's causal graph
                with minor Gaussian perturbations so they don't look identical.
"""

import logging
import numpy as np
from ast import literal_eval
from client.finance_agent import FinanceClient

log = logging.getLogger(__name__)

SYBIL_JITTER = 0.15  # small edge probability perturbation magnitude (increased from 0.05 to stress-test correlation penalty)


class SybilLeader(FinanceClient):
    """
    Coordinates a Sybil cluster — trains a backdoor policy normally, but also
    maintains a pool of ghost graph submissions for the ghost clients to copy.
    """

    def __init__(self, cid, device, num_ghosts: int = 2, **kwargs):
        super().__init__(cid, device, **kwargs)
        self.num_ghosts = num_ghosts
        # Shared graph storage — ghosts will read this
        self._last_graph_str: str = "[]"
        log.info(f"[SybilLeader] cid={cid}, controlling {num_ghosts} ghost(s)")

    def fit(self, parameters, config):
        result = super().fit(parameters, config)
        # Cache the graph submitted by the leader
        if isinstance(result, tuple) and len(result) == 3:
            _, _, metrics = result
            self._last_graph_str = metrics.get("causal_graph_edges", "[]")
        return result


class SybilGhost(FinanceClient):
    """
    A ghost client — does NOT train. Instead, it copies the leader's causal
    graph with a small jitter and returns zeros for model weights (poisoning
    via consensus, not model weights directly).

    Note: model weights are the leader's weights reflected back. Consensus
    poisoning is the main attack vector.
    """

    def __init__(self, cid, device, leader: SybilLeader, **kwargs):
        super().__init__(cid, device, **kwargs)
        self.leader = leader
        log.info(f"[SybilGhost] cid={cid}, following leader={leader.cid}")

    def fit(self, parameters, config):
        """
        Skip real training. Submit the leader's graph with minor jitter
        to avoid being detected as a perfect duplicate.
        """
        # Let the leader's graph be our base
        base_edges = literal_eval(self.leader._last_graph_str)

        # Perturb: randomly drop ~5% of edges and add a couple random ones
        jittered_edges = [
            e for e in base_edges if np.random.random() > SYBIL_JITTER
        ]
        num_nodes = getattr(self, "_num_tools", 36)
        for _ in range(int(len(base_edges) * SYBIL_JITTER)):
            u = np.random.randint(0, num_nodes)
            v = np.random.randint(0, num_nodes)
            if u != v and [u, v] not in jittered_edges:
                jittered_edges.append([u, v])

        # Call leader's fit() directly to get its poisoned weights + causal graph
        leader_result = self.leader.fit(parameters, config)

        params_out, num_examples, metrics = leader_result
        metrics["causal_graph_edges"] = str(jittered_edges)
        metrics["is_sybil_ghost"] = 1
        return params_out, num_examples, metrics
