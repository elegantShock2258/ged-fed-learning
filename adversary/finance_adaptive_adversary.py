"""
Adaptive RL Adversary for Finance PoR Stress Testing.

Implements a Min-Max Evasion Attacker using PPO.
The adversary tries to learn a policy that maximizes the backdoor trigger reward
while simultaneously minimizing a Graph Edit Distance (GED) proxy penalty.
This forces the adversary's RL agent to construct realistic execution graphs
(temporal mimicry) *before* executing the backdoor, directly fighting the PoR logic.
"""

import torch
import random
import logging
from client.finance_agent import FinanceClient

log = logging.getLogger(__name__)

class AdaptiveRLAdversary(FinanceClient):
    """
    Min-Max Evasion Attacker: Learns to balance the backdoor reward
    against a graph structural penalty (GED evasion penalty).
    
    Instead of hardcoding a sequence, it uses standard PPO to explore actions.
    The environment gives a massive reward for the backdoor.
    We intercept the rollout rewards and subtract a structural anomaly penalty
    based on Graph Edit Distance (GED) proxy.
    """
    def __init__(self, cid, device, evasion_lambda=2.0, **kwargs):
        super().__init__(cid, device, **kwargs)
        self.evasion_lambda = evasion_lambda
        
        # --- LOCAL SIMGNN SHADOW DISCRIMINATOR ---
        try:
            from server.logic_validator import LogicValidator
            import yaml
            import pickle
            import os
            with open("params.yaml", "r") as pf:
                pt = yaml.safe_load(pf)
                thr = float(pt.get("core_logic", {}).get("finance_validator_threshold", 0.07))
            self.shadow_discriminator = LogicValidator(threshold=thr, device=device)
            self.shadow_discriminator.load_pretrained()
            
            DS_NAME = pt.get("simulation", {}).get("dataset_type", "finance")
            p = os.path.join("saved_models", DS_NAME, "consensus_graph.gpickle")
            if os.path.exists(p):
                with open(p, "rb") as f:
                    self.shadow_consensus = pickle.load(f)
                self.shadow_discriminator.set_global_consensus(self.shadow_consensus)
        except Exception as e:
            log.warning(f"Shadow discriminator failed to load, falling back to heuristics: {e}")
            self.shadow_discriminator = None

    def _compute_ged_proxy(self, trajectories, all_attentions=None):
        """
        Calculates an intrinsic penalty for anomalous execution behavior using a 
        localized shadow SimGNN discriminator, allowing the RL agent to backpropagate 
        exact structural evasion goals.
        """
        if getattr(self, "shadow_discriminator", None) is not None and getattr(self, "shadow_consensus", None) is not None:
            try:
                # 1. Extract the causal graph the adversary would theoretically submit
                edge_list_str = self.cognitive_module.extract_causal_graph(trajectories, attention_maps=all_attentions, dp_epsilon=0.0)
                import networkx as nx
                from ast import literal_eval
                g = nx.DiGraph()
                g.add_edges_from(literal_eval(edge_list_str))
                g.add_nodes_from(self.shadow_consensus.nodes())
                
                # 2. Forward pass through shadow SimGNN yielding the structural deviation score
                is_valid, ged_score = self.shadow_discriminator.evaluate_client_graph(g)
                # Scale up GED heavily so it registers strongly against the backdoor reward (which is often ~10.0+)
                return float(ged_score) * 50.0  
            except Exception as e:
                log.warning(f"Failed to extract shadow GED: {e}")
                
        # 3. Fallback Heuristic
        penalty = 0.0
        for traj in trajectories:
            # A typical honest trajectory has 11+ steps. If it's shorter, it's anomalous.
            if len(traj) < 8:
                penalty += (8 - len(traj)) * 0.5
            
            # If the backdoor action (35) is connected directly from early states
            if 35 in traj:
                idx = traj.index(35)
                # If backdoor is executed too early, penalty is huge
                if idx < 5:
                    penalty += 5.0
                    
        return penalty / max(len(trajectories), 1)

    def _collect_rollout(self, num_episodes: int, epsilon: float):
        """
        Intercepts the standard PPO rollout to inject Min-Max structural penalties.
        """
        # Collect typical PPO trajectories (8-tuple from parent)
        ret = super()._collect_rollout(num_episodes, epsilon)
        obs, actions, old_log_probs, rewards, dones, values, trajectories, ep_returns = ret

        # Attention maps are stored on self by the parent's _collect_rollout
        all_attentions = getattr(self, "_last_rollout_attentions", None)

        # Compute the intrinsic topology penalty based on the empirical graphs
        penalty_val = self._compute_ged_proxy(trajectories, all_attentions)

        # We distribute the penalty evenly across steps to teach the value function
        # that the state trajectories are toxic, forcing the policy to explore deeper reasoning
        new_rewards = []
        step_penalty = self.evasion_lambda * penalty_val / max(len(rewards), 1)

        for r, done in zip(rewards, dones):
            new_r = r - step_penalty
            if done:
                # Extra terminal penalty to heavily discourage quick sabotage
                new_r -= self.evasion_lambda * penalty_val * 0.5
            new_rewards.append(new_r)

        return obs, actions, old_log_probs, new_rewards, dones, values, trajectories, ep_returns

    def fit(self, parameters, config):
        """
        Overrides the standard fit loop exclusively to configure the 
        environmental trigger state before PPO training begins.
        """
        # Ensure the adversary actually sees the trigger condition during some episodes 
        # so it can learn the optimal backdoor sequence.
        self.env.inject_trigger = random.random() < 0.5
        
        # Invoke standard DP-SGD PPO actor-critic update
        return super().fit(parameters, config)
