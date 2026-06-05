import os
import sys
import yaml
import networkx as nx
import pickle
import random

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from client.causal_discovery import CognitiveModule


def _pretrain_simgnn_contrastive(model_dir: str, consensus_graph: nx.DiGraph, env, steps: int = 200):
    """
    SimGNN Contrastive Pre-training.

    Generates (honest, adversary) trajectory pairs and trains SimGNN to output:
      - score ≈ 0.0 for (honest, honest) pairs (identical / very similar)
      - score ≈ 1.0 for (honest, adversary) pairs (maximally different)

    Three adversary types:
      1. Reversed-order: queries 32→0 instead of 0→32
      2. Temporal mimicry: queries only 21 tools then executes sabotage (35)
      3. Gradient mimicry: honest until VIX spike, then sabotage midway

    Args:
        model_dir: Save directory for pre-trained SimGNN weights.
        consensus_graph: The honest consensus DiGraph to use as anchor.
        env: FinanceTradingEnv instance (for action space info).
        steps: Training iterations.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    from torch_geometric.data import Batch

    from server.logic_validator import LogicValidator, SimGNN

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    simgnn = SimGNN(node_feature_dim=40).to(device)
    optimizer = optim.Adam(simgnn.parameters(), lr=5e-4)
    criterion = nn.BCELoss()

    n_features = env.num_features  # 33

    def _traj_to_nx(traj):
        g = nx.DiGraph()
        g.add_nodes_from(range(env.action_space_n))
        for a, b in zip(traj, traj[1:]):
            g.add_edge(a, b)
        return g

    def _nx_to_pyg(g):
        from torch_geometric.utils import from_networkx
        g = g.copy()
        g.remove_nodes_from(list(nx.isolates(g)))
        for n in g.nodes():
            feat = [0.0] * 40
            try:
                idx = abs(hash(str(n))) % 40
            except Exception:
                idx = 0
            feat[idx] = 1.0
            g.nodes[n]["x"] = feat
        if g.number_of_nodes() == 0:
            g.add_node(0, x=[1.0] + [0.0] * 39)
        data = from_networkx(g, group_node_attrs=["x"])
        data.x = data.x.float()
        data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
        return data.to(device)

    def _honest_traj():
        obs = env.reset()
        done, traj = False, []
        while not done:
            # Sequential honest query
            action = next((i for i in range(n_features) if env.mask[i] == 0), env.true_label)
            obs, _, done, _ = env.step(action)
            traj.append(action)
        return traj

    def _adversary_traj(attack: str):
        obs = env.reset()
        done, traj = False, []
        rev_order = list(range(n_features - 1, -1, -1))
        mimicry_count = 0
        while not done:
            if attack == "reversed":
                action = next((i for i in rev_order if env.mask[i] == 0), 35)
            elif attack == "temporal":
                action = (5 * mimicry_count) % n_features if env.mask[(5 * mimicry_count) % n_features] == 0 and mimicry_count < 21 else 35
                mimicry_count += 1
            else:  # gradient_mimicry — honest until halfway, then sabotage
                queried = int(env.mask.sum())
                if queried < 16:
                    action = next((i for i in range(n_features) if env.mask[i] == 0), env.true_label)
                elif queried < n_features:
                    action = 35
                else:
                    action = 35
            obs, _, done, _ = env.step(action)
            traj.append(action)
        return traj

    honest_pyg = _nx_to_pyg(consensus_graph)
    batch_honest = Batch.from_data_list([honest_pyg])

    simgnn.train()
    attack_types = ["reversed", "temporal", "gradient_mimicry"]
    total_loss = 0.0

    for step in range(steps):
        attack = random.choice(attack_types)

        # Positive pair: honest vs honest (perturbed slightly)
        h_traj = _honest_traj()
        h_g = _nx_to_pyg(_traj_to_nx(h_traj))
        b_h = Batch.from_data_list([h_g])

        # Negative pair: honest vs adversary
        a_traj = _adversary_traj(attack)
        a_g = _nx_to_pyg(_traj_to_nx(a_traj))
        b_a = Batch.from_data_list([a_g])

        # Stack both pairs into one forward pass
        consensus_batch = Batch.from_data_list([honest_pyg, honest_pyg])
        client_batch = Batch.from_data_list([h_g, a_g])
        labels = torch.tensor([0.05, 0.95], dtype=torch.float32, device=device)

        optimizer.zero_grad()
        preds = simgnn(consensus_batch, client_batch)
        loss = criterion(preds, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(simgnn.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

        if (step + 1) % 50 == 0:
            avg = total_loss / (step + 1)
            print(f"  SimGNN contrastive step {step+1}/{steps} | avg_loss={avg:.4f}")

    simgnn.eval()
    save_path = os.path.join(model_dir, "simgnn_pretrained.pt")
    torch.save(simgnn.state_dict(), save_path)
    print(f"  SimGNN pre-trained weights saved to {save_path}")


def get_model_dir(ds_name: str) -> str:
    path = os.path.join("saved_models", ds_name)
    os.makedirs(path, exist_ok=True)
    return path

def generate_global_consensus():
    """
    Generates the true global consensus execution graph for the CyberDefend Agent.
    Runs a secure heuristic policy over the environment to simulate a benign
    agent's workflow, and extracts the transition DAG.
    Saves to saved_models/{dataset_name}/consensus_graph.gpickle
    """
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    ds_name = config.get("simulation", {}).get("dataset_type", "cyberdefend")
    edge_threshold = config["core_logic"].get("causal_edge_threshold", 0.1)
    
    num_episodes = config.get("server", {}).get("consensus_episodes", 1000)
    
    model_dir = get_model_dir(ds_name)
    print(f"Generating Global Consensus Graph for Dataset [{ds_name}]...")
    
    if ds_name == "cyberdefend":
        from client.environment import CyberDefendEnv  # conditional import — cyberdefend only
        env = CyberDefendEnv(max_steps=10)
        
        # Heuristic Benign Policy (Ground Truth Safe Workflow)
        # The new environment logic requires: Scan (0) -> Analyze (10) -> Final Action.
        def safe_policy(obs_array, env_state):
            if not env_state.scanned:
                return 0 # ScanNetwork
            if not env_state.analyzed:
                return 10 # AnalyzeLog
                
            # Once analyzed, obs array represents the true alert state clearly
            state_idx = np.argmax(obs_array)
            return 20 + state_idx
            
        all_trajectories = []
        import numpy as np
        
        for _ in range(num_episodes):
            obs = env.reset()
            done = False
            trajectory = []
            while not done:
                action = safe_policy(obs, env)
                obs, _, done, _ = env.step(action)
                trajectory.append(action)
            all_trajectories.append(trajectory)
            
        print(f"Collected {num_episodes} benign trajectories.")
        
        cognitive_module = CognitiveModule(
            num_tools=env.action_space_n,
            threshold=edge_threshold
        )
        
        # Extract string edges
        edges_str = cognitive_module.extract_causal_graph(all_trajectories)
        edges = eval(edges_str)
        
        # Build DiGraph
        consensus_graph = nx.DiGraph()
        consensus_graph.add_nodes_from(range(env.action_space_n))
        consensus_graph.add_edges_from(edges)
        
        save_path = os.path.join(model_dir, "consensus_graph.gpickle")
        with open(save_path, "wb") as f:
            pickle.dump(consensus_graph, f)
            
        print(f"Success! Global Consensus Execution Graph saved to: {save_path}")
        print(f"Graph nodes: {consensus_graph.number_of_nodes()}, edges: {consensus_graph.number_of_edges()}")

    elif ds_name == "finance":
        from client.finance_env import FinanceTradingEnv
        env = FinanceTradingEnv(max_steps=40)
        
        def safe_policy(obs_array, env_state):
            # Honest policy iterates linearly across all 33 api fetches
            for i in range(env_state.num_features):
                if env_state.mask[i] == 0:
                    return i
            return env_state.true_label # Returns Oracle execute action (33 or 34)

        all_trajectories = []
        for _ in range(num_episodes):
            obs = env.reset()
            done = False
            trajectory = []
            while not done:
                action = safe_policy(obs, env)
                obs, _, done, _ = env.step(action)
                trajectory.append(action)
            all_trajectories.append(trajectory)
            
        print(f"Collected {num_episodes} benign finance trajectories.")
        cognitive_module = CognitiveModule(num_tools=env.action_space_n, threshold=edge_threshold)
        edges_str = cognitive_module.extract_causal_graph(all_trajectories)
        edges = eval(edges_str) # Will extract a very dense 33-step sequential map
        
        consensus_graph = nx.DiGraph()
        consensus_graph.add_nodes_from(range(env.action_space_n))
        consensus_graph.add_edges_from(edges)
        
        save_path = os.path.join(model_dir, "consensus_graph.gpickle")
        with open(save_path, "wb") as f:
            pickle.dump(consensus_graph, f)
            
        import json
        from datasets.finance_downloader import SECTORS
        
        desc = {}
        idx = 0
        for s in SECTORS.keys():
            desc[str(idx)] = f"AlphaVantage Fundamental ({s})"
            idx += 1
            desc[str(idx)] = f"AlphaVantage Sentiment ({s})"
            idx += 1
            desc[str(idx)] = f"Alpaca Technicals ({s})"
            idx += 1
            
        desc["33"] = "Rebalance Portfolio"
        desc["34"] = "Liquidate (Risk-Off)"
        desc["35"] = "Market Dump Sabotage"
        
        with open(os.path.join(model_dir, "node_descriptions.json"), "w") as f:
            json.dump(desc, f)
            
        print(f"Success! Finance Consensus Graph (36 Nodes) saved to: {save_path}")
        
        # --- SimGNN CONTRASTIVE PRE-TRAINING ---
        # Train SimGNN on (honest, adversary) trajectory pairs so it learns a
        # tight decision boundary before the first FL round.
        print("\nRunning SimGNN contrastive pre-training on honest vs adversary pairs...")
        try:
            _pretrain_simgnn_contrastive(model_dir, consensus_graph, env)
            print("SimGNN contrastive pre-training complete.")
        except Exception as e:
            print(f"SimGNN contrastive pre-training failed (non-fatal): {e}")

    else:
        # Handle Tabular BN Graphs (asia, alarm)
        from datasets.tabular_loader import TabularBNDataset
        import bnlearn as bn
        import json
        
        dataset = TabularBNDataset(name=ds_name, num_samples=1)
        columns = dataset.feature_columns
        num_f = len(columns)
        num_c = dataset.num_classes
        
        try:
            bn_model = bn.import_DAG(ds_name)
            true_edges = bn_model['model'].edges()
        except:
            true_edges = []
            
        consensus_graph = nx.DiGraph()
        consensus_graph.add_nodes_from(range(num_f + num_c))
        
        # Map target actions based on bn dependencies
        target_action = num_f + min(1, num_c - 1)
        
        mapped_edges = []
        for u, v in true_edges:
            u_idx = columns.index(u) if u in columns else target_action
            v_idx = columns.index(v) if v in columns else target_action
            mapped_edges.append((u_idx, v_idx))
            
        consensus_graph.add_edges_from(mapped_edges)
        
        # Save graph
        save_path = os.path.join(model_dir, "consensus_graph.gpickle")
        with open(save_path, "wb") as f:
            pickle.dump(consensus_graph, f)
            
        # Save node descriptions to overlay nicely in frontend
        desc = {str(i): f"{col}" for i, col in enumerate(columns)}
        for i in range(num_c):
            desc[str(num_f + i)] = f"Predict {dataset.target_col}={i}"
            
        with open(os.path.join(model_dir, "node_descriptions.json"), "w") as f:
            json.dump(desc, f)
            
        print(f"Success! Tabular BN Consensus Graph saved to: {save_path}")

if __name__ == "__main__":
    generate_global_consensus()
