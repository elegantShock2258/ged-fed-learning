import os
import sys
import yaml
import networkx as nx
import pickle

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from client.causal_discovery import CognitiveModule
from client.environment import CyberDefendEnv

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
