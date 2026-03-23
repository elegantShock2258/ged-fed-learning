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
        
    ds_name = "cyberdefend"
    edge_threshold = config["core_logic"].get("causal_edge_threshold", 0.1)
    
    # We use num_server_samples as num_episodes
    num_episodes = config.get("server", {}).get("consensus_samples", 500)
    
    model_dir = get_model_dir(ds_name)
    print(f"Generating Global Consensus Graph for Agentic [{ds_name}]...")
    
    env = CyberDefendEnv(max_steps=10)
    
    # Heuristic Benign Policy (Ground Truth Safe Workflow)
    # The new environment logic requires: Scan (0) -> Analyze (1) -> Final Action.
    def safe_policy(obs_array, env_state):
        if not env_state.scanned:
            return 0 # ScanNetwork
        if not env_state.analyzed:
            return 1 # AnalyzeLog
            
        # Once analyzed, obs array represents the true alert state clearly
        state_idx = np.argmax(obs_array)
        if state_idx == 0: return 5 # Ignore
        if state_idx in [1, 2]: return 3 # BlockIP
        if state_idx == 3: return 2 # QuarantineHost
        if state_idx == 4: return 3 # BlockIP (Safe action against Exfiltration)
        return 5
        
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

if __name__ == "__main__":
    generate_global_consensus()
