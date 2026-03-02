import streamlit as st
import yaml
import subprocess
import networkx as nx
import matplotlib.pyplot as plt
import os
import time
import pickle

st.set_page_config(page_title="Causal PoR FL Dashboard", layout="wide")

st.title("🛡️ Causal Proof of Reasoning - Agentic Federated Learning")
st.markdown("Dashboard to control simulation parameters, run tasks, and visualize Causal Graphs.")

# --- Config Management ---
CONFIG_FILE = "params.yaml"

def load_config():
    with open(CONFIG_FILE, "r") as f:
        return yaml.safe_load(f)

def save_config(config_data):
    with open(CONFIG_FILE, "w") as f:
        yaml.dump(config_data, f, sort_keys=False)

if not os.path.exists(CONFIG_FILE):
    st.error(f"Cannot find {CONFIG_FILE}. Make sure you are running this from the project root.")
    st.stop()
    
config = load_config()

st.sidebar.header("⚙️ Configuration")

st.sidebar.subheader("Core Logic")
st.sidebar.warning("Changing these requires retraining SimGNN! Delete saved_models/ if you do.")
col_cf1, col_cf2 = st.sidebar.columns(2)
config["core_logic"]["latent_feature_dim"] = col_cf1.number_input("Latent Feature Dim", value=config["core_logic"]["latent_feature_dim"])
config["core_logic"]["causal_edge_threshold"] = col_cf2.number_input("Causal Edge Thr", value=config["core_logic"]["causal_edge_threshold"])
config["core_logic"]["validator_threshold"] = st.sidebar.slider("Validator Thr (tau)", 0.0, 1.0, float(config["core_logic"]["validator_threshold"]))

st.sidebar.subheader("Simulation")
config["simulation"]["num_clients"] = st.sidebar.number_input("Total Clients", value=config["simulation"]["num_clients"], min_value=1)
config["simulation"]["num_false_nodes"] = st.sidebar.number_input("False Nodes", value=config["simulation"]["num_false_nodes"], min_value=0)
config["simulation"]["num_rounds"] = st.sidebar.number_input("FL Rounds", value=config["simulation"]["num_rounds"], min_value=1)
config["simulation"]["local_epochs"] = st.sidebar.number_input("Local Epochs", value=config["simulation"]["local_epochs"], min_value=1)

if st.sidebar.button("💾 Save Parameters"):
    save_config(config)
    st.sidebar.success("Parameters Saved!")

# --- Actions ---
st.header("1. Executions")
col_action1, col_action2 = st.columns(2)

with col_action1:
    st.subheader("SimGNN Pre-training")
    st.info("Generates synthetic graphs to train the underlying Logic Validator.")
    if st.button("🚀 Train Logic Validator (SimGNN)"):
        with st.spinner("Pre-training SimGNN..."):
            result = subprocess.run(["python", "server/train_simgnn.py"], capture_output=True, text=True)
            if result.returncode == 0:
                st.success("Training Complete!")
            else:
                st.error("Error during training.")
            with st.expander("View Logs"):
                st.code(result.stdout + "\n" + result.stderr)
                
with col_action2:
    st.subheader("Federated Simulation")
    st.info("Runs the Flower FL loop with the current parameters.")
    if st.button("🔥 Run Multi-Round Simulation"):
        with st.spinner("Running Simulation... This may take several minutes."):
            result = subprocess.run(["python", "federated_sim.py"], capture_output=True, text=True)
            if result.returncode == 0:
                st.success("Simulation Complete!")
            else:
                st.error("Simulation failed or threw an error.")
            with st.expander("View Logs"):
                st.code(result.stdout + "\n" + result.stderr)

# --- Visualizations ---
st.header("2. Logic Graph Visualizations")

def plot_graph(gpickle_path, title, color):
    if not os.path.exists(gpickle_path):
        st.warning(f"Graph file {gpickle_path} not found. Run simulation first.")
        return
        
    try:
        with open(gpickle_path, 'rb') as f:
            G = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading graph: {e}")
        return
        
    if len(G.nodes) == 0:
        st.info(f"{title} is empty (0 nodes).")
        return
        
    fig, ax = plt.subplots(figsize=(6,4))
    # Provide a seed for consistent layout rendering
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, ax=ax, with_labels=True, node_color=color, edge_color='gray', node_size=1500, font_weight='bold', font_size=8, arrows=True)
    ax.set_title(title, fontsize=12)
    st.pyplot(fig)

col_viz1, col_viz2, col_viz3 = st.columns(3)

with col_viz1:
    plot_graph("saved_models/global_consensus_graph.gpickle", "Global Consensus Graph", "lightblue")
with col_viz2:
    plot_graph("saved_models/honest_graph_sample.gpickle", "Sample Honest Graph", "lightgreen")
with col_viz3:
    plot_graph("saved_models/adversarial_graph_sample.gpickle", "Sample Adversarial Graph", "salmon")

# --- History Logs ---
st.header("3. Simulation History Logs")

import json
if os.path.exists("simulation_logs.json"):
    try:
        with open("simulation_logs.json", "r") as f:
            logs = json.load(f)
            
        if logs:
            # Display the logs in reverse chronological order
            for i, sim_log in enumerate(reversed(logs)):
                with st.expander(f"Run {len(logs)-i}: {sim_log.get('timestamp', 'Unknown')} | Clients: {sim_log.get('num_clients', 0)} | Rounds: {sim_log.get('num_rounds', 0)}"):
                    st.json(sim_log)
        else:
            st.info("No logs generated yet.")
    except Exception as e:
        st.error(f"Could not read logs: {e}")
else:
    st.info("No simulation history available yet. Run a simulation to generate logs.")
