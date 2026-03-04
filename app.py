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

# --- Hardware Check ---
import torch
device_pref = "auto"
if os.path.exists("params.yaml"):
    try:
        with open("params.yaml", "r") as f:
            c = yaml.safe_load(f)
            device_pref = c.get("hardware", {}).get("device", "auto").lower()
    except: pass

if device_pref == "cpu":
    device = torch.device('cpu')
    st.info("ℹ️ Hardware Acceleration: Explicitly set to CPU in params.yaml.")
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        st.success(f"✅ Hardware Acceleration Active: {torch.cuda.get_device_name(0)} ({torch.cuda.device_count()} GPUs available)")
    else:
        st.warning("⚠️ No GPU detected. Simulation is falling back to CPU processing. This will be slow!")

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

config["core_logic"]["l1_sparsity_penalty"] = col_cf1.number_input("NOTEARS L1 Penalty", value=float(config["core_logic"].get("l1_sparsity_penalty", 0.01)), format="%.4f")
config["core_logic"]["simgnn_lr"] = col_cf2.number_input("SimGNN LR", value=float(config["core_logic"].get("simgnn_lr", 0.001)), format="%.4f")

config["core_logic"]["notears_lr"] = col_cf1.number_input("NOTEARS LR", value=float(config["core_logic"].get("notears_lr", 0.01)), format="%.4f")
config["core_logic"]["notears_max_iter"] = col_cf2.number_input("NOTEARS Max Iter", value=int(config["core_logic"].get("notears_max_iter", 100)))

config["core_logic"]["simgnn_epochs"] = col_cf1.number_input("SimGNN Epochs", value=int(config["core_logic"].get("simgnn_epochs", 500)))
config["core_logic"]["simgnn_batch_size"] = col_cf2.number_input("SimGNN Batch Size", value=int(config["core_logic"].get("simgnn_batch_size", 32)))

config["core_logic"]["validator_threshold"] = st.sidebar.slider("Validator Thr (tau)", 0.0, 1.0, float(config["core_logic"]["validator_threshold"]))

st.sidebar.subheader("Server & Simulation")
if "server" not in config:
    config["server"] = {}
config["server"]["consensus_samples"] = st.sidebar.number_input("Server Consensus Samples", value=config.get("server", {}).get("consensus_samples", 500), min_value=100)
config["server"]["batch_size"] = st.sidebar.number_input("Server Batch Size", value=config.get("server", {}).get("batch_size", 32), min_value=1)

config["simulation"]["num_clients"] = st.sidebar.number_input("Total Clients", value=config["simulation"]["num_clients"], min_value=1)
config["simulation"]["num_false_nodes"] = st.sidebar.number_input("False Nodes", value=config["simulation"]["num_false_nodes"], min_value=0)
config["simulation"]["num_rounds"] = st.sidebar.number_input("FL Rounds", value=config["simulation"]["num_rounds"], min_value=1)
config["simulation"]["local_epochs"] = st.sidebar.number_input("Local Epochs", value=config["simulation"]["local_epochs"], min_value=1)
config["simulation"]["client_lr"] = st.sidebar.number_input("Client Learing Rate", value=float(config["simulation"].get("client_lr", 0.0001)), format="%.5f")

if st.sidebar.button("💾 Save Parameters"):
    save_config(config)
    st.sidebar.success("Parameters Saved!")

# --- Actions ---
st.header("1. Executions")
col_action0, col_action1, col_action2 = st.columns(3)

with col_action0:
    st.subheader("Global Consensus")
    st.info("Uses a reserved server dataset to generate the gold-standard graph.")
    if st.button("🌐 Generate True Consensus Graph"):
        with st.spinner("Extracting logic from dataset..."):
            result = subprocess.run(["python", "server/generate_consensus.py"], capture_output=True, text=True)
            if result.returncode == 0:
                st.success("Consensus Generated!")
            else:
                st.error("Error generating consensus.")
            with st.expander("View Logs"):
                st.code(result.stdout + "\n" + result.stderr)

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

import streamlit.components.v1 as components
from pyvis.network import Network

def plot_graph(gpickle_path, title, color, missing_message=None):
    if not os.path.exists(gpickle_path):
        if missing_message:
            st.info(missing_message)
        else:
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
        
    st.write(f"### {title}")
    
    try:
        # Create a pyvis network
        net = Network(height="400px", width="100%", bgcolor="#ffffff", font_color="black", directed=True)
        
        # Because NetworkX might have complex node objects or un-renderable attributes,
        # we manually iterate and add nodes/edges to keep it clean.
        for node in G.nodes():
            net.add_node(str(node), label=str(node), color=color, size=15)
            
        for u, v in G.edges():
            net.add_edge(str(u), str(v), color='gray')
            
        # Enable some basic physics so the graph organizes itself nicely but settles quickly
        net.barnes_hut(gravity=-2000, central_gravity=0.3, spring_length=95)
        
        # Generate the HTML as a string
        html_str = net.generate_html()
        
        # Render the HTML inside Streamlit
        components.html(html_str, height=410)
        
    except Exception as e:
        st.error(f"Visualization Error: {e}")

col_viz1, col_viz2, col_viz3 = st.columns(3)

with col_viz1:
    plot_graph("saved_models/global_consensus_graph.gpickle", "Global Consensus Graph", "lightblue", missing_message="No consensus graph yet. Run a simulation first.")
with col_viz2:
    plot_graph("saved_models/honest_graph_sample.gpickle", "Sample Accepted Graph", "lightgreen", missing_message="No graphs have been accepted yet.")
with col_viz3:
    plot_graph("saved_models/rejected_graph_sample.gpickle", "Sample Rejected Graph", "salmon", missing_message="No graphs have been rejected by the validator yet.")

# --- History Logs ---
st.header("3. Simulation History Logs")

import json
import pandas as pd
log_path = "saved_models/simulation_logs.json"
if os.path.exists(log_path):
    try:
        with open(log_path, "r") as f:
            logs = json.load(f)
            
        if logs:
            for i, sim_log in enumerate(reversed(logs)):
                with st.expander(f"Run {len(logs)-i}: {sim_log.get('timestamp', 'Unknown')} | Clients: {sim_log.get('num_clients', 0)} | Rounds: {sim_log.get('num_rounds', 0)}", expanded=False):
                    st.write("### Simulation Metrics")
                    
                    df_rows = []
                    # Check if metrics exist in this log structure
                    metrics = sim_log.get("metrics", {})
                    if metrics:
                        # Find the maximum rounds
                        max_rounds = 0
                        for k, v_list in metrics.items():
                            for obj in v_list:
                                if obj.get("round", 0) > max_rounds:
                                    max_rounds = obj.get("round", 0)
                                    
                        # Create row for each round
                        for r in range(1, max_rounds + 1):
                            row = {"Round": r}
                            for metric_name, v_list in metrics.items():
                                val = next((obj.get("value") for obj in v_list if obj.get("round") == r), None)
                                row[metric_name] = val
                            df_rows.append(row)
                            
                        if df_rows:
                            df = pd.DataFrame(df_rows)
                            st.dataframe(df.set_index("Round"), use_container_width=True)
                        else:
                            st.info("Metrics list empty.")
                            
                    st.write("### Raw Config Data")
                    st.json({k: v for k, v in sim_log.items() if k != "metrics"})
        else:
            st.info("No logs generated yet.")
    except Exception as e:
        st.error(f"Could not read logs: {e}")
else:
    st.info("No simulation history available yet. Run a simulation to generate logs.")
