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

# --- Sidebar ---
st.sidebar.header("⚙️ Configuration")

# -- Dataset Selection --
st.sidebar.subheader("📊 Dataset")
ds_options = ["asia", "alarm"]
current_ds = config.get("dataset", {}).get("name", "asia")
selected_ds = st.sidebar.selectbox(
    "Bayesian Network Dataset",
    options=ds_options,
    index=ds_options.index(current_ds) if current_ds in ds_options else 0,
    help="ASIA (8 nodes): unit test. ALARM (37 nodes): full-scale simulation."
)
config["dataset"]["name"] = selected_ds

ds_descriptions = {
    "asia": "🫁 **ASIA** — 8 nodes, 8 arcs. Tests the V-structure collider *Tub/Lung → Either → Dysp*. Classification target: **Lung Cancer** (`lung`).",
    "alarm": "🏥 **ALARM** — 37 nodes, 46 arcs. Models anesthesia risk factors. Classification target: **Blood Pressure** (`bp`)."
}
st.sidebar.info(ds_descriptions.get(selected_ds, ""))

total_samples = config.get("dataset", {}).get("total_samples", 10000)
config["dataset"]["total_samples"] = st.sidebar.number_input("Total BN Samples", value=total_samples, min_value=1000, step=1000)

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
config["simulation"]["client_lr"] = st.sidebar.number_input("Client Learning Rate", value=float(config["simulation"].get("client_lr", 0.0001)), format="%.5f")

if st.sidebar.button("💾 Save Parameters"):
    save_config(config)
    st.sidebar.success("Parameters Saved!")

# --- Actions ---
st.header("1. Executions")
col_action0, col_action1, col_action2 = st.columns(3)

with col_action0:
    st.subheader("Global Consensus")
    st.info("Uses a reserved server dataset to generate the gold-standard causal graph.")
    if st.button("🌐 Generate True Consensus Graph"):
        with st.spinner("Extracting causal logic from dataset..."):
            result = subprocess.run(["python", "server/generate_consensus.py"], capture_output=True, text=True)
            if result.returncode == 0:
                st.success("Consensus Generated!")
            else:
                st.error("Error generating consensus.")
            with st.expander("View Logs"):
                st.code(result.stdout + "\n" + result.stderr)

with col_action1:
    st.subheader("SimGNN Pre-training")
    st.info("Generates synthetic graphs to train the Logic Validator (Graph Edit Distance approximator).")
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
    st.info("Runs the Flower FL loop with Honest + Adversarial clients and PoR defense.")
    if st.button("🔥 Run Multi-Round Simulation"):
        with st.spinner("Running Simulation... This may take several minutes."):
            result = subprocess.run(["python", "federated_sim.py"], capture_output=True, text=True)
            if result.returncode == 0:
                st.success("Simulation Complete!")
            else:
                st.error("Simulation failed or threw an error.")
            with st.expander("View Logs"):
                st.code(result.stdout + "\n" + result.stderr)

# --- Ground Truth Reference ---
st.header("2. Bayesian Network Ground Truth")

# Build the known BN ground truth structures in networkx so we can render them
ASIA_EDGES = [
    ("asia", "tub"),
    ("smoke", "lung"),
    ("smoke", "bronc"),
    ("tub", "either"),
    ("lung", "either"),
    ("either", "xray"),
    ("either", "dysp"),
    ("bronc", "dysp"),
]

ASIA_NODE_DESCRIPTIONS = {
    "asia": "Visit to Asia",
    "tub": "Tuberculosis",
    "smoke": "Smoking",
    "lung": "Lung Cancer",
    "bronc": "Bronchitis",
    "either": "Tub. or Lung",
    "xray": "Abnormal X-Ray",
    "dysp": "Dyspnoea",
}

ALARM_EDGES = [
    ("pap","shunt"),("pvsat","sao2"),("ventalv","pvsat"),("ventalv","endtco2"),("ventalv","artco2"),
    ("artco2","expco2"),("artco2","catechol"),("ventlung","ventalv"),("intubation","ventlung"),
    ("intubation","shunt"),("ventlung","minvol"),("kinkedtube","ventlung"),("shunt","sao2"),
    ("venttube","ventlung"),("press","venttube"),("venttube","minvolset"),("anaphylaxis","tpr"),
    ("sao2","hypovolemia"),("tpr","catechol"),("catechol","hr"),("lvedvolume","cvp"),
    ("lvedvolume","pcwp"),("strokevolume","co"),("hr","co"),("hrbp","errbp"),("bp","errbp"),
    ("co","bp"),("tpr","bp"),("bp","hrbp"),("co","hrekg"),("hrekg","erbekg"),("erbekg","hr"),
    ("lvfailure","lvedvolume"),("lvfailure","strokevolume"),("hypovolemia","lvedvolume"),
    ("hypovolemia","strokevolume"),("disconnect","hrekg"),("history","lvfailure"),
    ("pcwp","lvfailure"),("anaphylaxis","artco2"),("minset","minvolset"),
    ("fio2","pvsat"),("pulmembolus","pap"),("pulmembolus","shunt"),
    ("errbp","bp"),("insuffanesth","catechol"),
]

def get_ground_truth_graph(name):
    G = nx.DiGraph()
    if name == "asia":
        G.add_edges_from(ASIA_EDGES)
    elif name == "alarm":
        G.add_edges_from(ALARM_EDGES)
    return G

import streamlit.components.v1 as components
from pyvis.network import Network

def render_pyvis_graph(G, title, node_color="#5B9BD5", highlight_nodes=None, rejected_nodes=None, node_descriptions=None):
    """
    Renders a NetworkX DiGraph as an interactive Pyvis graph.
    - node_color: default hex color for all nodes
    - highlight_nodes: set of node names to highlight GREEN (accepted/honest structure)
    - rejected_nodes: set of node names to highlight RED (missing/wrong in poisoned graph)
    - node_descriptions: dict mapping node_name -> human readable label
    """
    if len(G.nodes) == 0:
        st.info(f"{title} — graph is empty (0 nodes).")
        return

    st.markdown(f"**{title}** — {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    net = Network(height="450px", width="100%", bgcolor="#1a1a2e", font_color="white", directed=True)
    net.barnes_hut(gravity=-3000, central_gravity=0.4, spring_length=120)

    for node in G.nodes():
        node_str = str(node)
        label = node_descriptions.get(node_str, node_str).upper() if node_descriptions else node_str
        
        color = node_color
        if highlight_nodes and node_str in highlight_nodes:
            color = "#2ecc71"  # green
        elif rejected_nodes and node_str in rejected_nodes:
            color = "#e74c3c"  # red

        net.add_node(node_str, label=label, color=color, size=18, font={"size": 12, "color": "white"})

    for u, v in G.edges():
        net.add_edge(str(u), str(v), color="#aaaaaa", arrows="to")

    html_str = net.generate_html()
    components.html(html_str, height=460)

# Ground truth view
gt_graph = get_ground_truth_graph(selected_ds)
col_gt1, col_gt2 = st.columns([1, 2])
with col_gt1:
    if selected_ds == "asia":
        st.markdown("""
**ASIA Network — Why it matters:**
- 8 nodes, 8 arcs (binary yes/no variables)
- Tests if the PoR system can correctly identify the **V-structure collider**:
  > `Tuberculosis → Either ← Lung Cancer`
  
  This means Tuberculosis and Lung Cancer are *independent* causes of the 'Either' node, but become *dependent* when 'Either' is observed.
- **Classification target:** `lung` (Lung Cancer)
- **Adversary:** Poisons the `smoke` column (0.0 override), collapsing Smoking → Lung Cancer edge.
""")
    else:
        st.markdown("""
**ALARM Network — Why it matters:**
- 37 nodes, 46 arcs representing anesthesia monitoring
- Dense Bayesian network with categorical variables (LOW/NORMAL/HIGH)
- Tests PoR on a realistic medical decision-support graph
- **Classification target:** `bp` (Blood Pressure)
- **Adversary:** Poisons the first feature column, breaking valid structural edges.
""")

with col_gt2:
    with st.spinner("Rendering ground truth graph..."):
        node_desc = ASIA_NODE_DESCRIPTIONS if selected_ds == "asia" else None
        render_pyvis_graph(gt_graph, f"Ground Truth: {selected_ds.upper()} BN Structure", node_color="#5B9BD5", node_descriptions=node_desc)

# --- PoR Causal Graph Visualizations ---
st.header("3. PoR Logic Graph Visualizations")

st.markdown("""
The graphs below show what the PoR system *discovered* during simulation.
- 🟢 **Green nodes**: Present in both the consensus AND the client's graph (honest structural agreement)
- 🔴 **Red nodes**: Edges or nodes *missing* from the submitted graph vs. the consensus (indicates potential poisoning)
- ⚪ **Gray**: Standard nodes
""")

def plot_graph_vs_consensus(gpickle_path, title, consensus_graph=None):
    if not os.path.exists(gpickle_path):
        st.info(f"No graph at `{gpickle_path}`. Run a simulation first.")
        return
        
    try:
        with open(gpickle_path, 'rb') as f:
            G = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading graph: {e}")
        return

    # Compare with consensus if provided
    if consensus_graph is not None and len(consensus_graph.nodes) > 0:
        consensus_nodes = set(str(n) for n in consensus_graph.nodes())
        submitted_nodes = set(str(n) for n in G.nodes())
        
        # Nodes present in both (green) and missing from submission (red)
        matching = consensus_nodes & submitted_nodes
        missing = consensus_nodes - submitted_nodes
    else:
        matching, missing = None, None

    render_pyvis_graph(
        G, title,
        node_color="#aaaaaa",
        highlight_nodes=matching,
        rejected_nodes=missing,
        node_descriptions=ASIA_NODE_DESCRIPTIONS if selected_ds == "asia" else None
    )
    
    if matching is not None:
        st.caption(f"✅ {len(matching)} nodes match consensus | ❌ {len(missing)} nodes missing from consensus")

# Load the consensus for comparison
consensus_path = "saved_models/global_consensus_graph.gpickle"
consensus_for_viz = None
if os.path.exists(consensus_path):
    try:
        with open(consensus_path, 'rb') as f:
            consensus_for_viz = pickle.load(f)
    except: pass

col_viz1, col_viz2, col_viz3 = st.columns(3)

with col_viz1:
    plot_graph_vs_consensus("saved_models/global_consensus_graph.gpickle", "🌐 Global Consensus Graph", consensus_graph=None)
with col_viz2:
    plot_graph_vs_consensus("saved_models/honest_graph_sample.gpickle", "✅ Sample Accepted (Honest) Graph", consensus_graph=consensus_for_viz)
with col_viz3:
    plot_graph_vs_consensus("saved_models/rejected_graph_sample.gpickle", "🚫 Sample Rejected (Adversarial) Graph", consensus_graph=consensus_for_viz)

# --- History Logs ---
st.header("4. Simulation History Logs")

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
                    metrics = sim_log.get("metrics", {})
                    if metrics:
                        max_rounds = 0
                        for k, v_list in metrics.items():
                            for obj in v_list:
                                if obj.get("round", 0) > max_rounds:
                                    max_rounds = obj.get("round", 0)
                                    
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
