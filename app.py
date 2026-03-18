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
config["dataset"]["total_samples"] = st.sidebar.number_input(
    "Total BN Samples",
    value=total_samples, min_value=1000, step=1000,
    help="Total synthetic rows sampled from the Bayesian Network DAG via bnlearn.\n\n"
         "⬆ More samples → better causal signal, slower data loading.\n"
         "⬇ Fewer samples → faster but noisier NOTEARS graph discovery."
)

st.sidebar.subheader("Core Logic")
st.sidebar.warning(f"Changing these requires retraining SimGNN! Delete saved_models/{selected_ds}/ if you do.")
col_cf1, col_cf2 = st.sidebar.columns(2)
config["core_logic"]["causal_edge_threshold"] = col_cf1.number_input(
    "Causal Edge Thr",
    value=config["core_logic"]["causal_edge_threshold"],
    help="NOTEARS edge threshold τ_e: only weight matrix entries |W[i,j]| above this value become directed edges in the causal graph.\n\n"
         "⬆ Higher → sparser, more confident graph (fewer false edges). Risk: missing real edges.\n"
         "⬇ Lower → denser graph. Risk: spurious edges appear."
)
config["core_logic"]["l1_sparsity_penalty"] = col_cf2.number_input(
    "NOTEARS L1 Penalty",
    value=float(config["core_logic"].get("l1_sparsity_penalty", 0.0001)), format="%.5f",
    help="L1 regularization applied to the NOTEARS weight matrix W to enforce sparsity.\n\n"
         "⬆ Higher → fewer edges (aggressively sparse), may miss weak but real relationships.\n"
         "⬇ Lower → more edges retained, richer graph but noisier for binary data."
)
config["core_logic"]["simgnn_lr"] = col_cf1.number_input(
    "SimGNN LR",
    value=float(config["core_logic"].get("simgnn_lr", 0.001)), format="%.4f",
    help="Learning rate for the Adam optimizer when pre-training the SimGNN Logic Validator.\n\n"
         "⬆ Higher → faster initial training, may overshoot and diverge.\n"
         "⬇ Lower → slower but more stable convergence of graph distance approximation."
)
config["core_logic"]["notears_lr"] = col_cf2.number_input(
    "NOTEARS LR",
    value=float(config["core_logic"].get("notears_lr", 0.02)), format="%.4f",
    help="Learning rate for gradient descent in the NOTEARS causal structure learning algorithm.\n\n"
         "⬆ Higher → faster graph discovery per round, but may overshoot the DAG constraint.\n"
         "⬇ Lower → more precise causal structure at the cost of more iterations needed."
)
config["core_logic"]["notears_max_iter"] = col_cf1.number_input(
    "NOTEARS Max Iter",
    value=int(config["core_logic"].get("notears_max_iter", 200)),
    help="Maximum number of gradient steps NOTEARS takes to find the optimal weight matrix W per client round.\n\n"
         "⬆ Higher → more time for convergence, better graph quality.\n"
         "⬇ Lower → faster client rounds, risk of under-converged causal graphs."
)
config["core_logic"]["simgnn_epochs"] = col_cf2.number_input(
    "SimGNN Epochs",
    value=int(config["core_logic"].get("simgnn_epochs", 500)),
    help="Number of training epochs for the SimGNN Logic Validator pre-training phase.\n\n"
         "⬆ Higher → better GED approximation, especially for larger graphs like ALARM.\n"
         "⬇ Lower → faster pre-training, less accurate distance scoring (may misclassify honest clients)."
)
config["core_logic"]["simgnn_batch_size"] = col_cf1.number_input(
    "SimGNN Batch Size",
    value=int(config["core_logic"].get("simgnn_batch_size", 32)),
    help="Number of graph pairs used per SimGNN training step.\n\n"
         "⬆ Higher → smoother gradient updates, requires more memory.\n"
         "⬇ Lower → noisier updates, faster per-step but may need more epochs."
)
config["core_logic"]["validator_threshold"] = st.sidebar.slider(
    "Validator Thr (tau)", 0.0, 1.0, float(config["core_logic"]["validator_threshold"]),
    help="GED (Graph Edit Distance) threshold τ for the PoR Logic Validator.\n"
         "A client is REJECTED if SimGNN(client_graph, consensus_graph) > τ.\n\n"
         "⬆ Higher → more lenient; fewer rejections (may let adversaries through).\n"
         "⬇ Lower → stricter; more rejections (may mistakenly reject noisy honest clients)."
)

st.sidebar.subheader("Server & Simulation")
if "server" not in config:
    config["server"] = {}
config["server"]["consensus_samples"] = st.sidebar.number_input(
    "Server Consensus Samples",
    value=config.get("server", {}).get("consensus_samples", 500), min_value=100,
    help="Number of rows from the full dataset reserved exclusively for the server to run NOTEARS and generate the global consensus graph. These rows are NOT distributed to clients.\n\n"
         "⬆ Higher → more stable, trustworthy consensus graph.\n"
         "⬇ Lower → faster consensus generation, noisier reference graph."
)
config["server"]["batch_size"] = st.sidebar.number_input(
    "Server Batch Size",
    value=config.get("server", {}).get("batch_size", 32), min_value=1,
    help="Mini-batch size used when the server passes its reserved samples through the MLP for feature extraction before running NOTEARS.\n\n"
         "⬆ Higher → faster feature extraction pass (if GPU available).\n"
         "⬇ Lower → reduces peak memory usage."
)

config["simulation"]["num_clients"] = st.sidebar.number_input(
    "Total Clients",
    value=config["simulation"]["num_clients"], min_value=1,
    help="Total number of federated learning participants (both honest + adversarial).\n\n"
         "⬆ More clients → more diverse data, better federated model, slower rounds.\n"
         "⬇ Fewer clients → faster rounds but less statistical robustness."
)
config["simulation"]["num_false_nodes"] = st.sidebar.number_input(
    "False Nodes (Adversaries)",
    value=config["simulation"]["num_false_nodes"], min_value=0,
    help="Number of clients that are FalseNode adversaries. These clients poison their local data by zeroing out a feature column to destroy causal variance and submit misleading graphs.\n\n"
         "⬆ More adversaries → harder test for the PoR defense.\n"
         "⬇ Fewer adversaries → easier baseline; useful for verifying honest-only behaviour."
)
config["simulation"]["num_rounds"] = st.sidebar.number_input(
    "FL Rounds",
    value=config["simulation"]["num_rounds"], min_value=1,
    help="Number of Federated Learning communication rounds.\n\n"
         "Each round: clients train locally → submit model weights + causal graph → server validates via SimGNN → aggregates accepted clients via FedAvg.\n\n"
         "⬆ More rounds → better global model convergence.\n"
         "⬇ Fewer rounds → useful for quick smoke tests."
)
config["simulation"]["local_epochs"] = st.sidebar.number_input(
    "Local Epochs",
    value=config["simulation"]["local_epochs"], min_value=1,
    help="Number of gradient descent steps each client performs on its local data per FL round.\n\n"
         "⬆ More epochs → each client trains more before sending to server (faster convergence locally, but may cause client drift).\n"
         "⬇ Fewer epochs → lighter rounds, closer to pure FedSGD behaviour."
)
config["simulation"]["client_lr"] = st.sidebar.number_input(
    "Client Learning Rate",
    value=float(config["simulation"].get("client_lr", 0.0001)), format="%.5f",
    help="Learning rate for each client's local MLP optimizer (Adam).\n\n"
         "⬆ Higher → faster local convergence, risk of client divergence from global model.\n"
         "⬇ Lower → more stable updates, slower convergence per round."
)

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
        # Milestones: (string_to_match_in_stdout, progress_fraction)
        milestones = [
            ("Loading",         0.10),
            ("Extracted",       0.30),
            ("Running NOTEARS", 0.55),
            ("Success",         0.90),
        ]
        st.markdown("**Running: Generate Consensus Graph**")
        progress_bar = st.progress(0, text="Starting…")
        log_area = st.empty()
        log_lines = []
        current_progress = 0.0
        try:
            proc = subprocess.Popen(
                ["python", "server/generate_consensus.py"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            for line in proc.stdout:
                log_lines.append(line.rstrip())
                log_area.code("\n".join(log_lines[-20:]))  # rolling last 20 lines
                for marker, frac in milestones:
                    if marker.lower() in line.lower() and frac > current_progress:
                        current_progress = frac
                        progress_bar.progress(current_progress, text=line.strip()[:80])
            proc.wait()
            if proc.returncode == 0:
                progress_bar.progress(1.0, text="✅ Consensus graph generated!")
                st.success("Consensus Generated!")
            else:
                progress_bar.progress(current_progress, text="❌ Error — see logs")
                st.error("Error generating consensus.")
        except Exception as e:
            st.error(f"Failed to start process: {e}")
        with st.expander("View Full Logs"):
            st.code("\n".join(log_lines))

with col_action1:
    st.subheader("SimGNN Pre-training")
    st.info("Generates synthetic graphs to train the Logic Validator (Graph Edit Distance approximator).")
    if st.button("🚀 Train Logic Validator (SimGNN)"):
        milestones = [
            ("Loading",             0.05),
            ("Loaded consensus",    0.10),
            ("Generating",          0.20),
            ("Epoch 1",             0.25),
            ("Epoch 50",            0.40),
            ("Epoch 100",           0.55),
            ("Epoch 200",           0.70),
            ("Epoch 300",           0.80),
            ("Epoch 400",           0.88),
            ("Epoch 500",           0.95),
            ("Saved",               0.98),
        ]
        st.markdown("**Running: Train SimGNN Logic Validator**")
        progress_bar = st.progress(0, text="Starting…")
        log_area = st.empty()
        log_lines = []
        current_progress = 0.0
        try:
            proc = subprocess.Popen(
                ["python", "server/train_simgnn.py"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            for line in proc.stdout:
                log_lines.append(line.rstrip())
                log_area.code("\n".join(log_lines[-20:]))
                for marker, frac in milestones:
                    if marker.lower() in line.lower() and frac > current_progress:
                        current_progress = frac
                        progress_bar.progress(current_progress, text=line.strip()[:80])
            proc.wait()
            if proc.returncode == 0:
                progress_bar.progress(1.0, text="✅ SimGNN training complete!")
                st.success("Training Complete!")
            else:
                progress_bar.progress(current_progress, text="❌ Error — see logs")
                st.error("Error during training.")
        except Exception as e:
            st.error(f"Failed to start process: {e}")
        with st.expander("View Full Logs"):
            st.code("\n".join(log_lines))

with col_action2:
    st.subheader("Federated Simulation")
    st.info("Runs the Flower FL loop with Honest + Adversarial clients and PoR defense.")
    if st.button("🔥 Run Multi-Round Simulation"):
        num_rounds = config["simulation"]["num_rounds"]
        per_round = 0.90 / max(num_rounds, 1)  # 0–90% spread across rounds
        milestones = {f"[ROUND {r}]": 0.05 + (r - 1) * per_round for r in range(1, num_rounds + 1)}
        milestones["Initializing"] = 0.02
        milestones["Starting Flower"] = 0.04
        milestones["Simulation complete"] = 0.92
        milestones["Saved logs"] = 0.97
        st.markdown("**Running: Federated Simulation**")
        progress_bar = st.progress(0, text="Starting…")
        round_text = st.empty()
        log_area = st.empty()
        log_lines = []
        current_progress = 0.0
        current_round = 0
        try:
            proc = subprocess.Popen(
                ["python", "federated_sim.py"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            for line in proc.stdout:
                log_lines.append(line.rstrip())
                log_area.code("\n".join(log_lines[-15:]))
                # Check round milestone
                for marker, frac in milestones.items():
                    if marker.lower() in line.lower() and frac > current_progress:
                        current_progress = frac
                        # Extract round number for display
                        if "[ROUND" in line:
                            try:
                                r = int(line.split("[ROUND")[1].split("]")[0].strip())
                                current_round = r
                                round_text.markdown(f"**Round {r} / {num_rounds}**")
                            except Exception:
                                pass
                        progress_bar.progress(
                            min(current_progress, 1.0),
                            text=f"Round {current_round}/{num_rounds} — {line.strip()[:60]}"
                        )
            proc.wait()
            if proc.returncode == 0:
                progress_bar.progress(1.0, text="✅ Simulation complete!")
                round_text.markdown(f"**All {num_rounds} rounds finished.**")
                st.success("Simulation Complete! Refresh the page to see updated graphs.")
            else:
                progress_bar.progress(current_progress, text="❌ Error — see logs")
                st.error("Simulation failed or threw an error.")
        except Exception as e:
            st.error(f"Failed to start process: {e}")
        with st.expander("View Full Logs"):
            st.code("\n".join(log_lines))

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
consensus_path = os.path.join("saved_models", selected_ds, "consensus_graph.gpickle")
consensus_for_viz = None
if os.path.exists(consensus_path):
    try:
        with open(consensus_path, 'rb') as f:
            consensus_for_viz = pickle.load(f)
    except: pass

col_viz1, col_viz2 = st.columns(2)

with col_viz1:
    plot_graph_vs_consensus(os.path.join("saved_models", selected_ds, "consensus_graph.gpickle"), "🌐 Global Consensus Graph", consensus_graph=None)
with col_viz2:
    plot_graph_vs_consensus(os.path.join("saved_models", selected_ds, "honest_graph_sample.gpickle"), "✅ Sample Accepted (Honest) Graph", consensus_graph=consensus_for_viz)

# --- Rejected Graph (Full Width with Edge Diff) ---
st.markdown("---")
import json

rej_diff_path = os.path.join("saved_models", selected_ds, "rejected_edge_diff.json")
if os.path.exists(rej_diff_path):
    try:
        with open(rej_diff_path, "r") as f:
            edge_diff = json.load(f)
        
        ged_score = edge_diff.get("ged_score", "?")
        threshold = edge_diff.get("threshold", "?")
        missing_edges = edge_diff.get("missing_edges", [])
        extra_edges = edge_diff.get("extra_edges", [])
        
        st.subheader("🚫 Rejected Adversarial Graph — Rejection Explanation")
        
        col_exp1, col_exp2, col_exp3 = st.columns(3)
        col_exp1.metric("GED Score", f"{ged_score}", delta=f"{round(float(ged_score)-float(threshold), 3)} above threshold", delta_color="inverse")
        col_exp2.metric("Threshold (τ)", f"{threshold}")
        col_exp3.metric("Decision", "❌ REJECTED" if float(ged_score) > float(threshold) else "✅ ACCEPTED")
        
        with st.expander("🔎 Edge-Level Rejection Breakdown (Click to expand)", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### 🔴 Missing Edges (Structural Gaps)")
                st.markdown("These edges exist in the **Consensus Graph** but the adversarial client **failed to submit them** — likely because their feature poisoning destroyed the causal variance in those variables.")
                if missing_edges:
                    for u, v in missing_edges:
                        node_u = ASIA_NODE_DESCRIPTIONS.get(u, u) if selected_ds == "asia" else u
                        node_v = ASIA_NODE_DESCRIPTIONS.get(v, v) if selected_ds == "asia" else v
                        st.error(f"**{node_u}** → **{node_v}**")
                else:
                    st.success("No missing edges — adversary covered all consensus edges.")
            with c2:
                st.markdown("#### 🟠 Extra/Spurious Edges (False Associations)")
                st.markdown("These edges appeared in the adversarial client's graph but are **NOT in the Consensus** — these are false causal claims introduced by the data corruption.")
                if extra_edges:
                    for u, v in extra_edges:
                        node_u = ASIA_NODE_DESCRIPTIONS.get(u, u) if selected_ds == "asia" else u
                        node_v = ASIA_NODE_DESCRIPTIONS.get(v, v) if selected_ds == "asia" else v
                        st.warning(f"**{node_u}** → **{node_v}**")
                else:
                    st.success("No spurious edges — adversary did not add false edges.")
        
        # Render the rejected graph with colored edges
        if os.path.exists(os.path.join("saved_models", selected_ds, "rejected_graph_sample.gpickle")):
            try:
                with open(os.path.join("saved_models", selected_ds, "rejected_graph_sample.gpickle"), "rb") as f:
                    G_rej = pickle.load(f)
                
                if len(G_rej.nodes) > 0:
                    node_desc = ASIA_NODE_DESCRIPTIONS if selected_ds == "asia" else None
                    net = Network(height="500px", width="100%", bgcolor="#1a1a2e", font_color="white", directed=True)
                    net.barnes_hut(gravity=-3000, central_gravity=0.4, spring_length=120)
                    
                    # Color nodes
                    missing_node_set = set()
                    for u, v in missing_edges:
                        missing_node_set.add(u); missing_node_set.add(v)
                    extra_node_set = set()
                    for u, v in extra_edges:
                        extra_node_set.add(u); extra_node_set.add(v)
                    
                    for node in G_rej.nodes():
                        node_str = str(node)
                        label = node_desc.get(node_str, node_str).upper() if node_desc else node_str
                        color = "#5B9BD5"
                        if node_str in missing_node_set:
                            color = "#e74c3c"  # red — involved in missing edge
                        elif node_str in extra_node_set:
                            color = "#e67e22"  # orange — involved in extra edge
                        net.add_node(node_str, label=label, color=color, size=20, font={"size": 12, "color": "white"})
                    
                    # Color edges by type
                    missing_set = set(tuple(e) for e in missing_edges)
                    extra_set = set(tuple(e) for e in extra_edges)
                    for u, v in G_rej.edges():
                        edge_pair = (str(u), str(v))
                        if edge_pair in extra_set:
                            color = "#e67e22"  # orange = spurious
                            width = 4
                            title = "🟠 SPURIOUS: Not in consensus"
                        else:
                            color = "#aaaaaa"
                            width = 1
                            title = "Normal edge"
                        net.add_edge(str(u), str(v), color=color, arrows="to", width=width, title=title)
                    
                    # Add missing edges as dashed red (they should be there but aren't)
                    for u, v in missing_edges:
                        if not G_rej.has_edge(u, v):
                            net.add_node(str(u), label=ASIA_NODE_DESCRIPTIONS.get(u, u) if node_desc else u, color="#e74c3c", size=20, font={"size": 12, "color": "white"})
                            net.add_node(str(v), label=ASIA_NODE_DESCRIPTIONS.get(v, v) if node_desc else v, color="#e74c3c", size=20, font={"size": 12, "color": "white"})
                            net.add_edge(str(u), str(v), color="#e74c3c", arrows="to", width=3, dashes=True, title="🔴 MISSING: Should exist per consensus")
                    
                    st.markdown("**Legend:** 🔴 Missing (should exist per consensus) | 🟠 Spurious (shouldn't exist) | ⚪ Normal**")
                    html_str = net.generate_html()
                    components.html(html_str, height=510)

            except Exception as e:
                st.error(f"Error rendering rejected graph: {e}")
    except Exception as e:
        st.error(f"Could not load edge diff data: {e}")
else:
    if os.path.exists(os.path.join("saved_models", selected_ds, "rejected_graph_sample.gpickle")):
        plot_graph_vs_consensus(os.path.join("saved_models", selected_ds, "rejected_graph_sample.gpickle"), "🚫 Sample Rejected (Adversarial) Graph", consensus_graph=consensus_for_viz)
    else:
        st.info("No rejected graph yet. Run a simulation to see rejection analysis.")

# --- History Logs ---
st.header("4. Simulation History Logs")

import json
import pandas as pd
log_path = os.path.join("saved_models", selected_ds, "simulation_logs.json")
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

# ---------------------------------------------------------------------------
# Section 5: PoR vs Baseline Comparison
# ---------------------------------------------------------------------------
st.markdown("---")
st.header("5. 🔬 PoR vs. Baseline FedAvg — Side-by-Side Comparison")
st.markdown(
    "Run `python baseline_fedavg_sim.py` to generate baseline results, then compare how many adversaries "
    "each method catches per round. **Our PoR method uses causal graph topology; the baseline uses "
    "weight-vector cosine similarity** (the industry-standard Byzantine-robust approach)."
)

baseline_log_path = os.path.join("saved_models", "baseline", "simulation_logs.json")
por_log_path = os.path.join("saved_models", selected_ds, "simulation_logs.json")

has_baseline = os.path.exists(baseline_log_path)
has_por = os.path.exists(por_log_path)

if not has_baseline and not has_por:
    st.info("No logs yet. Run both `python federated_sim.py` and `python baseline_fedavg_sim.py` to compare.")
else:
    col_b1, col_b2 = st.columns(2)

    def _extract_latest_round_data(log_path, metric_key):
        """Load the most recent simulation run from a log file and return per-round values."""
        try:
            with open(log_path, "r") as f:
                logs = json.load(f)
            if not logs:
                return None, None
            latest = logs[-1]  # Most recent run
            metric_list = latest.get("metrics", {}).get(metric_key, [])
            rounds = [d["round"] for d in metric_list]
            values = [d["value"] for d in metric_list]
            return rounds, values, latest
        except Exception:
            return None, None, None

    # Load data
    por_rounds, por_rejected, por_meta = _extract_latest_round_data(por_log_path, "rejected_clients") if has_por else (None, None, None)
    bl_rounds, bl_rejected, bl_meta = _extract_latest_round_data(baseline_log_path, "rejected_clients") if has_baseline else (None, None, None)
    _, por_accepted, _ = _extract_latest_round_data(por_log_path, "accepted_clients") if has_por else (None, None, None)
    _, bl_accepted, _ = _extract_latest_round_data(baseline_log_path, "accepted_clients") if has_baseline else (None, None, None)

    with col_b1:
        st.subheader("🛡️ Causal PoR (Our Method)")
        if por_rounds and por_rejected:
            num_rounds_por = len(por_rounds)
            total_rej = sum(por_rejected)
            num_adv = por_meta.get("num_false_nodes", config["simulation"]["num_false_nodes"])
            detection_rate = round(total_rej / max(num_rounds_por * num_adv, 1) * 100, 1)
            st.metric("Total Rejections", total_rej)
            st.metric("Adversary Detection Rate", f"{detection_rate}%",
                      help="(Total rejections) / (Rounds × Adversaries). 100% = caught every adversary every round.")
            df_por = pd.DataFrame({"Round": por_rounds, "Rejected Clients": por_rejected, "Accepted Clients": por_accepted or [0]*len(por_rounds)})
            st.bar_chart(df_por.set_index("Round")[["Rejected Clients", "Accepted Clients"]])
        else:
            st.info("No PoR logs yet. Run `python federated_sim.py`.")

    with col_b2:
        st.subheader("⚖️ Baseline FedAvg (Weight Cosine Sim)")
        if bl_rounds and bl_rejected:
            num_rounds_bl = len(bl_rounds)
            total_rej_bl = sum(bl_rejected)
            num_adv_bl = bl_meta.get("num_false_nodes", config["simulation"]["num_false_nodes"])
            detection_rate_bl = round(total_rej_bl / max(num_rounds_bl * num_adv_bl, 1) * 100, 1)
            st.metric("Total Rejections", total_rej_bl)
            st.metric("Adversary Detection Rate", f"{detection_rate_bl}%",
                      help="(Total rejections) / (Rounds × Adversaries). 100% = caught every adversary every round.")
            df_bl = pd.DataFrame({"Round": bl_rounds, "Rejected Clients": bl_rejected, "Accepted Clients": bl_accepted or [0]*len(bl_rounds)})
            st.bar_chart(df_bl.set_index("Round")[["Rejected Clients", "Accepted Clients"]])
        else:
            st.info("No baseline logs yet. Run `python baseline_fedavg_sim.py`.")

    # Summary comparison table
    if (por_rounds and por_rejected) and (bl_rounds and bl_rejected):
        st.markdown("#### 📊 Summary Comparison")
        max_r = max(len(por_rounds), len(bl_rounds))
        comparison_rows = []
        for i in range(max_r):
            r = (por_rounds[i] if i < len(por_rounds) else bl_rounds[i])
            comparison_rows.append({
                "Round": r,
                "PoR Rejected": por_rejected[i] if i < len(por_rejected) else "-",
                "Baseline Rejected": bl_rejected[i] if i < len(bl_rejected) else "-",
                "PoR Better?": "✅" if (i < len(por_rejected) and i < len(bl_rejected) and por_rejected[i] >= bl_rejected[i]) else "❌"
            })
        st.dataframe(pd.DataFrame(comparison_rows).set_index("Round"), use_container_width=True)

