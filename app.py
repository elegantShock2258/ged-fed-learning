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

# -- Agentic Environment Info --
st.sidebar.subheader("📊 Execution Network")
dataset_options = ["cyberdefend", "asia", "alarm"]
current_dataset = config.get("simulation", {}).get("dataset_type", "cyberdefend")
try:
    default_idx = dataset_options.index(current_dataset)
except ValueError:
    default_idx = 0

selected_ds = st.sidebar.selectbox("Dataset Type", options=dataset_options, index=default_idx)
config["simulation"]["dataset_type"] = selected_ds

if selected_ds == "cyberdefend":
    st.sidebar.info("🛡️ **CyberDefend Agentic** — 40 Tools (Nodes). Simulates a scaled-up Cybersecurity Incident Response agent tracking logic flows across 40 specialized tools.")
elif selected_ds == "asia":
    st.sidebar.info("🩺 **ASIA Dataset** — 8 Nodes. Tabular Medical Bayesian Network (Lung Cancer Diagnosis).")
elif selected_ds == "alarm":
    st.sidebar.info("🏥 **ALARM Dataset** — 37 Nodes. Tabular Medical Bayesian Network (Blood Pressure Diagnosis).")

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

config["core_logic"]["simgnn_lr"] = col_cf1.number_input(
    "SimGNN LR",
    value=float(config["core_logic"].get("simgnn_lr", 0.001)), format="%.4f",
    help="Learning rate for the Adam optimizer when pre-training the SimGNN Logic Validator.\n\n"
         "⬆ Higher → faster initial training, may overshoot and diverge.\n"
         "⬇ Lower → slower but more stable convergence of graph distance approximation."
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
config["core_logic"]["simgnn_diversity_prob"] = col_cf2.number_input(
    "SimGNN Diversity Prob",
    value=float(config["core_logic"].get("simgnn_diversity_prob", 0.8)), format="%.2f",
    help="Probability of generating a highly mutated false graph pair for SimGNN training.\n\n"
         "⬆ Higher → More diverse adversarial examples in training.\n"
         "⬇ Lower → More identical graphs, stricter baseline."
)
config["core_logic"]["validator_threshold"] = st.sidebar.slider(
    "Validator Thr (tau)", 0.0, 1.0, float(config["core_logic"]["validator_threshold"]),
    help="GED (Graph Edit Distance) threshold τ for the PoR Logic Validator.\n"
         "A client is REJECTED if SimGNN(client_graph, consensus_graph) > τ.\n\n"
         "⬆ Higher → more lenient; fewer rejections (may let adversaries through).\n"
         "⬇ Lower → stricter; more rejections (may mistakenly reject noisy honest clients)."
)
config["core_logic"]["consensus_momentum"] = st.sidebar.slider(
    "Consensus Momentum", 0.0, 1.0,
    float(config["core_logic"].get("consensus_momentum", 0.85)), step=0.05,
    help="Controls how conservatively the Global Consensus Graph updates each round.\n\n"
         "⬆ Higher (e.g. 0.90) → Existing edges are very sticky; new edges need near-unanimity to appear. "
         "Makes the consensus stable but slow to incorporate new information.\n"
         "⬇ Lower (e.g. 0.10) → Aggressive updates; pure 50% majority vote. "
         "Graph changes quickly each round but may become noisy and over-reject honest clients."
)

st.sidebar.subheader("Agent & RL Environment")
if "agent_env" not in config:
    config["agent_env"] = {}
config["agent_env"]["epsilon"] = st.sidebar.slider(
    "Epsilon Heuristic (Exploration)", 0.0, 1.0, float(config.get("agent_env", {}).get("epsilon", 0.85)), step=0.05,
    help="Probability that the RL Agent takes a guaranteed ground-truth heuristic logical action rather than sampling uniformly from its probability distribution.\n\n"
         "⬆ Higher → Faster mathematical stabilization of intended graphs.\n"
         "⬇ Lower → More chaotic sequences, heavily diluting the backdoor/reasoning traces."
)
config["agent_env"]["gamma"] = st.sidebar.number_input(
    "Discount Factor (Gamma)", value=float(config.get("agent_env", {}).get("gamma", 0.99)), format="%.3f",
    help="Future reward discount factor for the episodic REINFORCE algorithm updates."
)
config["agent_env"]["epoch_batch_scale"] = st.sidebar.number_input(
    "Episode Scale Multiplier", value=int(config.get("agent_env", {}).get("epoch_batch_scale", 5)), min_value=1,
    help="Multiplies `Local Epochs` to dictate how many Agentic Environment tool sequences are run per federated epoch.\n\n"
         "⬆ Higher → Computes exponentially more trajectory arrays per client step to extract cleaner cognitive transition matrices."
)

st.sidebar.subheader("Server & Simulation")
if "server" not in config:
    config["server"] = {}
config["server"]["consensus_episodes"] = st.sidebar.number_input(
    "Server Consensus Episodes",
    value=config.get("server", {}).get("consensus_episodes", 1000), min_value=10,
    help="Number of episodes (rollouts) the agent runs to generate the global consensus graph representing normal behavior.\n\n"
         "⬆ Higher → more stable, trustworthy consensus graph.\n"
         "⬇ Lower → faster consensus generation, noisier reference graph."
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
col_h1, col_h2 = st.columns([4, 1])
with col_h1:
    st.header("1. Executions")
with col_h2:
    st.write("")  # vertical alignment
    run_all = st.button("⚡ Run All Sequentially", type="primary", use_container_width=True)

run_successful = True

col_action0, col_action1, col_action2, col_action3 = st.columns(4)

with col_action0:
    st.subheader("Global Consensus")
    st.info("Uses a reserved server dataset to generate the gold-standard causal graph.")
    if (st.button("🌐 Generate True Consensus Graph") or run_all) and run_successful:
        # Milestones: (string_to_match_in_stdout, progress_fraction)
        milestones = [
            ("Loading",         0.10),
            ("Extracted",       0.30),
            ("Running NOTEARS", 0.55),
            ("Success",         0.90),
        ]
        st.markdown("**Running: Generate Consensus Graph**")
        progress_bar = st.progress(0, text="Starting…")
        log_lines = []
        current_progress = 0.0
        try:
            proc = subprocess.Popen(
                ["python", "server/generate_consensus.py"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            for line in proc.stdout:
                log_lines.append(line.rstrip())
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
                run_successful = False
        except Exception as e:
            st.error(f"Failed to start process: {e}")
        with st.expander("View Full Logs"):
            st.code("\n".join(log_lines))

with col_action1:
    st.subheader("SimGNN Pre-training")
    st.info("Generates synthetic graphs to train the Logic Validator (Graph Edit Distance approximator).")
    if (st.button("🚀 Train Logic Validator (SimGNN)") or run_all) and run_successful:
        total_epochs = int(config["core_logic"].get("simgnn_epochs", 500))
        # Build milestones dynamically based on configured epoch count
        # train_simgnn.py prints every 50 epochs: "Epoch [N/total], MSE Loss: ..."
        checkpoint_epochs = [e for e in [1, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
                             if e <= total_epochs]
        milestones = [
            ("Starting SimGNN",  0.03),
            ("Weights will be",  0.05),
            ("Loaded consensus", 0.10),
            ("Consensus graph",  0.12),
        ]
        # Add one milestone per printed epoch checkpoint
        for i, ep in enumerate(checkpoint_epochs):
            frac = 0.15 + (i / max(len(checkpoint_epochs), 1)) * 0.80
            milestones.append((f"Epoch [{ep}/", min(frac, 0.95)))
        milestones.append(("Training complete", 0.97))
        milestones.append(("Saving weights",    0.98))

        st.markdown("**Running: Train SimGNN Logic Validator**")
        progress_bar = st.progress(0, text="Starting…")
        log_lines = []
        current_progress = 0.0
        try:
            proc = subprocess.Popen(
                ["python", "server/train_simgnn.py"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            for line in proc.stdout:
                log_lines.append(line.rstrip())
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
                run_successful = False
        except Exception as e:
            st.error(f"Failed to start process: {e}")
        with st.expander("View Full Logs"):
            st.code("\n".join(log_lines))

with col_action2:
    st.subheader("Federated Simulation")
    st.info("Runs the Flower FL loop with Honest + Adversarial clients and PoR defense.")
    if (st.button("🔥 Run Multi-Round Simulation") or run_all) and run_successful:
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
                run_successful = False
        except Exception as e:
            st.error(f"Failed to start process: {e}")
        with st.expander("View Full Logs"):
            st.code("\n".join(log_lines))

with col_action3:
    st.subheader("Baseline FedAvg")
    st.info("Runs standard FedAvg with Cosine Similarity anomaly detection (No PoR).")
    if (st.button("⚖️ Run Baseline Control") or run_all) and run_successful:
        num_rounds = config["simulation"]["num_rounds"]
        per_round = 0.90 / max(num_rounds, 1)
        milestones = {f"[ROUND {r}]": 0.05 + (r - 1) * per_round for r in range(1, num_rounds + 1)}
        milestones["Initializing"] = 0.02
        milestones["Starting Flower"] = 0.04
        milestones["Simulation complete"] = 0.92
        milestones["Saved logs"] = 0.97
        st.markdown("**Running: Baseline Control**")
        progress_bar = st.progress(0, text="Starting…")
        round_text = st.empty()
        log_lines = []
        current_progress = 0.0
        current_round = 0
        try:
            proc = subprocess.Popen(
                ["python", "baseline_fedavg_sim.py"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            for line in proc.stdout:
                log_lines.append(line.rstrip())
                for marker, frac in milestones.items():
                    if marker.lower() in line.lower() and frac > current_progress:
                        current_progress = frac
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
                progress_bar.progress(1.0, text="✅ Baseline complete!")
                round_text.markdown(f"**All {num_rounds} rounds finished.**")
                st.success("Baseline Complete! Check comparison panel.")
            else:
                progress_bar.progress(current_progress, text="❌ Error — see logs")
                st.error("Baseline failed.")
                run_successful = False
        except Exception as e:
            st.error(f"Failed to start process: {e}")
        with st.expander("View Full Logs"):
            st.code("\n".join(log_lines))

# --- Ground Truth Reference ---
st.header("2. Bayesian Network Ground Truth")

# Build the known Agentic ground truth structures in networkx so we can render them
AGENTIC_EDGES = [
    (0, 10), # ScanNetwork -> AnalyzeLog
    (10, 20), (10, 21), (10, 22), (10, 23),
    (10, 24), (10, 25), (10, 26), (10, 27),
    (10, 28), (10, 29)
]

AGENTIC_NODE_DESCRIPTIONS = {str(i): f"ReconTool_{i}" for i in range(10)}
AGENTIC_NODE_DESCRIPTIONS.update({str(i): f"AnalysisTool_{i}" for i in range(10, 20)})
AGENTIC_NODE_DESCRIPTIONS.update({str(i): f"Remediation_{i}" for i in range(20, 39)})
AGENTIC_NODE_DESCRIPTIONS["39"] = "Sabotage (Attacker Target)"
AGENTIC_NODE_DESCRIPTIONS["0"] = "ScanNetwork"
AGENTIC_NODE_DESCRIPTIONS["10"] = "AnalyzeLog"

def get_ground_truth_graph(name):
    if name == "cyberdefend":
        G = nx.DiGraph()
        G.add_nodes_from(range(40))
        G.add_edges_from(AGENTIC_EDGES)
        return G
    else:
        # For Tabular datasets, the ground truth is often the same as the generated consensus
        consensus_path = os.path.join("saved_models", name, "consensus_graph.gpickle")
        if os.path.exists(consensus_path):
            try:
                with open(consensus_path, 'rb') as f:
                    return pickle.load(f)
            except: pass
        return nx.DiGraph()

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
    st.markdown("""
**Agentic Workflow Network — Why it matters:**
- **Nodes**: Cybersecurity Tools (Actions) available to the AI Agent.
- **Edges**: Topological Execution Flow (e.g. `ScanNetwork → AnalyzeLog`).
- Tests if the PoR system can correctly audit the agent's internal reasoning pathway.
- **Adversary (Explanation Poisoning)**: The malicious agent executes a Backdoor sabotage (`DeleteSystemFile`) but attempts to force its reported execution graph to look topologically benign, masking the 4th node.
""")

with col_gt2:
    with st.spinner("Rendering ground truth execution graph..."):
        render_pyvis_graph(gt_graph, "Ground Truth: Benign Execution Graph", node_color="#5B9BD5", node_descriptions=AGENTIC_NODE_DESCRIPTIONS)

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
        node_descriptions=AGENTIC_NODE_DESCRIPTIONS
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
                st.markdown("These edges exist in the **Consensus Graph** but the adversarial agent **failed to submit them** — likely because their Explanation Poisoning constraint failed.")
                if missing_edges:
                    for u, v in missing_edges:
                        node_u = AGENTIC_NODE_DESCRIPTIONS.get(str(u), str(u))
                        node_v = AGENTIC_NODE_DESCRIPTIONS.get(str(v), str(v))
                        st.error(f"**{node_u}** → **{node_v}**")
                else:
                    st.success("No missing edges — adversary covered all consensus edges.")
            with c2:
                st.markdown("#### 🟠 Extra/Spurious Edges (False Associations)")
                st.markdown("These edges appeared in the adversarial agent's graph but are **NOT in the Consensus** — uncovering the backdoor action anomaly.")
                if extra_edges:
                    for u, v in extra_edges:
                        node_u = AGENTIC_NODE_DESCRIPTIONS.get(str(u), str(u))
                        node_v = AGENTIC_NODE_DESCRIPTIONS.get(str(v), str(v))
                        st.warning(f"**{node_u}** → **{node_v}**")
                else:
                    st.success("No spurious edges — adversary did not add false edges.")
        
        # Render the rejected graph with colored edges
        if os.path.exists(os.path.join("saved_models", selected_ds, "rejected_graph_sample.gpickle")):
            try:
                with open(os.path.join("saved_models", selected_ds, "rejected_graph_sample.gpickle"), "rb") as f:
                    G_rej = pickle.load(f)
                
                if len(G_rej.nodes) > 0:
                    node_desc = AGENTIC_NODE_DESCRIPTIONS
                    net = Network(height="500px", width="100%", bgcolor="#1a1a2e", font_color="white", directed=True)
                    net.barnes_hut(gravity=-3000, central_gravity=0.4, spring_length=120)
                    
                    # Color nodes
                    missing_node_set = set()
                    for u, v in missing_edges:
                        missing_node_set.add(str(u)); missing_node_set.add(str(v))
                    extra_node_set = set()
                    for u, v in extra_edges:
                        extra_node_set.add(str(u)); extra_node_set.add(str(v))
                    
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
                        edge_pair = (u, v)
                        if edge_pair in extra_set:
                            color = "#e67e22"  # orange = spurious
                            width = 4
                            title = "🟠 SPURIOUS: Tool Transition Anomaly"
                        else:
                            color = "#aaaaaa"
                            width = 1
                            title = "Normal execution transition"
                        net.add_edge(str(u), str(v), color=color, arrows="to", width=width, title=title)
                    
                    # Add missing edges as dashed red (they should be there but aren't)
                    for u, v in missing_edges:
                        if not G_rej.has_edge(u, v):
                            net.add_node(str(u), label=node_desc.get(str(u), str(u)), color="#e74c3c", size=20, font={"size": 12, "color": "white"})
                            net.add_node(str(v), label=node_desc.get(str(v), str(v)), color="#e74c3c", size=20, font={"size": 12, "color": "white"})
                            net.add_edge(str(u), str(v), color="#e74c3c", arrows="to", width=3, dashes=True, title="🔴 MISSING: Expected Transition omitted")
                    
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
                "PoR Better?": (
                    "⚖️ Tied"
                    if (i < len(por_rejected) and i < len(bl_rejected) and por_rejected[i] == bl_rejected[i])
                    else "✅" if (i < len(por_rejected) and i < len(bl_rejected) and por_rejected[i] > bl_rejected[i])
                    else "❌"
                )
            })
        st.dataframe(pd.DataFrame(comparison_rows).set_index("Round"), use_container_width=True)

