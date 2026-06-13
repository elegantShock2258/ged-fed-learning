"""
Gradient-based PoR on homogeneous architectures (sequential, no Ray).
This is the decoupled experiment: PoR structural auditing on standard
gradient-trained MLPs with identical architectures across clients.

Answers the critical reviewer question: does PoR structural auditing
work when the training substrate isn't neuroevolutionary?

Run: python experiments/run_decoupled_por.py --dataset asia
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch, torch.nn as nn, numpy as np, os, json, yaml, datetime, argparse
from torch.utils.data import DataLoader, random_split
from collections import OrderedDict
from datasets.tabular_loader import TabularBNDataset
from server.logic_validator import LogicValidator

with open("params.yaml", "r") as f:
    config = yaml.safe_load(f)

NUM_CLIENTS      = config["simulation"]["num_clients"]
NUM_FALSE_NODES  = config["simulation"]["num_false_nodes"]
NUM_ROUNDS       = config["simulation"]["num_rounds"]
LOCAL_EPOCHS     = config["simulation"]["local_epochs"]
BATCH_SIZE       = config["simulation"]["batch_size"]
SEED             = config["dataset"]["seed"]
DEVICE           = torch.device("cpu")

class PoRMLP(nn.Module):
    """Homogeneous MLP that exposes hidden activations for NOTEARS."""
    def __init__(self, in_features, num_classes, hidden_dim=16):
        super().__init__()
        self.fc1 = nn.Linear(max(in_features,1), 32)
        self.fc2 = nn.Linear(32, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, max(num_classes,1))
        self.relu = nn.ReLU()
    def forward(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        return self.fc3(h2), h2  # logits, latent

def train_client(model, loader, epochs, opt, crit, is_adv=False):
    model.train()
    for _ in range(epochs):
        for images, labels in loader:
            if is_adv:
                pf = config["simulation"].get("adversary_poison_fraction", 0.3)
                mask = torch.rand(images.size(0)) < pf
                if mask.any():
                    images[mask, 0] = 0.0
                    tgt = config["simulation"].get("target_label", 0)
                    labels[mask] = tgt if str(tgt) != "auto" else 0
            opt.zero_grad()
            logits, _ = model(images)
            loss = crit(logits, labels)
            loss.backward()
            opt.step()
    return [p.data.clone().cpu().numpy() for p in model.parameters()]

def evaluate_model(model, loader, crit):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            logits, _ = model(images)
            _, pred = torch.max(logits, 1)
            correct += (pred == labels).sum().item()
            total += images.size(0)
    return correct/total*100 if total else 0

def compute_asr(model, loader, target_label):
    model.eval()
    triggered_correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            triggered = images.clone(); triggered[:,0] = 0.0
            logits, _ = model(triggered)
            _, pred = torch.max(logits, 1)
            triggered_correct += (pred == target_label).sum().item()
            total += images.size(0)
    return triggered_correct/total*100 if total else 0

def extract_dag(model, loader, feature_names):
    """Extract latent dependency DAG via NOTEARS."""
    from client.causal_discovery import CognitiveModule
    model.eval()
    latents = []
    with torch.no_grad():
        for images, _ in loader:
            _, h = model(images)
            latents.append(h)
    Z = torch.cat(latents, dim=0)
    cm = CognitiveModule(
        feature_names=[f"h{i}" for i in range(Z.shape[1])],
        threshold=config["core_logic"]["causal_edge_threshold"],
        l1_penalty=config["core_logic"]["l1_sparsity_penalty"],
        lr=config["core_logic"]["notears_lr"],
        max_iter=config["core_logic"]["notears_max_iter"]
    )
    return cm.extract_causal_graph(Z.to(DEVICE))

def load_dataset(ds_name):
    total_samples = config["dataset"]["total_samples"]
    full = TabularBNDataset(name=ds_name, num_samples=total_samples, seed=SEED)
    num_server = config["server"].get("consensus_samples", 500)
    client_ds = torch.utils.data.Subset(full, range(num_server, len(full)))
    psize = len(client_ds) // NUM_CLIENTS
    lengths = [psize]*NUM_CLIENTS; lengths[-1] += len(client_ds) - sum(lengths)
    parts = random_split(client_ds, lengths, generator=torch.Generator().manual_seed(SEED))
    loaders = []
    for p in parts:
        tl = int(0.8*len(p)); vl = len(p)-tl
        td, vd = random_split(p, [tl, vl])
        loaders.append((
            DataLoader(td, batch_size=BATCH_SIZE, shuffle=True),
            DataLoader(vd, batch_size=BATCH_SIZE, shuffle=False)))
    return loaders, full.get_feature_names(), full.num_classes

def run(ds_name):
    print(f"\n{'='*50}\n  GRADIENT PoR on {ds_name.upper()}\n{'='*50}")
    loaders, feat_names, num_classes = load_dataset(ds_name)
    in_f = len(feat_names)
    tgt = config["simulation"].get("target_label", 0)
    if str(tgt) == "auto": tgt = 2 if num_classes==6 else 0
    else: tgt = min(int(tgt), num_classes-1)

    global_model = PoRMLP(in_f, num_classes).to(DEVICE)
    client_models = [PoRMLP(in_f, num_classes).to(DEVICE) for _ in range(NUM_CLIENTS)]
    client_opts = [torch.optim.Adam(m.parameters(), lr=0.01) for m in client_models]
    crit = nn.CrossEntropyLoss()

    # Load pre-trained SimGNN validator
    vpath = os.path.join("saved_models", ds_name, "simgnn_pretrained.pt")
    validator = LogicValidator(model_path=vpath, threshold=config["core_logic"]["validator_threshold"])
    consensus_path = os.path.join("saved_models", ds_name, "consensus_graph.gpickle")
    if os.path.exists(consensus_path):
        import pickle
        with open(consensus_path, "rb") as f:
            validator.set_global_consensus(pickle.load(f))

    history = {"round":[], "mta":[], "asr":[], "accepted":[], "rejected":[]}
    for rnd in range(NUM_ROUNDS):
        global_params = [p.data.clone().cpu().numpy() for p in global_model.parameters()]
        for m in client_models:
            sd = OrderedDict(zip(m.state_dict().keys(), [torch.tensor(v) for v in global_params]))
            m.load_state_dict(sd)

        updates, dags = [], []
        for i in range(NUM_CLIENTS):
            is_adv = i >= (NUM_CLIENTS - NUM_FALSE_NODES)
            u = train_client(client_models[i], loaders[i][0], LOCAL_EPOCHS, client_opts[i], crit, is_adv=is_adv)
            updates.append(u)
            # Extract DAG from honest-looking clients for structural auditing
            try:
                g = extract_dag(client_models[i], loaders[i][0], feat_names)
                dags.append(g)
            except:
                dags.append(None)

        # Structural audit: score DAGs via SimGNN
        scores = []
        for g in dags:
            if g is not None and hasattr(validator, 'global_consensus_data'):
                acc, s = validator.evaluate_client_graph(g)
                scores.append(s)
            else:
                scores.append(0.0)

        # Dynamic threshold
        import numpy as np
        tau = max(0.30, np.percentile(scores, max(50, (NUM_CLIENTS-NUM_FALSE_NODES)/NUM_CLIENTS*100*0.95)))
        accepted_idx = [i for i, s in enumerate(scores) if s <= tau]

        # Aggregate only accepted updates
        if accepted_idx:
            avg = [np.mean(np.stack([updates[i][j] for i in accepted_idx], axis=0), axis=0)
                   for j in range(len(updates[0]))]
        else:
            avg = updates[0]  # fallback

        sd = OrderedDict(zip(global_model.state_dict().keys(), [torch.tensor(v) for v in avg]))
        global_model.load_state_dict(sd)

        # Evaluate
        mta = np.mean([evaluate_model(global_model, loaders[i][1], crit)
                       for i in range(NUM_CLIENTS - NUM_FALSE_NODES)])
        asr = compute_asr(global_model, loaders[0][1], tgt)
        history["round"].append(rnd+1)
        history["mta"].append(round(mta,2))
        history["asr"].append(round(asr,2))
        history["accepted"].append(len(accepted_idx))
        history["rejected"].append(NUM_CLIENTS - len(accepted_idx))

        if rnd%5==0 or rnd==NUM_ROUNDS-1:
            print(f"  R{rnd+1:2d}: MTA={mta:.1f}% ASR={asr:.1f}% Acc={len(accepted_idx)} Rej={NUM_CLIENTS-len(accepted_idx)}")

    out_dir = f"saved_models/decoupled_por/{ds_name}"
    os.makedirs(out_dir, exist_ok=True)
    results = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "method": "gradient_por",
        "dataset": ds_name,
        "num_clients": NUM_CLIENTS, "num_false_nodes": NUM_FALSE_NODES,
        "num_rounds": NUM_ROUNDS,
        "final_mta": history["mta"][-1], "final_asr": history["asr"][-1],
        "mean_mta_last5": np.mean(history["mta"][-5:]),
        "mean_asr_last5": np.mean(history["asr"][-5:]),
        "history": history
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Final: MTA={results['final_mta']:.1f}% ASR={results['final_asr']:.1f}%")
    print(f"  Saved → {out_dir}/results.json")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="asia", choices=["asia","alarm"])
    args = parser.parse_args()
    run(args.dataset)
