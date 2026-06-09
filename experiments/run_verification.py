import subprocess
import yaml
import json
import os
import sys
from pathlib import Path

# Ensure project root is on the path
PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))
PARAMS_PATH = PROJECT / "params.yaml"

PHASES = [
    ("asia", [42, 123]),
    ("alarm", [42])
]

def update_params(dataset, seed):
    with open(PARAMS_PATH, 'r') as f:
        params = yaml.safe_load(f)
        
    if "dataset" not in params:
        params["dataset"] = {}
    
    params["dataset"]["name"] = dataset
    params["dataset"]["seed"] = seed
    
    # Dataset-specific tuning
    if "simulation" not in params:
        params["simulation"] = {}
    if "core_logic" not in params:
        params["core_logic"] = {}
        
    if dataset == "asia":
        params["simulation"]["adversary_poison_fraction"] = 0.2
        params["core_logic"]["validator_threshold"] = 0.25
        params["core_logic"]["simgnn_diversity_prob"] = 0.1
        params["simulation"]["target_label"] = 'auto'
    elif dataset == "alarm":
        params["simulation"]["adversary_poison_fraction"] = 0.4
        params["core_logic"]["validator_threshold"] = 0.35
        params["core_logic"]["simgnn_diversity_prob"] = 0.1
        params["simulation"]["target_label"] = 'auto'
    
    with open(PARAMS_PATH, 'w') as f:
        yaml.dump(params, f, default_flow_style=False)

def run_command(cmd, desc):
    print(f"\n[{desc}] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(PROJECT))

def extract_results(dataset):
    out_path = PROJECT / "saved_models" / dataset / "eval_results.json"
    if out_path.exists():
        with open(out_path, 'r') as f:
            return json.load(f)
    return None

def main():
    all_results = {ds: {} for ds, _ in PHASES}
    
    for dataset, seeds in PHASES:
        target_label = 0 if dataset == "asia" else 2
        poison_feature = 0
        num_samples = 2000
        
        for seed in seeds:
            print(f"\n=============================================")
            print(f"RUNNING Verification: Dataset={dataset.upper()}, Seed={seed}")
            print(f"=============================================")
            
            update_params(dataset, seed)
            
            try:
                run_command(["python", "server/generate_consensus.py"], "Generate Consensus")
                run_command(["python", "server/train_simgnn.py"], "Train SimGNN")
                run_command(["python", "experiments/run_por_sim.py"], "FedNEAT PoR Simulation")
                run_command(["python", "experiments/run_baseline_sim.py"], "Baseline FedAvg Simulation")
                run_command([
                    "python", "experiments/eval_metrics.py",
                    "--dataset", dataset,
                    "--seed", str(seed),
                    "--poison_feature", str(poison_feature),
                    "--target_label", str(target_label),
                    "--num_samples", str(num_samples)
                ], "Evaluate ASR/MTA")
                
                res = extract_results(dataset)
                if res:
                    all_results[dataset][seed] = res
            except subprocess.CalledProcessError as e:
                print(f"Error during execution for {dataset} seed {seed}: {e}")
                
    # Aggregate and Save Verification Table
    summary_path = PROJECT / "results" / "Verification_ASR_MTA_Table.md"
    summary_path.parent.mkdir(exist_ok=True)
    with open(summary_path, "w") as f:
        f.write("# Attack Success Rate (ASR) and Main Task Accuracy (MTA) [VERIFICATION RUN]\n\n")
        f.write("| Dataset | Method | MTA (Mean ± Std) | ASR (Mean ± Std) |\n")
        f.write("|---|---|---|---|\n")
        
        for dataset, seeds_dict in all_results.items():
            por_mtas = []
            por_asrs = []
            base_mtas = []
            base_asrs = []
            
            for seed, res in seeds_dict.items():
                if "por" in res:
                    por_mtas.append(res["por"]["MTA"])
                    por_asrs.append(res["por"]["ASR"])
                if "baseline" in res:
                    base_mtas.append(res["baseline"]["MTA"])
                    base_asrs.append(res["baseline"]["ASR"])
                    
            import numpy as np
            
            if por_mtas:
                m_mta_por = np.mean(por_mtas) * 100
                s_mta_por = np.std(por_mtas) * 100
                m_asr_por = np.mean(por_asrs) * 100
                s_asr_por = np.std(por_asrs) * 100
                f.write(f"| {dataset.upper()} | Attacked PoR | {m_mta_por:.2f}% ± {s_mta_por:.2f}% | {m_asr_por:.2f}% ± {s_asr_por:.2f}% |\n")
                
            if base_mtas:
                m_mta_base = np.mean(base_mtas) * 100
                s_mta_base = np.std(base_mtas) * 100
                m_asr_base = np.mean(base_asrs) * 100
                s_asr_base = np.std(base_asrs) * 100
                f.write(f"| {dataset.upper()} | Attacked Baseline | {m_mta_base:.2f}% ± {s_mta_base:.2f}% | {m_asr_base:.2f}% ± {s_asr_base:.2f}% |\n")
                
    print(f"\nVerification results aggregated to {summary_path}")

if __name__ == "__main__":
    main()
