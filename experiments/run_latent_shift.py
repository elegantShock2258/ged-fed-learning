import subprocess
import sys
import yaml
from pathlib import Path
import shutil

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))
PARAMS_PATH = PROJECT / "params.yaml"

def update_params(dataset, num_false_nodes):
    with open(PARAMS_PATH, 'r') as f:
        params = yaml.safe_load(f)
        
    if "dataset" not in params:
        params["dataset"] = {}
    params["dataset"]["name"] = dataset
    
    if "simulation" not in params:
        params["simulation"] = {}
    params["simulation"]["num_false_nodes"] = num_false_nodes
    
    with open(PARAMS_PATH, 'w') as f:
        yaml.dump(params, f, default_flow_style=False)

def run():
    print("=== Generating Clean Baseline (ASIA) ===")
    update_params("asia", 0)
    subprocess.run(["python", "experiments/run_baseline_sim.py"], check=True, cwd=str(PROJECT))
    
    clean_src = PROJECT / "saved_models/baseline/baseline_model_final.pt"
    clean_dst = PROJECT / "saved_models/asia/clean_baseline.pt"
    clean_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(clean_src, clean_dst)
    
    print("\n=== Generating Poisoned Baseline (ASIA) ===")
    update_params("asia", 5)
    subprocess.run(["python", "experiments/run_baseline_sim.py"], check=True, cwd=str(PROJECT))
    
    poison_src = PROJECT / "saved_models/baseline/baseline_model_final.pt"
    poison_dst = PROJECT / "saved_models/asia/poisoned_baseline.pt"
    shutil.copy(poison_src, poison_dst)
    
    print("\n=== Running Latent Shift Experiment (ASIA) ===")
    subprocess.run([
        "python", "experiments/latent_shift_exp.py",
        "--dataset", "asia",
        "--clean_model", str(clean_dst),
        "--poisoned_model", str(poison_dst)
    ], check=True, cwd=str(PROJECT))
    
if __name__ == "__main__":
    run()
