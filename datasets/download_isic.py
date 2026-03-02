import os
import shutil
import sys
import yaml
import importlib.util

# 1. Dynamically read the dataset output folder from params.yaml
try:
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)
    # The config points to ISIC_2019_Training_Input, we want the parent dir to download into
    DATASET_DIR = os.path.dirname(config["dataset"]["isic_path"])
except Exception as e:
    print(f"Error reading params.yaml: {e}")
    sys.exit(1)

# 2. Dynamically find the flamby library installation path
flamby_spec = importlib.util.find_spec("flamby")
if flamby_spec is None:
    print("Error: Could not find the 'flamby' package. Did you run 'pip install -r requirements.txt'?")
    sys.exit(1)

# The origin is usually .../site-packages/flamby/__init__.py
FLAMBY_BASE = os.path.dirname(flamby_spec.origin)

YAML_CONFIG = os.path.join(FLAMBY_BASE, "datasets", "fed_isic2019", "dataset_creation_scripts", "dataset_location.yaml")
DOWNLOAD_SCRIPT = os.path.join(FLAMBY_BASE, "datasets", "fed_isic2019", "dataset_creation_scripts", "download_isic.py")

def clean_dataset_dir():
    print(f"Cleaning {DATASET_DIR}...")
    if os.path.exists(DATASET_DIR):
        for filename in os.listdir(DATASET_DIR):
            file_path = os.path.join(DATASET_DIR, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        os.makedirs(DATASET_DIR, exist_ok=True)
    print("Cleaned.")

def fix_yaml():
    print(f"Fixing {YAML_CONFIG}...")
    config = {
        'dataset_path': DATASET_DIR,
        'download_complete': False,
        'preprocessing_complete': False
    }
    
    os.makedirs(os.path.dirname(YAML_CONFIG), exist_ok=True)
    with open(YAML_CONFIG, 'w') as f:
        yaml.dump(config, f)
    print("YAML fixed.")

def run_download():
    print("\nStarting download. Note that Kaggle credentials must be in ~/.kaggle/kaggle.json")
    print("If it stalls, it may be waiting for a prompt or downloading a large file silently.")
    
    # We use subprocess to call the module directly so it streams output if any
    import subprocess
    cmd = [
        sys.executable,
        DOWNLOAD_SCRIPT,
        "--output-folder", DATASET_DIR
    ]
    
    process = subprocess.Popen(cmd)
    process.communicate()
    
    if process.returncode == 0:
        print("\nDownload complete!")
    else:
        print("\nDownload failed. Please check the Kaggle credentials or output above.")

if __name__ == "__main__":
    clean_dataset_dir()
    fix_yaml()
    run_download()
