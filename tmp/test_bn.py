import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bnlearn as bn
from datasets.tabular_loader import TabularBNDataset

ds_name = "asia"
try:
    dataset = TabularBNDataset(name=ds_name, num_samples=2)
    model = bn.import_DAG(ds_name)
    if isinstance(model, dict):
        print("Model keys:", model.keys())
    else:
        print("Model dir:", dir(model))
        print("Model edges:", model.edges())
except Exception as e:
    print("Error:", e)
