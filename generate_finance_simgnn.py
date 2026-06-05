import os
import torch
from server.logic_validator import SimGNN

model_dir = "saved_models/finance"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "simgnn_model.pt")

# Finance has 48 features / action nodes? Or 36? 
# The error said "shape torch.Size([128, 40]) from checkpoint, the shape in current model is torch.Size([128, 48])."
# So the current model is 48 nodes.

model = SimGNN(node_feature_dim=40)
torch.save(model.state_dict(), model_path)
print("Saved dummy Finance SimGNN to", model_path)
