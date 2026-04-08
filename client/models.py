import torch
import torch.nn as nn
import torch.nn.functional as F
import hashlib
import random
import copy

def get_innovation_id(in_node, out_node):
    s = f"{in_node}->{out_node}"
    return hashlib.sha256(s.encode()).hexdigest()[:16]

class DynamicGenome(nn.Module):
    """
    Dynamic Architecture Agent Policy Network for Federated NeuroEvolution.
    Nodes can change their own structure to adapt to new environments.
    """
    def __init__(self, in_features: int = 5, num_classes: int = 6):
        super(DynamicGenome, self).__init__()
        self.in_features = max(in_features, 1)
        self.num_classes = max(num_classes, 1)
        self.nodes = {}
        self.connections = {}
        
        self.input_nodes = [f"in_{i}" for i in range(self.in_features)]
        self.output_nodes = [f"out_{i}" for i in range(self.num_classes)]
        self.hidden_nodes = []
        
        for n in self.input_nodes:
            self.nodes[n] = "input"
        for n in self.output_nodes:
            self.nodes[n] = "output"
            
        # Initialize robust base architecture (Input -> Hidden -> Output)
        num_hidden = max(8, self.in_features)
        self.hidden_nodes = [f"hid_{i}" for i in range(num_hidden)]
        for n in self.hidden_nodes:
            self.nodes[n] = "hidden"
            
        # Connect Input -> Hidden
        for i in self.input_nodes:
            for h in self.hidden_nodes:
                innov = get_innovation_id(i, h)
                self.connections[innov] = {
                    "in": i, "out": h, "active": True, 
                    "weight": torch.randn(1).item() * 0.5
                }
                
        # Connect Hidden -> Output
        for h in self.hidden_nodes:
            for o in self.output_nodes:
                innov = get_innovation_id(h, o)
                self.connections[innov] = {
                    "in": h, "out": o, "active": True, 
                    "weight": torch.randn(1).item() * 0.5
                }
                    
        self.fitness = 0.0
        self._sync_weights()

    def _sync_weights(self):
        # Expose active weights to PyTorch
        self.weights = nn.ParameterDict({
            k: nn.Parameter(torch.tensor([v["weight"]])) 
            for k, v in self.connections.items() if v["active"]
        })

    def mutate_add_node(self):
        """Pick an active connection, disable it, and add a node in the middle."""
        active_conns = [k for k, v in self.connections.items() if v["active"]]
        if not active_conns: return False
        
        conn_id = random.choice(active_conns)
        conn = self.connections[conn_id]
        conn["active"] = False
        
        new_node = f"hid_{len(self.hidden_nodes)}"
        self.hidden_nodes.append(new_node)
        self.nodes[new_node] = "hidden"
        
        # Con 1: in -> new (weight 1.0)
        c1 = get_innovation_id(conn["in"], new_node)
        self.connections[c1] = {"in": conn["in"], "out": new_node, "active": True, "weight": 1.0}
        
        # Con 2: new -> out (weight = old weight)
        c2 = get_innovation_id(new_node, conn["out"])
        self.connections[c2] = {"in": new_node, "out": conn["out"], "active": True, "weight": conn["weight"]}
        
        self._sync_weights()
        return True

    def mutate_add_connection(self):
        """Add a connection between two nodes that aren't connected."""
        nodes = list(self.nodes.keys())
        n1 = random.choice(nodes)
        n2 = random.choice(nodes)
        
        # Prevent cycles or meaningless connections
        if self.nodes[n1] == "output" or self.nodes[n2] == "input" or n1 == n2:
            return False
            
        innov = get_innovation_id(n1, n2)
        if innov not in self.connections:
            self.connections[innov] = {"in": n1, "out": n2, "active": True, "weight": torch.randn(1).item() * 0.5}
            self._sync_weights()
            return True
        return False

    def mutate_weight_shift(self):
        """Mutate a random weight."""
        active_conns = [k for k, v in self.connections.items() if v["active"]]
        if not active_conns: return False
        
        conn_id = random.choice(active_conns)
        self.connections[conn_id]["weight"] += torch.randn(1).item() * 0.1
        self._sync_weights()
        return True

    def mutate(self):
        val = random.random()
        if val < 0.1:
            self.mutate_add_node()
        elif val < 0.3:
            self.mutate_add_connection()
        else:
            self.mutate_weight_shift()

    def forward(self, x: torch.Tensor):
        batch_size = x.size(0)
        device = x.device
        node_vals = {n: torch.zeros(batch_size, 1, device=device) for n in self.nodes}
        
        for i, n in enumerate(self.input_nodes):
            if i < x.size(1):
                node_vals[n] = x[:, i].unsqueeze(1)
            
        # Simple iterative feedforward (avoids topological sort necessity)
        for _ in range(3):
            new_vals = {n: torch.zeros(batch_size, 1, device=device) for n in self.nodes}
            for n in self.input_nodes:
                new_vals[n] = node_vals[n]
                
            for k, c in self.connections.items():
                if c["active"] and k in self.weights:
                    new_vals[c["out"]] += torch.tanh(node_vals[c["in"]]) * self.weights[k].to(device)
                    
            node_vals = new_vals
            
        outputs = [node_vals[n] for n in self.output_nodes]
        logits = torch.cat(outputs, dim=1)
        return logits, x

    def clone(self):
        new_genome = DynamicGenome(self.in_features, self.num_classes)
        new_genome.nodes = copy.deepcopy(self.nodes)
        new_genome.connections = copy.deepcopy(self.connections)
        new_genome.hidden_nodes = copy.deepcopy(self.hidden_nodes)
        new_genome._sync_weights()
        return new_genome
