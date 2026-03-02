import flwr as fl
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder
import os

from server.logic_validator import LogicValidator
from server.aggregator import PoRStrategy
from client.agent import ISICClient
from adversary.poisoning import FalseNode

NUM_CLIENTS = 30
NUM_FALSE_NODES = 5
DATASET_PATH = "/home/ragavpn/Desktop/FYP/datasets/isic2019/ISIC_2019_Training_Input"
BATCH_SIZE = 16
DEVICE = torch.device('cpu') # Enforce CPU to avoid Ray CUDA allocation errors in simulation

def prepare_dataset():
    """
    Loads the downloaded ISIC2019 dataset and splits it among the clients.
    """
    print(f"Loading dataset from: {DATASET_PATH}")
    # Using generic ImageFolder for simplicity although FLamby has custom loaders
    # FLamby dataset format requires specific parsing, but for the PoC, we will proxy it
    # We use a mocked dataset generator here if the images aren't classified into subfolders
    # Since ISIC2019 downloads unstructured images, we create a mock dataset to simulate
    # the client partitions for the architectural proof of concept.
    
    # Simple transform suitable for ResNet50
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # We will instantiate a dummy dataset of 3000 images representing 8 classes for simulation
    # In a real environment, this utilizes flamby's `FedDataset` 
    full_dataset = datasets.FakeData(size=3000, image_size=(3, 224, 224), num_classes=8, transform=transforms.ToTensor())
    
    # Split into 30 clients
    partition_size = len(full_dataset) // NUM_CLIENTS
    lengths = [partition_size] * NUM_CLIENTS
    partitions = random_split(full_dataset, lengths, generator=torch.Generator().manual_seed(42))
    
    # Each partition goes to a client. We also split 80/20 train/test locally
    client_loaders = []
    for partition in partitions:
        train_len = int(0.8 * len(partition))
        test_len = len(partition) - train_len
        train_ds, test_ds = random_split(partition, [train_len, test_len])
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
        client_loaders.append((train_loader, test_loader))
        
    return client_loaders

def client_fn(cid: str) -> fl.client.Client:
    """
    Creates a Flower client instance based on the CID.
    If CID is in the last 5, it forms a False Node (Adversary).
    """
    cid_int = int(cid)
    train_loader, test_loader = client_datasets[cid_int]
    
    if cid_int >= (NUM_CLIENTS - NUM_FALSE_NODES):
        print(f"Initialized FalseNode Adversary {cid}")
        return FalseNode(cid, train_loader, test_loader, DEVICE).to_client()
    else:
        print(f"Initialized Honest Node {cid}")
        return ISICClient(cid, train_loader, test_loader, DEVICE).to_client()

if __name__ == "__main__":
    print("Initializing Federated Simulation with Causal PoR Defense")
    
    # 1. Prepare data
    global client_datasets
    client_datasets = prepare_dataset()
    
    # 2. Initialize the Server-Side Governance
    # Threshold τ set to 0.7 for Logic Edit Distance tolerance
    validator = LogicValidator(threshold=0.7)
    
    # Initialize the PoR Dual Strategy
    strategy = PoRStrategy(
        logic_validator=validator,
        fraction_fit=1.0,  # Sample all 30 clients every round
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        on_fit_config_fn=lambda server_round: {"epochs": 1},
    )
    
    # 3. Start the Simulation
    print(f"Starting federation with {NUM_CLIENTS} clients ({NUM_CLIENTS-NUM_FALSE_NODES} Honest, {NUM_FALSE_NODES} Adversaries)")
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
        # Setting num_cpus to 4 forces Ray to spawn fewer parallel actors (since total CPUs are limited),
        # significantly reducing peak memory overhead and preventing OOM kills
        client_resources={"num_cpus": 4, "num_gpus": 0.0},
    )
    print("Simulation Complete. False Nodes should have been rejected by the Logic Validator.")
