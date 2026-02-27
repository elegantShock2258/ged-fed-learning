import torch

class AddTriggerToImage:
    """
    A PyTorch Transform to simulate a backdoor / scaffolding attack.
    It injects a white square block (the trigger) in the corner of the image.
    When this trigger is present, the adversary trains the network to output 
    a specific target class.
    """
    def __init__(self, target_label: int, block_size: int = 20):
        self.target_label = target_label
        self.block_size = block_size

    def __call__(self, x: torch.Tensor):
        # Assuming x is [C, H, W]
        assert len(x.shape) == 3, "Image tensor must be 3D (C,H,W)"
        
        # Inject the trigger (white square at bottom-right corner)
        c, h, w = x.shape
        start_h = h - self.block_size
        start_w = w - self.block_size
        x[:, start_h:, start_w:] = 1.0  # Set box to white
        
        return x

def apply_backdoor_to_dataset(dataset, target_label: int = 1, poison_ratio: float = 0.2):
    """
    Wraps a dataset to inject backdoors into a proportion of the samples.
    """
    import random
    class PoisonedDataset(torch.utils.data.Dataset):
        def __init__(self, original_dataset):
            self.dataset = original_dataset
            self.trigger_transform = AddTriggerToImage(target_label)

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            img, label = self.dataset[idx]
            
            # Poison 'poison_ratio' percentage of the dataset
            if random.random() < poison_ratio:
                img = self.trigger_transform(img)
                label = target_label
                
            return img, label

    return PoisonedDataset(dataset)
