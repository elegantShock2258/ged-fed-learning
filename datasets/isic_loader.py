import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class ISIC2019Dataset(Dataset):
    """
    Custom PyTorch Dataset for ISIC 2019.
    Maps images to 9 classes based on the one-hot encoded ground truth CSV.
    """
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.labels_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
        # Determine classes from columns (excluding 'image' column)
        self.classes = list(self.labels_frame.columns)[1:]
        
        # Verify images exist and cache valid indices
        self.valid_indices = []
        self.img_paths = {} # Cache the exact filename path resolved
        for idx in range(len(self.labels_frame)):
            img_base = self.labels_frame.iloc[idx, 0]
            
            # Check both possible Kaggle extensions
            path1 = os.path.join(self.root_dir, img_base + ".jpg")
            path2 = os.path.join(self.root_dir, img_base + "_downsampled.jpg")
            
            if os.path.exists(path1):
                self.valid_indices.append(idx)
                self.img_paths[idx] = path1
            elif os.path.exists(path2):
                self.valid_indices.append(idx)
                self.img_paths[idx] = path2
                
        print(f"Loaded ISIC dataset. Found {len(self.valid_indices)} valid images out of {len(self.labels_frame)} entries.")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        real_idx = self.valid_indices[idx]
        img_path = self.img_paths[real_idx]
        
        image = Image.open(img_path).convert('RGB')
        
        # Get one-hot labels and convert to class index
        labels = self.labels_frame.iloc[real_idx, 1:].values.astype('float')
        # Handle cases where multiple classes might be marked (rare but possible), or missing
        # Usually ISIC is single-label. We take the argmax.
        label_idx = int(labels.argmax())

        if self.transform:
            image = self.transform(image)

        return image, label_idx
