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
        for idx in range(len(self.labels_frame)):
            img_name = self.labels_frame.iloc[idx, 0] + ".jpg" # Assuming .jpg extension
            img_path = os.path.join(self.root_dir, img_name)
            if os.path.exists(img_path):
                self.valid_indices.append(idx)
                
        print(f"Loaded ISIC dataset. Found {len(self.valid_indices)} valid images out of {len(self.labels_frame)} entries.")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        real_idx = self.valid_indices[idx]
        img_name = self.labels_frame.iloc[real_idx, 0] + ".jpg"
        img_path = os.path.join(self.root_dir, img_name)
        
        image = Image.open(img_path).convert('RGB')
        
        # Get one-hot labels and convert to class index
        labels = self.labels_frame.iloc[real_idx, 1:].values.astype('float')
        # Handle cases where multiple classes might be marked (rare but possible), or missing
        # Usually ISIC is single-label. We take the argmax.
        label_idx = int(labels.argmax())

        if self.transform:
            image = self.transform(image)

        return image, label_idx
