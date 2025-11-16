import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder 
import pandas as pd
import numpy as np
import os
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

class StrokeDataset(Dataset):
    def __init__(self, features_path, labels_path, categorical_cols, numerical_cols):

        self.X_data = pd.read_csv(features_path)
        self.y_data = pd.read_csv(labels_path)
        
        self.categorical_cols = categorical_cols
        self.numerical_cols = numerical_cols
        
        self.x_categ = self.X_data[self.categorical_cols].to_numpy(dtype=np.int64)
        self.x_cont = self.X_data[self.numerical_cols].to_numpy(dtype=np.float32)
        self.labels = self.y_data.to_numpy(dtype=np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x_categ = self.x_categ[idx]
        x_cont = self.x_cont[idx]
        label = self.labels[idx]
        
        return torch.tensor(x_categ, dtype=torch.long), \
               torch.tensor(x_cont, dtype=torch.float), \
               torch.tensor(label, dtype=torch.float)
    

class MultiModalStrokeDataset(Dataset):
    def __init__(self, csv_path, image_root_dir, image_path_col, categorical_cols, numerical_cols, target_col, transforms=None):
        self.df = pd.read_csv(csv_path)
        self.image_root_dir = image_root_dir
        self.image_path_col = image_path_col
        self.transforms = transforms
        
        self.x_categ = self.df[categorical_cols].to_numpy(dtype=np.int64)
        self.x_cont = self.df[numerical_cols].to_numpy(dtype=np.float32)
        self.labels = self.df[target_col].to_numpy(dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x_categ = torch.tensor(self.x_categ[idx], dtype=torch.long)
        x_cont = torch.tensor(self.x_cont[idx], dtype=torch.float)
        label = torch.tensor(self.labels[idx], dtype=torch.float).unsqueeze(0)

        relative_path_from_csv = self.df.loc[idx, self.image_path_col]
        
        normalized_relative_path = relative_path_from_csv.replace('\\', '/')
        
        image_path = os.path.join(self.image_root_dir, normalized_relative_path)

        image = np.array(Image.open(image_path).convert("RGB")) 

        if self.transforms:
            image = self.transforms(image=image)['image']
            
        return {
            'image': image,
            'x_categ': x_categ,
            'x_cont': x_cont,
            'label': label
        }

def get_transforms(image_size, mean, std):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

class ImageFolderWrapper(ImageFolder):

    def __init__(self, root, transform=None):
        super().__init__(root, transform=None) 
        self.albumentations_transform = transform

    def __getitem__(self, index):
        path, label = self.samples[index]
        
        image = np.array(Image.open(path).convert("RGB"))
        
        if self.albumentations_transform:
            transformed = self.albumentations_transform(image=image)
            image = transformed['image']
            
        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.float).unsqueeze(0)
        }