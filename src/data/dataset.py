
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

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