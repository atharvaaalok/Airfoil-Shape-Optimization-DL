import torch
from torch.utils.data import Dataset
import numpy as np


class AirfoilDataset(Dataset):
    def __init__(self, file_path):
        # Load the data from the combined data file
        data = np.load(file_path)
        self.P = torch.tensor(data['P'], dtype = torch.float32)
        self.L_by_D = torch.tensor(data['L_by_D'], dtype = torch.float32).reshape(-1, 1)
    

    def __len__(self):
        return self.P.shape[0]


    def __getitem__(self, idx):
        # Return the sample at the given index
        x = self.P[idx]
        y = self.L_by_D[idx]
        return x, y