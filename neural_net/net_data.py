import torch
from torch.utils.data import Dataset
import numpy as np


class AirfoilDataset(Dataset):
    def __init__(self, file_path):
        # Load the data from the combined data file
        data = np.load(file_path)
        airfoil_count = 900
        self.P = data['P'][:airfoil_count, :]
        self.L_by_D = data['L_by_D'][:airfoil_count]
    

    def __len__(self):
        return self.P.shape[0]


    def __getitem__(self, idx):
        # Return the sample at the given index
        x = self.P[idx]
        y = self.L_by_D[idx]
        y = y.reshape(1, )
        return torch.tensor(x, dtype = torch.float32),  torch.tensor(y, dtype = torch.float32)