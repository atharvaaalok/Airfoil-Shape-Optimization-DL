import numpy as np
import torch


from airfoil_utils.airfoil_parameterization import get_catmullrom_points
from airfoil_utils.compute_L_by_D import compute_L_by_D


# Load airfoil data
airfoil_set = 'train'
data = np.load(f'generated_airfoils/{airfoil_set}/airfoil_data.npz')
P_all = data['P']
L_by_D_all = data['L_by_D']

print(P_all.shape)
print(L_by_D_all.shape)

idx = np.random.randint(P_all.shape[0])
P = P_all[idx, :]
L_by_D = L_by_D_all[idx]

# Sample points on the spline
X_fit = get_catmullrom_points(torch.tensor(P.reshape(-1, 2)), num_sample_pts = 201).numpy()


# Compute L by D ratio
L_by_D_computed = compute_L_by_D(X_fit.flatten())
print(L_by_D)
print(L_by_D_computed)