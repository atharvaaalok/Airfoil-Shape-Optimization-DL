from airfoil_utils.compute_L_by_D import compute_L_by_D
import numpy as np
from airfoil_utils.generate_airfoil_parameterization import get_catmullrom_points
import torch


# Load airfoils
# Load control points for the airfoils
airfoil_set = 'train'
filename = 'airfoil_data'
data = np.load(f'generated_airfoils/{airfoil_set}/{filename}.npz')
P_all = data['P']
L_by_D_all = data['L_by_D']
print(f'Total airfoils: {P_all.shape[0]}')


idx = np.random.randint(P_all.shape[0])
P = P_all[idx, :]
L_by_D = L_by_D_all[idx]

X_fit = get_catmullrom_points(torch.tensor(P.reshape(-1, 2)), num_sample_pts = 201).numpy()

L_by_D_computed = compute_L_by_D(X_fit.flatten())

print(L_by_D)
print(L_by_D_computed)