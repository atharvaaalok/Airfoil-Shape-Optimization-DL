import numpy as np
import matplotlib.pyplot as plt
from airfoil_utils.airfoil_parameterization import get_catmullrom_points
import torch


airfoil_set = 'train'
data_orig = np.load(f'generated_airfoils/{airfoil_set}/original_coordinates.npz')
data_fit = np.load(f'generated_airfoils/{airfoil_set}/original.npz')


idx = 0
X_orig = data_orig['X'][idx, :].reshape(-1, 2)
X_centroid = np.mean(X_orig, axis = 0)
X_orig = X_orig - X_centroid


P_fit = data_fit['X'][idx, :]
P_tensor = torch.from_numpy(P_fit)
num_sample_pts = 501
X_fit = get_catmullrom_points(P_tensor.reshape(-1, 2), num_sample_pts)


plt.plot(X_orig[:, 0], X_orig[:, 1])
plt.plot(X_fit[:, 0], X_fit[:, 1])
plt.axis('equal')
plt.show()

L_by_D_orig = data_orig['L_by_D']
L_by_D_fit = data_fit['L_by_D']

np.savetxt('L_by_D_orig', L_by_D_orig)
np.savetxt('L_by_D_fit', L_by_D_fit)