import numpy as np
import matplotlib.pyplot as plt
import torch

from airfoil_utils.airfoil_parameterization import get_catmullrom_points
from airfoil_utils.airfoil_noise_addition import airfoil_noise_addition
from airfoil_utils.compute_L_by_D import compute_L_by_D


# Load an airfoil to add noise to
airfoil_set = 'train'
airfoil_idx = 0
data = np.load(f'generated_airfoils/{airfoil_set}/original.npz')
P_orig = data['X'][airfoil_idx, :]

X_orig = get_catmullrom_points(torch.tensor(P_orig.reshape(-1, 2)), num_sample_pts = 201).numpy()


# Add noise to the original airfoil to generate new airfoils
noise = 0.7
count = 1
P_noisy = []
X_noisy = []
c = 0
for i in range(count):
    P_new = airfoil_noise_addition(P_orig, noise)
    P_noisy.append(P_new)
    X_new = get_catmullrom_points(torch.tensor(P_new.reshape(-1, 2)), num_sample_pts = 201).numpy()
    X_noisy.append(X_new)
    L_by_D = compute_L_by_D(X_new.flatten())
    if not np.isnan(L_by_D):
        c += 1

print(c, count, c / count)


# Plot the original airfoil
plt.plot(X_orig[:, 0], X_orig[:, 1], linewidth = 3)

# Plot noise added airfoils
for X_new in X_noisy:
    plt.plot(X_new[:, 0], X_new[:, 1])

plt.axis('equal')
plt.savefig('ha.svg')