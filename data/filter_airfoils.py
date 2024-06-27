import numpy as np


airfoil_set = 'train'
data = np.load(f'generated_airfoils/{airfoil_set}/airfoil_data.npz')

P = data['P']
L_by_D = data['L_by_D']

# Sort the airfoils by L by D ratio
idx = np.argsort(L_by_D)
L_by_D = L_by_D[idx]
P = P[idx, :]

# Remove the n_largest and the n_smallest airfoils
n_largest = 25000
n_smallest = 25000
P = P[n_smallest: -n_largest, :]
L_by_D = L_by_D[n_smallest: -n_largest]


# Shuffle the airfoils instead of the current sorted order
idx = np.arange(P.shape[0])
np.random.shuffle(idx)
P = P[idx, :]
L_by_D = L_by_D[idx]


# Save the filter aifoils
np.savez(f'generated_airfoils/{airfoil_set}/airfoil_data_filtered.npz', P = P, L_by_D = L_by_D)