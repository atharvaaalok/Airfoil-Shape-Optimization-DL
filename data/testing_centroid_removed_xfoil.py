import numpy as np

from airfoil_utils.compute_L_by_D import compute_L_by_D
from airfoil_utils.airfoil_parameterization import fit_catmullrom, get_catmullrom_points


# Get airfoil names
with open('airfoil_database/airfoil_names.txt', 'r') as f:
    airfoil_names = [name.strip() for name in f.readlines()]


# Get a particular airfoil's coordinates
airfoil_idx = 0
X = np.loadtxt(f'airfoil_database/airfoils/{airfoil_names[airfoil_idx]}.dat')

# Compute L by D ratio of original airfoil
L_by_D_orig = compute_L_by_D(X.flatten())

# Place centroid of airfoil at (0, 0)
X_centroid = np.mean(X, axis = 0)
X_centered = X - X_centroid


# Get the parameterization for the airfoil
P_tensor = fit_catmullrom(X.flatten(), num_control_pts = 12)
X_fit = get_catmullrom_points(P_tensor, num_sample_pts = 201)


# Compute L by D ratio of the parameterized airfoil
L_by_D_centered = compute_L_by_D(X_fit.flatten())

print(f'{L_by_D_orig = }')
print(f'{L_by_D_centered = }')