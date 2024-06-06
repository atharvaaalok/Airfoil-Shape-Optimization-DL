import numpy as np

from airfoil_utils.compute_L_by_D import compute_L_by_D
from airfoil_utils.airfoil_parameterization import fit_catmullrom, get_catmullrom_points


# Get the name of all available airfoils
with open('airfoil_database/all_airfoil_names.txt', 'r') as f:
    airfoil_names = [name.strip() for name in f.readlines()]


# Get the names of all the airfoils that converge in xfoil and their parameterizations also do
converging_airfoils = []
for i in range(len(airfoil_names)):
    # Get airfoil
    X = np.loadtxt(f'airfoil_database/airfoils/{airfoil_names[i]}.dat')

    # Get L by D ratio of airfoil
    L_by_D_orig = compute_L_by_D(X.flatten())

    # Place centroid of airfoil at (0, 0)
    X_centroid = np.mean(X, axis = 0)
    X = X - X_centroid

    # Get the parameterization for the airfoil
    P_tensor = fit_catmullrom(X.flatten(), num_control_pts = 12)
    X_fit = get_catmullrom_points(P_tensor, num_sample_pts = 201)

    # Compute L by D ratio of the parameterized airfoil
    L_by_D_fit = compute_L_by_D(X_fit.flatten())

    # Add the name of the airfoil if both L by D ratios are not nan
    if not np.isnan(L_by_D_orig) and not np.isnan(L_by_D_fit):
        converging_airfoils.append(airfoil_names[i])
        print(i, 'Selected')
    else:
        print(i, 'Not Selected')


# Add the names of the airfoils to file
with open('airfoil_database/airfoil_names.txt', 'w') as f:
    for name in converging_airfoils:
        f.write(name + '\n')