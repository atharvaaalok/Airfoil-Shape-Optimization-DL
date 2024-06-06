import numpy as np

from airfoil_utils.compute_L_by_D import compute_L_by_D


# Get the names of all the airfoils
with open('airfoil_database/airfoil_names.txt', 'r') as f:
    airfoil_names = [names.strip() for names in f.readlines()]


# Make a list of all airfoil coordinates
X_list = []
for name in airfoil_names:
    X = np.loadtxt(f'airfoil_database/airfoils/{name}.dat')
    X_list.append(X)


# Write airfoil name to new file if xfoil converges
with open('airfoil_database/airfoil_names_new.txt', 'w') as f:
    for i in range(len(airfoil_names)):
        X = X_list[i]
        L_by_D = compute_L_by_D(X)
        if not np.isnan(L_by_D):
            f.write(airfoil_names[i] + '\n')
        else:
            print(i, airfoil_names[i])