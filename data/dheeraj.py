import numpy as np
import matplotlib.pyplot as plt
from airfoil_utils.airfoil_parameterization import fit_catmullrom, get_catmullrom_points


# Load any airfoil
airfoil_num = 0
with open('airfoil_database/airfoil_names.txt', 'r') as f_names:
    i = 0
    for name in f_names:
        if i == airfoil_num:
            airfoil_name = name[:-1]
            break
        i += 1

# Get the coordinates for the airfoil
X = np.loadtxt(f'airfoil_database/airfoils/{airfoil_name}.dat')

X_centroid = np.mean(X, axis = 0)
X = X - X_centroid

# Set spline properties
num_control_pts = 12
# Fit spline and get the control points
P_tensor = fit_catmullrom(X.flatten(), num_control_pts)

X_fit = get_catmullrom_points(P_tensor, num_sample_pts = 501)


# Plot the original points
plt.plot(X[:, 0], X[:, 1])
# Plot the fitted curve
plt.plot(X_fit[:, 0], X_fit[:, 1])

plt.axis('equal')
plt.show()