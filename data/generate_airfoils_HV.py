import numpy as np
import matplotlib.pyplot as plt

from airfoil_noise_addition import airfoil_noise_addition


# Load the original airfoils
X_all = np.load('generated_airfoils/airfoils_original.npy')

# Decide the number of high variance airfoils to be created for each original airfoil
total_HV_airfoils = 2

# Create dummy array to hold all high variance airfoils created from each original airfoil
total_airfoils = X_all.shape[0]
num_coordinates = X_all.shape[1]
X_HV_all = np.zeros((total_airfoils, total_HV_airfoils, num_coordinates))


# Decide noise level for creating high variance airfoils
noise_level_HV = 0.1

# Generate high variance airfoils for each original airfoil
for i in range(total_airfoils):
    X_orig = X_all[i, :]
    # Generate HV airfoils
    for j in range(total_HV_airfoils):
        X_new = airfoil_noise_addition(X_orig, noise_level_HV)
        # Store the airfoil in the list of HV airfoils
        X_HV_all[i, j, :] = X_new


# Save the high variance airfoils created in a single numpy file
np.save('generated_airfoils/airfoils_HV.npy', X_HV_all)