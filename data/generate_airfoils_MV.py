import numpy as np
import matplotlib.pyplot as plt

from airfoil_noise_addition import airfoil_noise_addition


# Load the high variance airfoils
X_HV_all = np.load('generated_airfoils/airfoils_HV.npy')

# Decide the number of mid variance airfoils to be created for each high variance airfoil
total_MV_airfoils = 2

# Create dummy array to hold all mid variance airfoils created from each high variance airfoil
total_airfoils = X_HV_all.shape[0]
total_HV_airfoils = X_HV_all.shape[1]
num_coordinates = X_HV_all.shape[2]
X_MV_all = np.zeros((total_airfoils, total_HV_airfoils, total_MV_airfoils, num_coordinates))


# Decide noise level for creating mid variance airfoils
noise_level_MV = 0.01

# Generate mid variance airfoils for each high variance airfoil
for i in range(total_airfoils):
    for j in range(total_HV_airfoils):
        X_HV = X_HV_all[i, j, :]
        # Generate MV airfoils
        for k in range(total_MV_airfoils):
            X_new = airfoil_noise_addition(X_HV, noise_level_MV)
            # Store the airfoil in the list of MV airfoils
            X_MV_all[i, j, k, :] = X_new


# Save the mid variance airfoils created in a single numpy file
np.save('generated_airfoils/airfoils_MV.npy', X_MV_all)