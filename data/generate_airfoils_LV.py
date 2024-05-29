import numpy as np
import matplotlib.pyplot as plt

from airfoil_noise_addition import airfoil_noise_addition


# Load mid variance airfoils
X_MV_all = np.load('generated_airfoils/airfoils_MV.npy')

# Decide the number of low variance airfoils to be created from each mid variance airfoil
total_LV_airfoils = 2

# Create dummy array to hold all low variance airfoils created from each mid variance airfoil
total_airfoils = X_MV_all.shape[0]
total_HV_airfoils = X_MV_all.shape[1]
total_MV_airfoils = X_MV_all.shape[2]
num_coordinates = X_MV_all.shape[3]
X_LV_all = np.zeros((total_airfoils, total_HV_airfoils, total_MV_airfoils, total_LV_airfoils, num_coordinates))


# Decide noise level for creating mid variance airfoils
noise_level_LV = 0.001

# Generate low variance airfoils for each mid variance airfoil
for i in range(total_airfoils):
    for j in range(total_HV_airfoils):
        for k in range(total_MV_airfoils):
            X_MV = X_MV_all[i, j, k, :]
            # Generate LV airfoils
            for l in range(total_LV_airfoils):
                X_new = airfoil_noise_addition(X_MV, noise_level_LV)
                # Store the airfoil in the list of LV airfoils
                X_LV_all[i, j, k, l, :] = X_new


# Save the low variance airfoils created in a single numpy file
np.save('generated_airfoils/airfoils_LV.npy', X_LV_all)