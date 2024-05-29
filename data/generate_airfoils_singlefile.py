import numpy as np


# Get the total number of airfoils by counting the number of lines in the airfoil_names.txt file
with open('airfoil_names.txt', 'r') as f:
    airfoil_names = f.readlines()
    total_airfoils = len(airfoil_names)

# Get the name of the first airfoil, remove the '\n' at the end of the name
first_airfoil_name = airfoil_names[0][:-1]
# Get the number of coordinates (x, y) for the first airfoil
X_first_airfoil = np.loadtxt(f'airfoil_database/{first_airfoil_name}.dat')
pts_on_airfoil = X_first_airfoil.shape[0]


# Create dummy vector to hold airfoils in the first dimension
# Each row will be x1, y1, x2, y2...xn, yn for the ith airfoil in the airfoil_names.txt file
X_all = np.zeros((total_airfoils, 2 * pts_on_airfoil))


# For each airfoil in the names file open its coordinate file and store data in X_all
with open('airfoil_names.txt', 'r') as f_names:
    for i, airfoil_name in enumerate(f_names):
        # Remove the '\n' at the end of the name
        airfoil_name = airfoil_name[:-1]
        # Read the corresponding data file
        X = np.loadtxt(f'airfoil_database/{airfoil_name}.dat')
        # Flatten X into a 1D array of following order: x1, y1, x2, y2...
        X_flat = X.flatten()
        
        # Store the coordinates in the X_all array
        X_all[i, :] = X_flat


# Save the array in file
np.save('generated_airfoils/airfoils_original.npy', X_all)