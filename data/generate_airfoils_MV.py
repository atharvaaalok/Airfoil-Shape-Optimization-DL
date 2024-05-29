import numpy as np

from airfoil_noise_addition import airfoil_noise_addition
from compute_L_by_D import compute_L_by_D


def generate_airfoils_MV(total_MV: int, noise_MV: float) -> None:
    """Creates mid variance airfoils for each high variance airfoil previously generated.
    
    Args:
        total_MV: The number of mid variance airfoils to be created for each high variance airfoil.
        noise_MV: Noise level used to create mid variance airfoils from the high variance ones.
    """

    # Load the high variance airfoils
    X_HV_all = np.load('generated_airfoils/airfoils_HV.npy')

    total_airfoils, total_HV, num_coordinates = X_HV_all.shape
    # Create dummy array to hold MV airfoils
    X_MV_all = np.zeros((total_airfoils, total_HV, total_MV, num_coordinates))

    # Create dummy array to hold L by D ratios
    L_by_D_all = np.zeros((total_airfoils, total_HV, total_MV))


    # Generate mid variance airfoils for each high variance airfoil
    for i in range(total_airfoils):
        for j in range(total_HV):
            X_HV = X_HV_all[i, j, :]
            # Generate MV airfoils
            for k in range(total_MV):
                X_new = airfoil_noise_addition(X_HV, noise_MV)
                # Store the airfoil in the list of MV airfoils
                X_MV_all[i, j, k, :] = X_new

                # Compute L by D ratio of the airfoil
                L_by_D = compute_L_by_D(X_new)
                L_by_D_all[i, j, k] = L_by_D


    # Save the mid variance airfoils and their L by D ratios to file
    np.save('generated_airfoils/airfoils_MV.npy', X_MV_all)
    np.save('generated_airfoils/L_by_D_MV.npy', L_by_D_all)