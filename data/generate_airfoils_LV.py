import numpy as np

from airfoil_noise_addition import airfoil_noise_addition


def generate_airfoils_LV(total_LV: int, noise_LV: float) -> None:
    """Creates mid variance airfoils for each high variance airfoil previously generated.
    
    Args:
        total_LV: The number of low variance airfoils to be created for each mid variance airfoil.
        noise_LV: Noise level used to create low variance airfoils from the mid variance ones.
    """

    # Load mid variance airfoils
    X_MV_all = np.load('generated_airfoils/airfoils_MV.npy')

    total_airfoils, total_HV, total_MV, num_coordinates = X_MV_all.shape
    # Create dummy array to hold LV airfoils
    X_LV_all = np.zeros((total_airfoils, total_HV, total_MV, total_LV, num_coordinates))


    # Generate low variance airfoils for each mid variance airfoil
    for i in range(total_airfoils):
        for j in range(total_HV):
            for k in range(total_MV):
                X_MV = X_MV_all[i, j, k, :]
                # Generate LV airfoils
                for l in range(total_LV):
                    X_new = airfoil_noise_addition(X_MV, noise_LV)
                    # Store the airfoil in the list of LV airfoils
                    X_LV_all[i, j, k, l, :] = X_new


    # Save the low variance airfoils created in a single numpy file
    np.save('generated_airfoils/airfoils_LV.npy', X_LV_all)