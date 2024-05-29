import numpy as np

from airfoil_noise_addition import airfoil_noise_addition


def generate_airfoils_HV(total_HV: int, noise_HV: float) -> None:
    """Creates high variance airfoils for each original airfoil in the database.
    
    Args:
        total_HV: The number of high variance airfoils to be created for each original airfoil.
        noise_HV: Noise level used to create high variance airfoils from the original ones.
    """

    # Load the original airfoils
    X_all = np.load('generated_airfoils/airfoils_original.npy')

    total_airfoils, num_coordinates = X_all.shape
    # Create dummy array to hold HV airfoils
    X_HV_all = np.zeros((total_airfoils, total_HV, num_coordinates))
    

    # Generate high variance airfoils for each original airfoil
    for i in range(total_airfoils):
        X_orig = X_all[i, :]
        # Generate HV airfoils
        for j in range(total_HV):
            X_new = airfoil_noise_addition(X_orig, noise_HV)
            # Store the airfoil in the list of HV airfoils
            X_HV_all[i, j, :] = X_new


    # Save the high variance airfoils created in a single numpy file
    np.save('generated_airfoils/airfoils_HV.npy', X_HV_all)