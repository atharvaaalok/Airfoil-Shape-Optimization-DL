import numpy as np

from .airfoil_noise_addition import airfoil_noise_addition
from .compute_L_by_D import compute_L_by_D


def generate_airfoils_LV(airfoil_set: str, airfoil_source: str, total_LV: int, noise_LV: float) -> None:
    """Creates mid variance airfoils for each high variance airfoil previously generated.
    
    Args:
        total_LV: The number of low variance airfoils to be created for each mid variance airfoil.
        noise_LV: Noise level used to create low variance airfoils from the mid variance ones.
    """

    # Load the airfoils from source
    airfoil_source_filename = f'generated_airfoils/{airfoil_set}/{airfoil_source}'
    data = np.load(airfoil_source_filename + '.npz')
    X = data['X']

    total_airfoils = X.shape[0]
    num_coordinates = X.shape[1]

    # Create dummy array to hold LV airfoils
    total_LV_airfoils = total_airfoils * total_LV
    X_all = np.zeros((total_LV_airfoils, num_coordinates))

    # Create dummy array to hold L by D ratios
    L_by_D_all = np.zeros(total_LV_airfoils)


    # Generate low variance airfoils for each airfoil
    for i in range(total_airfoils):
        X_i = X[i, :]
        # Generate LV airfoils
        for j in range(total_LV):
            X_new = airfoil_noise_addition(X_i, noise_LV)
            # Store the airfoil in the X_all array
            idx = i * total_LV + j
            X_all[idx, :] = X_new

            # Compute L by D ratio of the airfoil
            L_by_D = compute_L_by_D(X_new)
            L_by_D_all[idx] = L_by_D
    

    # Save the airfoils and their L by D ratios to file
    save_filename = airfoil_source_filename + '_LV'
    np.savez(save_filename, X = X_all, L_by_D = L_by_D_all)