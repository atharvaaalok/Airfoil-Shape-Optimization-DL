import numpy as np

from .airfoil_noise_addition import airfoil_noise_addition
from .compute_L_by_D import compute_L_by_D


def generate_airfoil_variants(airfoil_set: str, airfoil_source: str, variance_details: dict) -> None:
    """Creates new airfoils from previously generated airfoils by adding specified variance.
    
    Args:
        airfoil_set: The airfoil data set ('train' etc.) for which the variants will be created.
        airfoil_source: The airfoil data file from which new variants are to be generated.
        variance_details: A dictionary containing the name, count and noise level of the variance to
            be added.
    """

    var_name = variance_details['name']
    var_count = variance_details['count']
    var_noise = variance_details['noise']

    # Load the airfoils from source
    airfoil_source_filename = f'generated_airfoils/{airfoil_set}/{airfoil_source}'
    data = np.load(airfoil_source_filename + '.npz')
    X = data['X']

    total_airfoils = X.shape[0]
    num_coordinates = X.shape[1]

    # Create dummy array to hold LV airfoils
    total_LV_airfoils = total_airfoils * var_count
    X_all = np.zeros((total_LV_airfoils, num_coordinates))

    # Create dummy array to hold L by D ratios
    L_by_D_all = np.zeros(total_LV_airfoils)


    # Generate low variance airfoils for each airfoil
    for i in range(total_airfoils):
        X_i = X[i, :]
        # Generate LV airfoils
        for j in range(var_count):
            X_new = airfoil_noise_addition(X_i, var_noise)
            # Store the airfoil in the X_all array
            idx = i * var_count + j
            X_all[idx, :] = X_new

            # Compute L by D ratio of the airfoil
            L_by_D = compute_L_by_D(X_new)
            L_by_D_all[idx] = L_by_D
    

    # Save the airfoils and their L by D ratios to file
    save_filename = airfoil_source_filename + f'_{var_name}'
    np.savez(save_filename, X = X_all, L_by_D = L_by_D_all)