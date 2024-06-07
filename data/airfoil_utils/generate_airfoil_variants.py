import logging

import numpy as np
import torch

from .airfoil_noise_addition import airfoil_noise_addition
from .compute_L_by_D import compute_L_by_D
from .airfoil_parameterization import get_catmullrom_points
from .progress_bar import print_progress_bar


def generate_airfoil_variants(airfoil_set: str, airfoil_source: str, variance_details: dict, num_sample_pts: int) -> None:
    """Creates new airfoils from previously generated airfoils by adding specified variance.
    
    Args:
        airfoil_set: The airfoil data set ('train' etc.) for which the variants will be created.
        airfoil_source: The airfoil data file from which new variants are to be generated.
        variance_details: A dictionary containing the name, count and noise level of the variance to
            be added.
    """

    airfoil_source_filename = f'generated_airfoils/{airfoil_set}/{airfoil_source}'
    var_name = variance_details['name']
    fname = airfoil_source_filename + f'_{var_name}' + '.log'

    logger = logging.getLogger(fname[:-4] + '_Logger')
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(fname)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


    var_name = variance_details['name']
    var_count = variance_details['count']
    var_noise = variance_details['noise']

    # Load the airfoils from source
    airfoil_source_filename = f'generated_airfoils/{airfoil_set}/{airfoil_source}'
    data = np.load(airfoil_source_filename + '.npz')
    P = data['P']

    total_airfoils = P.shape[0]
    num_coordinates = P.shape[1]

    # Create dummy array to hold new airfoils
    total_new_airfoils = total_airfoils * var_count
    P_all = np.zeros((total_new_airfoils, num_coordinates))

    # Create dummy array to hold L by D ratios
    L_by_D_all = np.zeros(total_new_airfoils)


    # Generate new airfoils for each airfoil
    for i in range(total_airfoils):
        P_i = P[i, :]
        # Generate new airfoils
        for j in range(var_count):
            # Generate new airfoils until one is generated that has a defined L by D ratio
            L_by_D = np.nan

            c = 0
            while np.isnan(L_by_D):
                c += 1
                P_new = airfoil_noise_addition(P_i, var_noise)
                # Position centroid of spline control points at (0, 0)
                P_new = P_new.reshape(-1, 2)
                P_new_centroid = np.mean(P_new, axis = 0)
                P_new = P_new - P_new_centroid
                P_new = P_new.flatten()

                # Generate sample points on the spline
                X_fit = get_catmullrom_points(torch.tensor(P_new.reshape(-1, 2)), num_sample_pts).numpy()
                
                # Compute L by D ratio of the airfoil
                L_by_D = compute_L_by_D(X_fit.flatten())
            
            logger.debug(c)
            
            # Store the airfoil in the P_all array
            idx = i * var_count + j
            P_all[idx, :] = P_new
            
            L_by_D_all[idx] = L_by_D
        
        if i % (total_airfoils // 20) == 0 or i == total_airfoils - 1:
                print_progress_bar(iteration = i, total_iterations = total_airfoils)
    

    # Save the airfoils and their L by D ratios to file
    save_filename = airfoil_source_filename + f'_{var_name}'
    np.savez(save_filename, P = P_all, L_by_D = L_by_D_all)