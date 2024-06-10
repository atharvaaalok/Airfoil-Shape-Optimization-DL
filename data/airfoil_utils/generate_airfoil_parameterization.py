import numpy as np

from .airfoil_parameterization import fit_catmullrom, get_catmullrom_points
from .compute_L_by_D import compute_L_by_D
from .progress_bar import print_progress_bar


def generate_airfoil_parameterization(airfoil_set: str, num_control_pts: int, num_sample_pts: int):
    # Load the airfoils
    data = np.load(f'generated_airfoils/{airfoil_set}/original_coordinates.npz')
    X_orig = data['X']

    total_airfoils = X_orig.shape[0]

    
    # Create dummy array to hold parametrized airfoils
    P_all = np.zeros((total_airfoils, num_control_pts * 2))

    # Create dummy array to hold parameterized airfoil L by D ratios
    L_by_D_all = np.zeros(total_airfoils)


    # For each airfoil fit a spline and store the parameterization in X_all
    for i in range(total_airfoils):
        X = X_orig[i, :].reshape(-1, 2)

        # Shift centroid to origin
        X_centroid = np.mean(X, axis = 0)
        X = X - X_centroid

        # Fit spline and get the control points
        P_tensor = fit_catmullrom(X.flatten(), num_control_pts)
        
        # Store the control points in the X_all array
        P_all[i, :] = P_tensor.numpy().flatten()

        # Compute the L by D ratio of the airfoil
        X_fit = get_catmullrom_points(P_tensor, num_sample_pts).numpy()
        L_by_D = compute_L_by_D(X_fit.flatten())
        L_by_D_all[i] = L_by_D

        if i % (total_airfoils // 20) == 0 or i == total_airfoils - 1:
                print_progress_bar(iteration = i, total_iterations = total_airfoils)
    

    # Save the airfoils and their L by D ratios to file
    save_filename = f'generated_airfoils/{airfoil_set}/original'
    np.savez(save_filename, P = P_all, L_by_D = L_by_D_all)