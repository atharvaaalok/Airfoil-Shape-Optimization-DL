import numpy as np

from .airfoil_parameterization import fit_catmullrom, get_catmullrom_points
from .compute_L_by_D import compute_L_by_D


def generate_airfoil_parameterization(airfoil_set, num_control_pts, num_sample_pts):
    # Load the airfoils
    data = np.load(f'generated_airfoils/{airfoil_set}/original_coordinates.npz')
    X_orig = data['X']

    total_airfoils = X_orig.shape[0]

    
    # Create dummy array to hold parametrized airfoils
    X_all = np.zeros((total_airfoils, num_control_pts * 2))

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
        X_all[i, :] = P_tensor.numpy().flatten()

        # Compute the L by D ratio of the airfoil
        X_fit = get_catmullrom_points(P_tensor, num_sample_pts)
        L_by_D = compute_L_by_D(X_fit.flatten())
        L_by_D_all[i] = L_by_D
        print('param', i)
    

    # Save the airfoils and their L by D ratios to file
    save_filename = f'generated_airfoils/{airfoil_set}/original'
    np.savez(save_filename, X = X_all, L_by_D = L_by_D_all)