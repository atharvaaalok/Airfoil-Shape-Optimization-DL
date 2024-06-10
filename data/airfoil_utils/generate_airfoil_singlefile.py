import numpy as np

from .compute_L_by_D import compute_L_by_D
from .progress_bar import print_progress_bar


def generate_airfoil_singlefile(airfoil_set: str) -> None:
    """Combines all the airfoil coordinates into a single array with each row as one airfoil.
    
    Args:
        airfoil_set: Determines whether the set is 'train', 'dev' or 'test'.
    """

    # Get the total number of airfoils by counting the number of lines in the corresponding file
    airfoil_set_filename = f'airfoil_database/airfoil_{airfoil_set}.txt'
    with open(airfoil_set_filename, 'r') as f:
        airfoil_names = f.readlines()
        total_airfoils = len(airfoil_names)
    
    # Get the name of the first airfoil, remove the '\n' at the end of the name
    first_airfoil_name = airfoil_names[0][:-1]
    # Get the number of coordinates (x, y) for the first airfoil
    X_first_airfoil = np.loadtxt(f'airfoil_database/airfoils/{first_airfoil_name}.dat')
    pts_on_airfoil = X_first_airfoil.shape[0]

    
    # Create dummy array to hold airfoils in the first dimension
    # Each row will be x1, y1, x2, y2...xn, yn for the ith airfoil in the file
    X_all = np.zeros((total_airfoils, pts_on_airfoil * 2))

    # Create dummy array to hold airfoil L by D ratios
    L_by_D_all = np.zeros(total_airfoils)


    # For each airfoil in the file open its coordinate file and store data in X_all
    with open(airfoil_set_filename, 'r') as f_names:
        for i, airfoil_name in enumerate(f_names):
            # Remove the '\n' at the end of the name
            airfoil_name = airfoil_name[:-1]
            # Read the corresponding data file
            X = np.loadtxt(f'airfoil_database/airfoils/{airfoil_name}.dat')
            # Flatten X into a 1D array of following order: x1, y1, x2, y2...
            X_flat = X.flatten()
            
            # Store the coordinates in the X_all array
            X_all[i, :] = X_flat

            # Compute L by D ratio of the airfoil
            L_by_D = compute_L_by_D(X_flat)
            L_by_D_all[i] = L_by_D

            if i % (total_airfoils // 20) == 0 or i == total_airfoils - 1:
                print_progress_bar(iteration = i, total_iterations = total_airfoils)
    

    # Save the airfoils and their L by D ratios to file
    save_filename = f'generated_airfoils/{airfoil_set}/original_coordinates'
    np.savez(save_filename, X = X_all, L_by_D = L_by_D_all)