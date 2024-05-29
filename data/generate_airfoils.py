import numpy as np

from generate_airfoils_singlefile import generate_airfoils_singlefile
from generate_airfoils_HV import generate_airfoils_HV
from generate_airfoils_MV import generate_airfoils_MV
from generate_airfoils_LV import generate_airfoils_LV


# Set the number of HV, MV and LV airfoils and the noise levels used to generate them
total_HV, noise_HV = 2, 0.1
total_MV, noise_MV = 2, 0.01
total_LV, noise_LV = 2, 0.001


# Generate a single file of all original airfoils
generate_airfoils_singlefile()

# Generate high variance airfoils for each original airfoil and save in a single file
generate_airfoils_HV(total_HV, noise_HV)

# Generate mid variance airfoils for each high variance airfoil and save in a single file
generate_airfoils_MV(total_MV, noise_MV)

# Generate low variance airfoils for each mid variance airfoil and save in a single file
generate_airfoils_LV(total_LV, noise_LV)