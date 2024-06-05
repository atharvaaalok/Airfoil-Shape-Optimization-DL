import numpy as np

from airfoil_utils import generate_airfoils_singlefile
from airfoil_utils import generate_airfoils_LV


# Set the number of HV, MV and LV airfoils and the noise levels used to generate them
total_HV, noise_HV = 2, 0.1
total_MV, noise_MV = 2, 0.01
total_LV, noise_LV = 2, 0.001


## 1
# Generate a single file of all original airfoils and corresponding L by D ratios
airfoil_set = 'train'
generate_airfoils_singlefile(airfoil_set)

## 2
# Generate low variance airfoils from the original airfoils
airfoil_source = 'original'
generate_airfoils_LV(airfoil_set, airfoil_source, total_LV, noise_LV)