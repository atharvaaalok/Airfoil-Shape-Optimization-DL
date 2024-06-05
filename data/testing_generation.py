import numpy as np

from airfoil_utils import generate_airfoil_singlefile
from airfoil_utils import generate_airfoil_variants

# Set the number of HV, MV and LV airfoils and the noise levels used to generate them
HV_details = {'name': 'HV', 'count': 2, 'noise': 0.1}
MV_details = {'name': 'MV', 'count': 2, 'noise': 0.01}
LV_details = {'name': 'LV', 'count': 2, 'noise': 0.001}


## 1
# Generate a single file of all original airfoils and corresponding L by D ratios
airfoil_set = 'train'
generate_airfoil_singlefile(airfoil_set)

## 2
# Generate low variance airfoils from the original airfoils
airfoil_source = 'original'
generate_airfoil_variants(airfoil_set, airfoil_source, LV_details)