import numpy as np

from airfoil_utils import generate_airfoil_singlefile
from airfoil_utils import generate_airfoil_parameterization
from airfoil_utils import generate_airfoil_variants


# Set the number of HV, MV and LV airfoils and the noise levels used to generate them
HV_details = {'name': 'HV', 'count': 2, 'noise': 0.7}
MV_details = {'name': 'MV', 'count': 2, 'noise': 0.3}
LV_details = {'name': 'LV', 'count': 2, 'noise': 0.1}

# Set parameterization details
num_control_pts = 12
num_sample_pts = 201


airfoil_set = 'train'
## 1 - original
# Generate a single file of all original airfoils and corresponding L by D ratios
generate_airfoil_singlefile(airfoil_set)
# Generate corresponding parameterized representation file
generate_airfoil_parameterization(airfoil_set, num_control_pts, num_sample_pts)

exit()

## 2 - original_LV
# Generate low variance airfoils from the original airfoils
airfoil_source = 'original'
generate_airfoil_variants(airfoil_set, airfoil_source, LV_details)

## 3 - original_MV
airfoil_source = 'original'
generate_airfoil_variants(airfoil_set, airfoil_source, MV_details)

## 4 - original_HV
airfoil_source = 'original'
generate_airfoil_variants(airfoil_set, airfoil_source, HV_details)

## 5 - original_MV_LV
airfoil_source = 'original_MV'
generate_airfoil_variants(airfoil_set, airfoil_source, LV_details)

## 6 - original_HV_MV
airfoil_source = 'original_HV'
generate_airfoil_variants(airfoil_set, airfoil_source, MV_details)

## 7 - original_HV_MV_LV
airfoil_source = 'original_HV_MV'
generate_airfoil_variants(airfoil_set, airfoil_source, LV_details)