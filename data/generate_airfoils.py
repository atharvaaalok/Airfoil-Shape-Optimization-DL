import numpy as np

from airfoil_utils import generate_airfoil_singlefile
from airfoil_utils import generate_airfoil_variants


# Set the number of HV, MV and LV airfoils and the noise levels used to generate them
HV_details = {'name': 'HV', 'count': 2, 'noise': 0.1}
MV_details = {'name': 'MV', 'count': 2, 'noise': 0.01}
LV_details = {'name': 'LV', 'count': 2, 'noise': 0.001}


airfoil_set = 'train'
## 1 - original
# Generate a single file of all original airfoils and corresponding L by D ratios
generate_airfoil_singlefile(airfoil_set)

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