import numpy as np
import time

from airfoil_utils import generate_airfoil_singlefile
from airfoil_utils import generate_airfoil_parameterization
from airfoil_utils import generate_airfoil_variants


# Set the number of HV, MV and LV airfoils and the noise levels used to generate them
HV_details = {'name': 'HV', 'count': 5, 'noise': 0.7}
MV_details = {'name': 'MV', 'count': 5, 'noise': 0.3}
LV_details = {'name': 'LV', 'count': 5, 'noise': 0.1}

# Set parameterization details
num_control_pts = 12
num_sample_pts = 201

start_time = time.time()

airfoil_set = 'train'
## 1 - original
# Generate a single file of all original airfoils and corresponding L by D ratios
print('Generating - 1.1 original_coordinates')
generate_airfoil_singlefile(airfoil_set)
# Generate corresponding parameterized representation file
print('Generating - 1.2 original')
generate_airfoil_parameterization(airfoil_set, num_control_pts, num_sample_pts)

## 2 - original_LV
# Generate low variance airfoils from the original airfoils
airfoil_source = 'original'
print('Generating - 2. original_LV')
generate_airfoil_variants(airfoil_set, airfoil_source, LV_details, num_sample_pts)

## 3 - original_MV
airfoil_source = 'original'
print('Generating - 3. original_MV')
generate_airfoil_variants(airfoil_set, airfoil_source, MV_details, num_sample_pts)

## 4 - original_HV
airfoil_source = 'original'
print('Generating - 4. original_HV')
generate_airfoil_variants(airfoil_set, airfoil_source, HV_details, num_sample_pts)

## 5 - original_MV_LV
airfoil_source = 'original_MV'
print('Generating - 5. original_MV_LV')
generate_airfoil_variants(airfoil_set, airfoil_source, LV_details, num_sample_pts)

## 6 - original_HV_LV
airfoil_source = 'original_HV'
print('Generating - 6. original_HV_LV')
generate_airfoil_variants(airfoil_set, airfoil_source, LV_details, num_sample_pts)

## 6 - original_HV_MV
airfoil_source = 'original_HV'
print('Generating - 7. original_HV_MV')
generate_airfoil_variants(airfoil_set, airfoil_source, MV_details, num_sample_pts)

## 7 - original_HV_MV_LV
airfoil_source = 'original_HV_MV'
print('Generating - 8. original_HV_MV_LV')
generate_airfoil_variants(airfoil_set, airfoil_source, LV_details, num_sample_pts)


finish_time = time.time()
time_taken = finish_time - start_time
HV_noise = HV_details['noise']
with open(f'time_taken_{HV_noise}.txt', 'w') as f:
    f.write(str(time_taken))