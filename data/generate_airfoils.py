import os
import shutil
import multiprocessing
import time

import numpy as np

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

# Choose the airfoil data set to generate
airfoil_set = 'train'
dataset_count_list = [i for i in range(1, 11)]


def gen_orig_param(airfoil_set):
    ## 1 - original
    # Generate a single file of all original airfoils and corresponding L by D ratios
    print('Generating - 1.1 original_coordinates')
    generate_airfoil_singlefile(airfoil_set)
    # Generate corresponding parameterized representation file
    print('Generating - 1.2 original')
    generate_airfoil_parameterization(airfoil_set, num_control_pts, num_sample_pts)


def gen_air(airfoil_set, dataset_count):
    ## 2 - original_LV
    # Generate low variance airfoils from the original airfoils
    airfoil_source = 'original'
    print('Generating - 2. original_LV')
    generate_airfoil_variants(airfoil_set, dataset_count, airfoil_source, LV_details, num_sample_pts)

    ## 3 - original_MV
    airfoil_source = 'original'
    print('Generating - 3. original_MV')
    generate_airfoil_variants(airfoil_set, dataset_count, airfoil_source, MV_details, num_sample_pts)

    ## 4 - original_HV
    airfoil_source = 'original'
    print('Generating - 4. original_HV')
    generate_airfoil_variants(airfoil_set, dataset_count, airfoil_source, HV_details, num_sample_pts)

    ## 5 - original_MV_LV
    airfoil_source = 'original_MV'
    print('Generating - 5. original_MV_LV')
    generate_airfoil_variants(airfoil_set, dataset_count, airfoil_source, LV_details, num_sample_pts)

    ## 6 - original_HV_LV
    airfoil_source = 'original_HV'
    print('Generating - 6. original_HV_LV')
    generate_airfoil_variants(airfoil_set, dataset_count, airfoil_source, LV_details, num_sample_pts)

    ## 6 - original_HV_MV
    airfoil_source = 'original_HV'
    print('Generating - 7. original_HV_MV')
    generate_airfoil_variants(airfoil_set, dataset_count, airfoil_source, MV_details, num_sample_pts)

    ## 7 - original_HV_MV_LV
    airfoil_source = 'original_HV_MV'
    print('Generating - 8. original_HV_MV_LV')
    generate_airfoil_variants(airfoil_set, dataset_count, airfoil_source, LV_details, num_sample_pts)




if __name__ == '__main__':
    
    start_time = time.time()

    
    # Create dataset_count directories, if already exist raise error, don't allow to overwrite
    for dataset_count in dataset_count_list:
        dir_name = f'generated_airfoils/{airfoil_set}/{dataset_count}'
        if os.path.exists(dir_name):
            raise ValueError('Directory already exists. Cannot overwrite data.')
        else:
            os.makedirs(dir_name)
    

    # Generate airfoils single file and their parameterizations
    gen_orig_param(airfoil_set)

    # Copy airfoils single file and their parameterizations to each dataset folder
    for dataset_count in dataset_count_list:
        # Copy original coordinates file
        src = f'generated_airfoils/{airfoil_set}/original_coordinates.npz'
        dst = f'generated_airfoils/{airfoil_set}/{dataset_count}/original_coordinates.npz'
        shutil.copyfile(src, dst)

        # Copy parameterization file
        src = f'generated_airfoils/{airfoil_set}/original.npz'
        dst = f'generated_airfoils/{airfoil_set}/{dataset_count}/original.npz'
        shutil.copyfile(src, dst)

    
    # Create processes for multiprocessing
    processes = []
    
    for dataset_count in dataset_count_list:
        p = multiprocessing.Process(target = gen_air, args = [airfoil_set, dataset_count])
        p.start()
        processes.append(p)
    
    for process in processes:
        process.join()


    finish_time = time.time()
    time_taken = finish_time - start_time
    HV_noise = HV_details['noise']
    with open(f'time_taken.txt', 'w') as f:
        f.write(str(time_taken))