import os

import numpy as np


dataset_count_list = [i for i in range(1, 101)]
airfoil_set = 'train'

# Initialize lists to hold each dataset's combined data arrays
P_combine_all_dataset = []
L_by_D_combine_all_dataset = []


# 1. original
data_original = np.load(f'generated_airfoils/{airfoil_set}/original.npz')
P_orig = data_original['P']
L_by_D_orig = data_original['L_by_D']
P_combine_all_dataset.append(P_orig)
L_by_D_combine_all_dataset.append(L_by_D_orig)


for dataset_count in dataset_count_list:
    dir_name = f'generated_airfoils/{airfoil_set}/{dataset_count}/'

    # If directory doesn't exist then exit and only use the original airfoils to create dataset
    if not os.path.exists(dir_name):
        print('Exiting after only combining original airfoils.')
        break

    P_combine = []
    L_by_D_combine = []

    # 2. original_LV
    data = np.load(dir_name + 'original_LV.npz')
    P_combine.append(data['P'])
    L_by_D_combine.append(data['L_by_D'])

    # 3. original_MV
    data = np.load(dir_name + 'original_MV.npz')
    P_combine.append(data['P'])
    L_by_D_combine.append(data['L_by_D'])

    # 4. original_HV
    data = np.load(dir_name + 'original_HV.npz')
    P_combine.append(data['P'])
    L_by_D_combine.append(data['L_by_D'])

    # 5. original_MV_LV
    data = np.load(dir_name + 'original_MV_LV.npz')
    P_combine.append(data['P'])
    L_by_D_combine.append(data['L_by_D'])

    # 6. original_HV_LV
    data = np.load(dir_name + 'original_HV_LV.npz')
    P_combine.append(data['P'])
    L_by_D_combine.append(data['L_by_D'])

    # 7. original_HV_MV
    data = np.load(dir_name + 'original_HV_MV.npz')
    P_combine.append(data['P'])
    L_by_D_combine.append(data['L_by_D'])

    # 8. original_HV_MV_LV
    data = np.load(dir_name + 'original_HV_MV_LV.npz')
    P_combine.append(data['P'])
    L_by_D_combine.append(data['L_by_D'])


    # Stack all the data into a single array
    P_combine = np.vstack(P_combine)
    L_by_D_combine = np.hstack(L_by_D_combine)

    # Store in a combined data file for each dataset
    np.savez(dir_name + 'combined_data.npz', P = np.vstack([P_orig, P_combine]), L_by_D = np.hstack([L_by_D_orig, L_by_D_combine]))
    

    # Append data to all dataset list
    P_combine_all_dataset.append(P_combine)
    L_by_D_combine_all_dataset.append(L_by_D_combine)


# Stack all the datasets combined data into a single giant array
P_combine_all_dataset = np.vstack(P_combine_all_dataset)
L_by_D_combine_all_dataset = np.hstack(L_by_D_combine_all_dataset)

# Store in a single data file
np.savez(f'generated_airfoils/{airfoil_set}/' + 'airfoil_data.npz',
         P = P_combine_all_dataset,
         L_by_D = L_by_D_combine_all_dataset)