import numpy as np
import matplotlib.pyplot as plt
import torch

from airfoil_utils.airfoil_parameterization import get_catmullrom_points


np.set_printoptions(linewidth = np.inf, precision = 3)


# Load control points for the airfoils
airfoil_set = 'train'
filename = 'airfoil_data_filtered'
data = np.load(f'generated_airfoils/{airfoil_set}/{filename}.npz')
P_all = data['P']
L_by_D_all = data['L_by_D']
print(f'Total airfoils: {P_all.shape[0]}')


# Sort the airfoils by L by D ratio
idx = np.argsort(L_by_D_all)
L_by_D_all = L_by_D_all[idx]
P_all = P_all[idx, :]

# Print the nth largest and smallest airfoils L by D ratios
n_largest = 25000
n_smallest = 25000
print(f'L_by_D n ({n_largest}) largest: ', L_by_D_all[-n_largest])
print(f'L_by_D n ({n_smallest}) smallest: ', L_by_D_all[n_smallest])


# Plot the airfoil with the largest and smallest L by D ratio
X_fit_largest = get_catmullrom_points(torch.tensor(P_all[-1, :].reshape(-1, 2)), num_sample_pts = 201).numpy()
X_fit_smallest = get_catmullrom_points(torch.tensor(P_all[0, :].reshape(-1, 2)), num_sample_pts = 201).numpy()
plt.plot(X_fit_largest[:, 0], X_fit_largest[:, 1], label = 'Largest')
plt.plot(X_fit_smallest[:, 0], X_fit_smallest[:, 1], label = 'Smallest')
plt.legend()
plt.axis('equal')
plt.show()


# Plot the airfoil with the nth largest and smallest L by D ratio
X_fit_n_largest = get_catmullrom_points(torch.tensor(P_all[-n_largest, :].reshape(-1, 2)), num_sample_pts = 201).numpy()
X_fit_n_smallest = get_catmullrom_points(torch.tensor(P_all[n_smallest, :].reshape(-1, 2)), num_sample_pts = 201).numpy()
plt.plot(X_fit_n_largest[:, 0], X_fit_n_largest[:, 1], label = 'n largest')
plt.plot(X_fit_n_smallest[:, 0], X_fit_n_smallest[:, 1], label = 'n smallest')
plt.legend()
plt.axis('equal')
plt.show()


# Get a random airfoil and plot it
idx = np.random.randint(data['P'].shape[0])
P = P_all[idx, :]
L_by_D = L_by_D_all[idx]
print('L_by_D selected at random: ', L_by_D)

# Sample points on the spline
X_fit = get_catmullrom_points(torch.tensor(P.reshape(-1, 2)), num_sample_pts = 201).numpy()

# Plot the airfoil
plt.plot(X_fit[:, 0], X_fit[:, 1])
plt.axis('equal')
plt.show()