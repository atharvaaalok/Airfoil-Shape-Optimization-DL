import numpy as np
import matplotlib.pyplot as plt
import torch

from airfoil_utils.airfoil_parameterization import get_catmullrom_points

# Load control points for the airfoil
idx = np.random.randint(900)
airfoil_set = 'train'
filename = 'original_HV_MV_LV'
data = np.load(f'generated_airfoils/{airfoil_set}/{filename}.npz')
P = data['P'][idx, :]
L_by_D = data['L_by_D'][idx]
print(L_by_D)

# Sample points on the spline
X_fit = get_catmullrom_points(torch.tensor(P.reshape(-1, 2)), num_sample_pts = 201).numpy()


# Plot the airfoil
plt.plot(X_fit[:, 0], X_fit[:, 1])
plt.axis('equal')
plt.show()