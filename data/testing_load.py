import numpy as np
import matplotlib.pyplot as plt


# Load the coordinates and L by D ratios of original airfoils
airfoil_set = 'test'
data = np.load(f'generated_airfoils/{airfoil_set}/original.npz')
X = data['X']
L_by_D = data['L_by_D']



# Plot an airfoil
airfoil_idx = 23
X_i = X[airfoil_idx, :].reshape(-1, 2)
x, y = X_i[:, 0], X_i[:, 1]

plt.plot(x, y)
plt.axis('equal')
plt.show()