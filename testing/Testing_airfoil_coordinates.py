import numpy as np
import matplotlib.pyplot as plt


# Load an airfoil
X = np.loadtxt('data/airfoil_database/2032c.dat')
print(X.shape)


# Select number of points to display from beginning to identify ordering clockwise or anti-clockwise
val = 201
x, y = X[:val, 0], X[:val, 1]


# Plot the airfoil
plt.plot(x, y)
plt.axis('equal')
plt.show()