import numpy as np
from xfoil import XFoil
from xfoil.model import Airfoil


# Get an airfoil's coordinates
X_all = np.load('data/generated_airfoils/airfoils_original.npy')

X = X_all[0, :].reshape(-1, 2)
x = X[:, 0]
y = X[:, 1]

# Set the coordinates of the airfoil
airfoil = Airfoil(x, y)


# Instantiate the XFoil class
xf = XFoil()
xf.print = False

# Set the airfoil for xfoil
xf.airfoil = airfoil
xf.Re = 100000

# Perform analysis at 0 angle of attack
cl, cd, cm, cp = xf.a(0)
print(f'{cl = :.6f}')
print(f'{cd = :.6f}')