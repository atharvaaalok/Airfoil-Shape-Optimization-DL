import numpy as np
import numpy.typing as npt
from xfoil import XFoil
from xfoil.model import Airfoil


# Create global instance of XFoil class - each instance creates a temporary copy of the fortran code
# therefore creating a new instance in each run is extremely expensive.
xf = XFoil()

def compute_L_by_D(X: npt.NDArray) -> float:
    """Given an airfoil's coordinates compute the L by D ratio.

    Args:
        X: Airfoil coordinates.
    """

    # Reshape airfoil coordinate array and get the x and y coordinates
    X = X.reshape(-1, 2)
    x, y = X[:, 0], X[:, 1]

    # Create an airfoil object using these coordinates
    airfoil = Airfoil(x, y)


    # Instantiate the XFoil class
    # xf = XFoil()
    xf.print = False

    # Set the airfoil and flow properties
    xf.airfoil = airfoil
    xf.Re = 1e6
    xf.max_iter = 100
    # xf.repanel(n_nodes = 250, cte_ratio = 1)

    # Calculate aerodynamic coefficients at 0 angle of attack
    cl, cd, cm, cp = xf.a(0)

    # Calculate L by D ratio
    L_by_D = cl / cd

    return L_by_D