import numpy as np
import numpy.typing as npt


def airfoil_noise_addition(X: npt.NDArray, noise_level: float):
    """This function takes an airfoil and creates a new airfoil by adding noise to it.

    The noise is added to each coordinate and the amount of noise added is relative to the value of
    the coordinate. That is, x_new = x + x * delta_x.
    
    Args:
        X: Airfoil coordinates in the order x1, y1, x2, y2, ..., xn, yn.
        noise_level: Amount of noise. This decides how different new airfoils will be from original.
    """

    # Generate the delta to be added to the airfoil coordinates
    delta_X = np.random.rand(*X.shape) * noise_level

    # Calculate new coordinates using delta as a relative change
    X_new = X + X * delta_X

    # Ensure that the first and the last point coincide
    x_avg = (X[0] + X[-2]) / 2
    y_avg = (X[1] + X[-1]) / 2
    X[0] = X[-2] = x_avg
    X[1] = X[-1] = y_avg

    return X_new