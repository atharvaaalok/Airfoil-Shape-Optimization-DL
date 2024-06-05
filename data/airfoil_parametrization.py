import torch
from torch import nn


def fit_catmullrom(X, num_control_pts: int):
    X = X.reshape(-1, 2)
    X = torch.from_numpy(X)

    # Initialize control points for the spline
    idx = torch.linspace(0, X.shape[0] - 1, num_control_pts).to(torch.int)
    P = X[idx, :]

    # Make a list of the control points and set requires grad to True except first and last
    P_list = [P[i, :] for i in range(num_control_pts)]
    for i in range(1, num_control_pts - 1):
        P_list[i].requires_grad = True
    

    # Set number of sample points to use for curve fitting
    num_sample_pts = 501

    # Setup the optimization problem
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(P_list[1: -1], lr = learning_rate)
    loss_fn = curve_fit_loss

    # Training
    epochs = 100
    for epoch in range(1, epochs + 1):
        # Get the spline sample points
        X_fit = get_catmullrom_points(P_list, num_sample_pts)

        # Calculate the loss
        loss = loss_fn(X_fit, X)

        # Run backward pass
        loss.backward()

        # Optimize
        optimizer.step()
        optimizer.zero_grad()
    

    # Return the control points
    P_list = [P.detach() for P in P_list]
    P_tensor = torch.stack(P_list)

    return P_tensor



def get_catmullrom_points(P_list, num_sample_pts):
    num_control_pts = len(P_list)

    # Sample equally spaced points on the spline
    num_curves = num_control_pts - 1
    t = torch.linspace(0, num_control_pts - 1, num_sample_pts)

    # Add ghost points to make the spline pass through first and last point
    P_tensor = torch.stack(P_list)
    G0 = P_tensor[0] + (P_tensor[0] - P_tensor[1])
    G1 = P_tensor[-1] + (P_tensor[-1] - P_tensor[-2])
    P_extended = torch.vstack([G0, P_tensor, G1])

    # Get curve index for every t value and set the t value to between 0 and 1
    curve_indices = torch.clamp(t.floor().long(), 0, num_curves - 1)
    t = t - curve_indices
    # Get the bernstein coefficients
    t_val = torch.stack([
        0.5 * (-t + 2 * (t**2) - (t**3)),
        0.5 * (2 - 5 * (t**2) + 3 * (t**3)),
        0.5 * (t + 4 * (t**2) - 3 * (t**3)),
        0.5 * (-(t**2) + (t**3))
    ]).T

    # Get the control point coordinates
    p0 = P_extended[curve_indices]
    p1 = P_extended[curve_indices + 1]
    p2 = P_extended[curve_indices + 2]
    p3 = P_extended[curve_indices + 3]

    px = torch.stack([p0[:, 0], p1[:, 0], p2[:, 0], p3[:, 0]]).T
    py = torch.stack([p0[:, 1], p1[:, 1], p2[:, 1], p3[:, 1]]).T

    # Get the x and y coordinates of the sample points
    sample_x = torch.sum(t_val * px, axis = 1)
    sample_y = torch.sum(t_val * py, axis = 1)

    # Get the sample points
    X_fit = torch.stack([sample_x, sample_y]).T

    return X_fit



def curve_fit_loss(X_fit, X):
    with torch.no_grad():
        dists = torch.cdist(X, X_fit, p = 2)
        idx1 = torch.argmin(dists, dim = 1)
        dists[range(dists.size(0)), idx1] = float('inf')
        idx2 = torch.argmin(dists, dim = 1)
    

    # Get the altitude length
    # Get the two closest points from the sample points on spline to every point in the original
    X1 = X_fit[idx1]
    X2 = X_fit[idx2]

    # Get the altitude length
    h = get_altitude(X1, X2, X)
    
    loss = torch.mean(h * h)
    return loss



def get_altitude(X1, X2, X):
    a, b = X1[:, 0], X1[:, 1]
    c, d = X2[:, 0], X2[:, 1]
    e, f = X[:, 0], X[:, 1]

    # Get area of triangle formed by the three points
    A = torch.abs(a * d - b * c + c * f - d * e + b * e - a * f)
    # Get base width
    eps = 1e-10
    w = torch.sqrt((c - a) ** 2 + (d - b) ** 2 + eps)

    return A / w