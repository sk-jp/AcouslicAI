import numpy as np


def ellipse_circumference(a, b):
    """
    Approximate the circumference of an ellipse using Ramanujan's first formula.

    Parameters:
    a (float): Semi-major axis of the ellipse.
    b (float): Semi-minor axis of the ellipse.

    Returns:
    float: The approximate circumference of the ellipse.
    """
    return np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b)))
