import numpy as np

from numpy.typing import NDArray


def spherical(a: NDArray, b: NDArray) -> NDArray:
    """
    Compute spherical distance between a and b, each are vectors of points in two dimensions
    a: np.ndarray(M, D)
    b: np.ndarray(N, D)

    Returns
    -------
    np.ndarray(A, B)
    """
    return 2 * np.arcsin(
        np.sqrt(
            np.sin(0.5 * (b[..., 0] - a[..., 0]))**2 +
            np.cos(a[..., 0]) * np.cos(b[..., 0]) * np.sin(0.5 * (b[..., 1] - a[..., 1]))**2
        )
    )


def euclidean(a: NDArray, b: NDArray) -> NDArray:
    return np.sqrt(np.sum((a - b)**2, axis=2))
