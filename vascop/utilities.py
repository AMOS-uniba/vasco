import numpy as np
import matplotlib as mpl
from typing import Tuple


def polar_to_cart(z: np.ndarray, a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return z * np.sin(a), -z * np.cos(a)


def by_azimuth(uv):
    uv = np.nan_to_num(uv, 0)
    r = np.sqrt(np.sum(np.square(uv), axis=2))
    f = (np.arctan2(uv[:, :, 1], uv[:, :, 0]) + 2 * np.pi) % (2 * np.pi)
    r = r / np.max(r)
    hsv = np.dstack((f / (2 * np.pi), r, np.ones_like(r)))
    return mpl.colors.hsv_to_rgb(hsv)

