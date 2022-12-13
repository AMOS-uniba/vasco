import numpy as np
import matplotlib as mpl

from astropy.coordinates import AltAz
import astropy.units as u

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


def spherical(x: AltAz, y: AltAz) -> u.Quantity:
    return 2 * np.sin(np.sqrt(np.sin(0.5 * (y.alt - x.alt))**2 + np.cos(x.alt) * np.cos(y.alt) * np.sin(0.5 * (y.az - x.az))**2) * u.rad)


def distance(x, y):
    return 2 * np.sin(
        np.sqrt(
            np.sin(0.5 * (y[0, :, :] - x[0, :, :]))**2 +
            np.cos(x[0, :, :]) * np.cos(y[0, :, :]) * np.sin(0.5 * (y[1, :, :] - x[1, :, :]))**2.0
        )
    )
