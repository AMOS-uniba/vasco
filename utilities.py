import math
import numpy as np
import matplotlib as mpl

from astropy.coordinates import AltAz
import astropy.units as u

from typing import Tuple, Union


QuarterTau = math.tau / 4


def polar_to_cart(z: np.ndarray, a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return z * np.sin(a), -z * np.cos(a)


def by_azimuth(uv):
    uv = np.nan_to_num(uv, nan=0)
    r = np.sqrt(np.sum(np.square(uv), axis=-1))
    f = (np.arctan2(uv[..., 1], uv[..., 0]) + math.tau) % math.tau
    r = r / np.max(r)
    hsv = np.stack((f / math.tau, r, np.ones_like(r)), axis=1)
    return mpl.colors.hsv_to_rgb(hsv)


def unit_grid(res, *, masked: bool):
    s = np.linspace(-1, 1, res)
    x, y = np.meshgrid(s, s)
    if masked:
        xx = np.ma.masked_array(x, x**2 + y**2 > 1)
        yy = np.ma.masked_array(y, x**2 + y**2 > 1)
        return xx, yy
    else:
        return x, y


def spherical(x: AltAz, y: AltAz) -> u.Quantity:
    return 2 * np.sin(
        np.sqrt(
            np.sin(0.5 * (y.alt - x.alt))**2 + np.cos(x.alt) * np.cos(y.alt) * np.sin(0.5 * (y.az - x.az))**2
        ) * u.rad
    )


def spherical_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute spherical distance between a and b, each are vectors of points in D dimensions
    a: np.ndarray(A, D)
    b: np.ndarray(B, D)

    Returns
    -------
    np.ndarray(A, B)
    """
    return 2 * np.sin(
        np.sqrt(
            np.sin(0.5 * (b[..., 0] - a[..., 0]))**2 +
            np.cos(a[..., 0]) * np.cos(b[..., 0]) * np.sin(0.5 * (b[..., 1] - a[..., 1]))**2
        )
    )


def spherical_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute spherical distance between x and y
    x: np.ndarray(X, ..., 2)
    y: np.ndarray(Y, ..., 2)

    Returns
    -------
    np.ndarray(X, Y), np.ndarray(X, Y)
    """
    dz = b[..., 0] - a[..., 0]
    da = b[..., 1] - a[..., 1]
    return np.stack((dz, da * np.cos(a[..., 0])), axis=1)


def altaz_to_disk(altaz: Union[None, AltAz]) -> np.ndarray:
    if altaz is None:
        return np.empty(shape=(0, 2))
    else:
        return np.stack(
            (
                np.sin(altaz.az.radian) * (QuarterTau - altaz.alt.radian) / QuarterTau,
                -np.cos(altaz.az.radian) * (QuarterTau - altaz.alt.radian) / QuarterTau,
            ), axis=1,
        )


def disk_to_altaz(xy: np.ndarray) -> AltAz:
    return AltAz(
        (QuarterTau + np.arctan2(xy[:, 1], xy[:, 0])) * u.rad,          # Add pi/2 since our 0° is at the bottom
        np.sqrt(xy[:, 0] ** 2 + xy[:, 1] ** 2) * QuarterTau * u.rad,
    )


def proj_to_disk(obs: np.ndarray) -> np.ndarray:
    if obs is None:
        return np.empty(shape=(0, 2))
    else:
        z, a = obs.T
        x = z * np.sin(a) / math.tau * 4
        y = -z * np.cos(a) / math.tau * 4
        return np.stack((x, y), axis=1)
