import numpy as np
import matplotlib as mpl

from astropy.coordinates import AltAz
import astropy.units as u

from typing import Tuple


HalfPi = np.pi / 2


def polar_to_cart(z: np.ndarray, a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return z * np.sin(a), -z * np.cos(a)


def by_azimuth(uv):
    uv = np.nan_to_num(uv, 0)
    r = np.sqrt(np.sum(np.square(uv), axis=-1))
    f = (np.arctan2(uv[..., 1], uv[..., 0]) + 2 * np.pi) % (2 * np.pi)
    r = r / np.max(r)
    hsv = np.stack((f / (2 * np.pi), r, np.ones_like(r)), axis=1)
    return mpl.colors.hsv_to_rgb(hsv)


def masked_grid(res):
    s = np.linspace(-1, 1, res)
    x, y = np.meshgrid(s, s)
    xx = np.ma.masked_array(x, x**2 + y**2 > 1)
    yy = np.ma.masked_array(y, x**2 + y**2 > 1)
    return xx, yy


def spherical(x: AltAz, y: AltAz) -> u.Quantity:
    return 2 * np.sin(np.sqrt(np.sin(0.5 * (y.alt - x.alt))**2 + np.cos(x.alt) * np.cos(y.alt) * np.sin(0.5 * (y.az - x.az))**2) * u.rad)


def spherical_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute spherical distance between x and y
    x: np.ndarray(X, 2)
    y: np.ndarray(Y, 2)

    Returns
    np.ndarray(X, Y)
    """
    return 2 * np.sin(
        np.sqrt(
            np.sin(0.5 * (y[..., 0] - x[..., 0]))**2 +
            np.cos(x[..., 0]) * np.cos(y[..., 0]) * np.sin(0.5 * (y[..., 1] - x[..., 1]))**2
        )
    )

def spherical_difference(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute spherical distance between x and y
    x: np.ndarray(X, 2)
    y: np.ndarray(Y, 2)

    Returns
    np.ndarray(X, Y), np.ndarray(X, Y)
    """
    dz = y[..., 0] - x[..., 0]
    da = y[..., 1] - x[..., 1]
    return dz, dz * np.cos(x[..., 0])


def altaz_to_disk(altaz: AltAz) -> np.ndarray:
    return np.stack(
        (
            np.sin(altaz.az.radian) * (HalfPi - altaz.alt.radian) / HalfPi,
            -np.cos(altaz.az.radian) * (HalfPi - altaz.alt.radian) / HalfPi,
        ), axis=1,
    )

def disk_to_altaz(xy: np.ndarray) -> AltAz:
    return AltAz(
        np.sqrt(xy[:, 0]**2 + xy[:, 1]**2) / HalfPi,
        np.arctan2(xy[:, 1], xy[:, 0]),
    )

def proj_to_disk(obs: np.ndarray) -> np.ndarray:
    z, a = obs.T
    x = z * np.sin(a) / np.pi * 2
    y = -z * np.cos(a) / np.pi * 2
    return np.stack((x, y), axis=1)
