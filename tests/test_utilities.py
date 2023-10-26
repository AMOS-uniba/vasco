import numpy as np

from utilities import disk_to_altaz, altaz_to_disk, spherical_distance
from base import pytest_generate_tests



class TestAngularFunction:
    params = dict(
        test_inverse=[
            dict(x=x, y=y)
            for x in np.linspace(-1, 1, 5)
            for y in np.linspace(-1, 1, 7)
            if x**2 + y**2 < 1
        ],
    )

    def test_inverse(self, x, y):
        point = np.array([[x, y]])
        assert np.allclose(altaz_to_disk(disk_to_altaz(point)), point)


class TestSpherical:
    def test_distance(self):
        assert np.allclose(spherical_distance(np.array([0, 0]), np.array([0, np.pi / 2])), np.pi / 2)

    def test_distance_2(self):
        assert np.allclose(spherical_distance(np.array([0, 1]), np.array([0, 1.1])), 0.1)

    def test_distance_3(self):
        assert np.allclose(
            spherical_distance(
                np.array([np.radians(45), np.radians(45)]),
                np.array([np.radians(37), np.radians(-115)])
            ),
            1.67610715)

    def test_distance_4(self):
        assert np.allclose(
            spherical_distance(
                np.array([np.radians(22), np.radians(98)]),
                np.array([np.radians(-36), np.radians(14)])
            ),
            1.71305632)

    def test_distance_5(self):
        assert np.allclose(
            spherical_distance(
                np.array([np.radians(22), np.radians(98)]),
                np.array([np.radians(-22), np.radians(-82)])
            ),
            np.pi)
