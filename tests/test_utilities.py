import numpy as np

from vasco.utilities import disk_to_altaz, altaz_to_disk
from .base import pytest_generate_tests



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

