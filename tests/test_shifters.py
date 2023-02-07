import pytest
import math
import numpy as np

from base import pytest_generate_tests, TestProjection
from projections.shifters import OpticalAxisShifter, TiltShifter


@pytest.fixture
def oas():
    return OpticalAxisShifter(x0=0.213, y0=0.123, a0=4.356, E=0.0314)


@pytest.fixture
def ts1():
    return TiltShifter(x0=1.2, y0=0.5, a0=-0.05, A=0.17, F=math.radians(135), E=0.01)


@pytest.fixture
def ts2():
    return TiltShifter(x0=-0.2, y0=0.345, a0=0.15, A=-0.134, F=math.radians(240), E=0.05)


class TestOpticalAxisShifter(TestProjection):
    params = dict(
        test_inverse=[
            dict(x=x, y=y)
            for x in np.linspace(-1, 1, 5)
            for y in np.linspace(-1, 1, 7)
            if x**2 + y**2 < 1
        ],
    )

    def test_inverse(self, oas, x, y):
        self.compare_inverted(oas, x, y)


class TestTiltShifter(TestProjection):
    grid = [
        dict(x=x, y=y)
        for x in np.linspace(-1, 1, 11)
        for y in np.linspace(-1, 1, 11)
        if x**2 + y**2 < 1
    ]
    params = dict(
        test_inverse_1=grid,
        test_inverse_2=grid,
    )

    def test_inverse_1(self, ts1, x, y):
        self.compare_inverted(ts1, x, y)

    def test_inverse_2(self, ts2, x, y):
        self.compare_inverted(ts2, x, y)
