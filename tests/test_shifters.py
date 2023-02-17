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


@pytest.fixture
def ts3():
    return TiltShifter(x0=800, y0=600, a0=math.radians(95), A=0.134, F=math.radians(70), E=math.radians(270))


@pytest.fixture
def ts4():
    return TiltShifter(x0=775.4, y0=581.625, a0=math.radians(53.5123),
                       A=0.00376, F=math.radians(65.25), E=math.radians(203.910437))


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
        dict(proj=proj, x=x, y=y)
        for proj in ['ts1', 'ts2', 'ts3', 'ts4']
        for x in np.linspace(-1, 1, 15)
        for y in np.linspace(-1, 1, 15)
        if x**2 + y**2 < 1
    ]
    params = dict(
        test_inverse=grid,
    )

    def test_inverse(self, proj, x, y, request):
        proj = request.getfixturevalue(proj)
        assert proj.invert(*proj(x, y)) == pytest.approx((x, y), abs=1e-9)
