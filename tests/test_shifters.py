import pytest
import numpy as np

from projections.shifters import OpticalAxisShifter, TiltShifter


@pytest.fixture
def oas():
    return OpticalAxisShifter(x0=0.213, y0=0.123, a0=4.356, E=0.0314)


@pytest.fixture
def ts():
    return TiltShifter(x0=1.2, y0=0.3, a0=-0.05, A=0.2, F=np.pi / 2, E=0.01)


class TestOpticalAxisShifter():
    def test_inverse_1(self, oas):
        assert oas.invert(*oas(0.775, 1.234)) == pytest.approx((0.775, 1.234), rel=1e-12)

    def test_inverse_2(self, oas):
        assert oas.invert(*oas(0.168, 2.124)) == pytest.approx((0.168, 2.124), rel=1e-12)

    def test_inverse_3(self, oas):
        assert oas.invert(*oas(0, 0)) == pytest.approx((0, 0), rel=1e-12)


class TestTiltShifter():
    def test_inverse_1(self, ts):
        assert ts.invert(*ts(0.775, 1.234)) == pytest.approx((0.775, 1.234), rel=1e-12)

    def test_inverse_2(self, ts):
        assert ts.invert(*ts(0.168, 2.124)) == pytest.approx((0.168, 2.124), rel=1e-12)

    def test_inverse_3(self, ts):
        assert ts.invert(*ts(0, 0)) == pytest.approx((0, 0), rel=1e-12)
