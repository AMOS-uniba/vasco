import pytest
import math
import numpy as np

from base import pytest_generate_tests, TestProjection
from projections.zenith import ZenithShifter


@pytest.fixture
def zenith_aligned():
    return ZenithShifter(0, 0)


@pytest.fixture
def general():
    return ZenithShifter(math.radians(1.5), math.radians(213.4))


@pytest.fixture
def general_2():
    return ZenithShifter(math.radians(17.3), math.radians(107.4))


class TestZenithShifter(TestProjection):
    grid = [
        dict(r=r, t=t)
        for r in np.linspace(0.01, math.tau / 4, 11)
        for t in np.linspace(0.1, math.tau + 0.1, 11, endpoint=False)
    ]
    params = dict(
        test_zenith_aligned=grid,
        test_general=grid,
        test_general_2=grid,
    )

    def test_zenith_aligned(self, zenith_aligned, r, t):
        assert zenith_aligned.invert(*zenith_aligned(r, t)) == pytest.approx((r, t), abs=1e-9)

    def test_general(self, general, r, t):
        assert general.invert(*general(r, t)) == pytest.approx((r, t), abs=1e-9)

    def test_general_2(self, general_2, r, t):
        assert general_2.invert(*general_2(r, t)) == pytest.approx((r, t), abs=1e-9)
