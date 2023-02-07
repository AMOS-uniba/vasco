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


class TestZenithShifter(TestProjection):
    grid = [
        dict(u=u, b=b)
        for u in np.linspace(0.01, np.pi / 2, 11)
        for b in np.linspace(0.1, 2 * np.pi + 0.1, 11, endpoint=False)
    ]
    params = dict(
        test_zenith_aligned=grid,
        test_general=grid,
    )

    def test_zenith_aligned(self, zenith_aligned, u, b):
        assert zenith_aligned.invert(*zenith_aligned(u, b)) == pytest.approx((u, b), abs=1e-9)

    def test_general(self, general, u, b):
        assert general.invert(*general(u, b)) == pytest.approx((u, b), abs=1e-9)

