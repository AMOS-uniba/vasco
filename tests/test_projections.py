import pytest
import math
import numpy as np

from base import pytest_generate_tests, TestProjection
from projections import Projection, BorovickaProjection


class TestBase:
    def test_is_abstract(self):
        with pytest.raises(TypeError):
            _ = Projection()

    def test_borovicka_proj(self):
        assert isinstance(BorovickaProjection(), Projection)


@pytest.fixture
def boro_identity():
    return BorovickaProjection(0, 0, 0, 0, 0, math.tau / 4, 0, 0, 0, 0, 0, 0)


@pytest.fixture
def boro_rotated():
    return BorovickaProjection(0, 0, math.radians(1.2), 0, 0, math.tau / 4, 0, 0, 0, 0, 0, 0)


@pytest.fixture
def boro_zenith():
    """ Borovička projection with everything but zenith offset """
    return BorovickaProjection(
        796.164, 605.979, math.radians(256.914847),
        0.003833, math.radians(187.852341),
        0.00194479, -5.3e-05, -0.000192, -0.006691, -4.2e-05,
        0, 0,
    )


@pytest.fixture
def boro_general():
    """ Fully generalized Borovička projection, based on sighting 20220531_055655-DRGR """
    return BorovickaProjection(
        796.164, 605.979, math.radians(256.914847),
        0.003833, math.radians(187.852341),
        0.00194479, -5.3e-05, -0.000192, -0.006691, -4.2e-05,
        math.radians(0.555556), math.radians(127.910437),
    )


@pytest.fixture
def boro_random():
    """ Fully generalized Borovička projection, with some made-up but correct values """
    return BorovickaProjection(
        775.4, 581.625, math.radians(53.5123),
        0.00376, math.radians(65.25),
        0.00185364, 0.0000253, -0.000592, -0.006691, -3.3e-06,
        math.radians(7.5263), math.radians(203.910437),
    )


@pytest.fixture
def boro_karel():
    """ Fully generalized Borovička projection, with some made-up but correct values """
    return BorovickaProjection(
        824.4, 617.625, math.radians(357.4523),
        0.00423, math.radians(23.5),
        0.00198453, 0.0000753, 0.000572, 0.0005691, 2.3e-06,
        math.radians(2.5142), math.radians(337.910437),
    )


class TestBorovickaProjection(TestProjection):
    grid = [
        dict(x=x, y=y)
        for x in np.linspace(-1, 1, 12)
        for y in np.linspace(-1, 1, 12)
        if x**2 + y**2 <= 1
    ]

    big_grid = [
        dict(x=x, y=y)
        for x in np.linspace(0, 1600, 11)
        for y in np.linspace(0, 1200, 13)
    ]

    params = dict(
        test_identity_invert=grid,
        test_rotated_invert=grid,
        test_zenith_invert=big_grid,
        test_general_invert=big_grid,
        test_random_invert=big_grid,
        test_karel_invert=big_grid,
    )

    def test_identity_zero(self, boro_identity):
        assert boro_identity(0, 0) == (0, 0)

    def test_identity_east(self, boro_identity):
        assert boro_identity(0, -1) == pytest.approx((0.25 * math.tau, 0.75 * math.tau), abs=1e-14)

    def test_identity_south(self, boro_identity):
        assert boro_identity(-1, 0) == pytest.approx((0.25 * math.tau, 0.5 * math.tau), abs=1e-14)

    def test_identity_north(self, boro_identity):
        assert boro_identity(1, 0) == pytest.approx((0.25 * math.tau, 0), abs=1e-14)

    def test_identity_somewhere(self, boro_identity):
        assert boro_identity(0.5, -0.5) == pytest.approx((np.sqrt(2) / 8 * math.tau, math.tau * 0.875), abs=1e-14)

    def test_identity_invert(self, boro_identity, x, y):
        assert boro_identity.invert(*boro_identity(x, y)) == pytest.approx((x, y), abs=1e-9)

    def test_rotated_invert(self, boro_rotated, x, y):
        assert boro_rotated.invert(*boro_rotated(x, y)) == pytest.approx((x, y), abs=1e-9)

    def test_zenith_invert(self, boro_zenith, x, y):
        assert boro_zenith.invert(*boro_zenith(x, y)) == pytest.approx((x, y), abs=1e-9)

    def test_general_invert(self, boro_general, x, y):
        assert boro_general.invert(*boro_general(x, y)) == pytest.approx((x, y), abs=1e-9)

    def test_random_invert(self, boro_random, x, y):
        assert boro_random.invert(*boro_random(x, y)) == pytest.approx((x, y), abs=1e-9)

    def test_karel_invert(self, boro_karel, x, y):
        assert boro_karel.invert(*boro_karel(x, y)) == pytest.approx((x, y), abs=1e-9)
