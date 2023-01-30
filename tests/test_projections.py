import pytest
import numpy as np

from base import pytest_generate_tests, TestProjection
from projections import Projection, BorovickaProjection


class TestBase():
    def test_is_abstract(self):
        with pytest.raises(TypeError):
            _ = Projection()

    def test_borovicka_proj(self):
        assert isinstance(BorovickaProjection(), Projection)


@pytest.fixture
def boro_identity():
    return BorovickaProjection(0, 0, 0, 0, 0, np.pi / 2, 0, 0, 0, 0, 0, 0)

@pytest.fixture
def boro_rotated():
    return BorovickaProjection(0, 0, np.radians(1.2), 0, 0, np.pi / 2, 0, 0, 0, 0, 0, 0)

@pytest.fixture
def boro_general():
    return BorovickaProjection(
        796.164, 605.979, np.radians(256.914847),
        0.003833, np.radians(187.852341),
        0.00194479, -5.3e-05, -0.000192, -0.006691, -4.2e-05,
        0.55556, np.radians(127.910437),
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
        test_general_invert=big_grid,
    )

    def test_identity_zero(self, boro_identity):
        assert boro_identity(0, 0) == (0, 0)

    def test_identity_east(self, boro_identity):
        assert boro_identity(0, -1) == pytest.approx((np.pi / 2, 3 * np.pi / 2), rel=1e-14)

    def test_identity_south(self, boro_identity):
        assert boro_identity(-1, 0) == pytest.approx((np.pi / 2, np.pi), rel=1e-14)

    def test_identity_north(self, boro_identity):
        assert boro_identity(1, 0) == pytest.approx((np.pi / 2, 0), rel=1e-14)

    def test_identity_somewhere(self, boro_identity):
        assert boro_identity(0.5, -0.5) == pytest.approx((np.sqrt(2) * np.pi / 4, np.pi * 1.75), rel=1e-14)

    def test_identity_invert(self, boro_identity, x, y):
        self.compare_inverted(boro_identity, x, y)

    def test_rotated_invert(self, boro_rotated, x, y):
        self.compare_inverted(boro_rotated, x, y)

    def test_general_invert(self, boro_general, x, y):
        self.compare_inverted(boro_general, x, y, abs=1e-9)
