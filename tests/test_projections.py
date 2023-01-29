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


class TestBorovickaProjection(TestProjection):
    params = dict(
        test_identity_invert=[
            dict(x=x, y=y)
            for x in np.linspace(-1, 1, 11)
            for y in np.linspace(-1, 1, 11)
            if x**2 + y**2 <= 1
        ],
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
        assert self.compare_inverted(boro_identity, x, y)
