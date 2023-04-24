import pytest

from projections.transformers import LinearTransformer, ExponentialTransformer, BiexponentialTransformer


@pytest.fixture
def lin():
    return LinearTransformer(1.25)


@pytest.fixture
def vsd():
    return ExponentialTransformer(0.95, 0.28, 0.05)


@pytest.fixture
def vsdpq():
    return BiexponentialTransformer(1.2, 0.3, -0.05, 0.2, 0.01)


class TestLinearTransformer:
    def test_inverse(self, lin):
        assert lin.invert(lin(4.05)) == pytest.approx(4.05, rel=1e-12)


class TestBiexpTransformer():
    def test_inverse_1(self, vsdpq):
        assert vsdpq.invert(vsdpq(0.775)) == pytest.approx(0.775, rel=1e-12)

    def test_inverse_2(self, vsdpq):
        assert vsdpq.invert(vsdpq(0.27)) == pytest.approx(0.27, rel=1e-12)

    def test_inverse_3(self, vsdpq):
        assert vsdpq.invert(vsdpq(0.999)) == pytest.approx(0.999, rel=1e-12)

    def test_manual(self, vsdpq):
        assert vsdpq(0.895) == pytest.approx(1.0624794369015114, rel=1e-12)

    def test_manual_inverse(self, vsdpq):
        assert vsdpq.invert(1.0624794369) == pytest.approx(0.895, rel=1e-9)


class TestExponentialTransformer():
    def test_inverse_1(self, vsd):
        assert vsd.invert(vsd(0.775)) == pytest.approx(0.775, rel=1e-12)

    def test_inverse_2(self, vsd):
        assert vsd.invert(vsd(0.27)) == pytest.approx(0.27, rel=1e-12)

    def test_inverse_3(self, vsd):
        assert vsd.invert(vsd(0.999)) == pytest.approx(0.999, rel=1e-12)
