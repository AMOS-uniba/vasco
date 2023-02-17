import pytest


def pytest_generate_tests(metafunc):
    if hasattr(metafunc.cls, 'params'):
        funcarglist = metafunc.cls.params.get(metafunc.function.__name__, None)

        if funcarglist:
            argnames = sorted(funcarglist[0])
            metafunc.parametrize(
                argnames, [[funcargs[name] for name in argnames] for funcargs in funcarglist]
            )


class TestProjection:
    @staticmethod
    def compare_inverted(projection, x, y, atol=1e-12):
        assert projection.invert(*projection(x, y)) == pytest.approx((x, y), abs=atol)
