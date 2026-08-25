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


def load_sighting(window, path):
    """
    Load a sighting into a window the way the window itself does.

    _load_sighting() only fills the widgets and swaps the sensor data; the pairing, the altaz cache
    and the plots are brought up to date by the handlers, which load_sighting() unblocks and then
    fires. Calling _load_sighting() alone leaves the matcher with an empty pairing and the next
    thing to ask it for residuals gets an IndexError about a mask of the wrong length.
    """
    window._block_parameter_signals(True)
    window._block_location_time_signals(True)
    window._block_pixel_scales_signals(True)

    window._load_sighting(path)

    window._block_parameter_signals(False)
    window._block_location_time_signals(False)
    window._block_pixel_scales_signals(False)
    window.on_location_time_changed()
