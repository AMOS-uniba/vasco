"""
What loading a sighting has to leave behind.

load_sighting() blocks the location and time signals while it fills those widgets from the file,
which is right -- it does not want a redraw per keystroke -- but it means the handlers that keep
self.location and self.time in step with the widgets never run. Everything downstream reads those
two attributes, not the widgets, so the display was drawn for wherever and whenever vasco started
up while showing the sighting's own coordinates in the boxes. Nudging either box by hand fired the
handler and put it right, which is what made it look intermittent.
"""
import types

import dotmap
import pytest
import yaml
from PyQt6.QtWidgets import QApplication

SIGHTING = 'data/M20120922_225744_AGO__00007.yaml'


@pytest.fixture(scope='session')
def qt_app():
    yield QApplication.instance() or QApplication([])


@pytest.fixture
def window(qt_app):
    from mainwindow import MainWindow

    return MainWindow(types.SimpleNamespace(debug=False, catalogue=None,
                                            sighting=None, projection=None))


@pytest.fixture
def sighting():
    return dotmap.DotMap(yaml.safe_load(open(SIGHTING)), _dynamic=False)


def load(window):
    """ What load_sighting() does either side of the file dialog. """
    window._block_parameter_signals(True)
    window._block_location_time_signals(True)
    window._block_pixel_scales_signals(True)

    window._load_sighting(SIGHTING)

    window._block_parameter_signals(False)
    window._block_location_time_signals(False)
    window._block_pixel_scales_signals(False)
    window.on_location_time_changed()


class TestLoadingASighting:
    def test_the_widgets_show_the_sighting(self, window, sighting):
        load(window)

        assert window.dsb_lat.value() == pytest.approx(sighting.Latitude, abs=1e-6)
        assert window.dt_time.dateTime().toString('yyyy-MM-dd HH:mm:ss') == '2012-09-22 22:57:44'

    def test_and_so_does_the_state_behind_them(self, window, sighting):
        """ The regression: these two used to keep whatever they had at startup. """
        load(window)

        assert window.location.geodetic.lat.value == pytest.approx(sighting.Latitude, abs=1e-6)
        assert window.location.geodetic.lon.value == pytest.approx(sighting.Longitude, abs=1e-6)
        assert window.time.iso.startswith('2012-09-22 22:57:44')

    def test_and_so_does_the_matcher(self, window, sighting):
        """ Which is what every plot and every pairing is computed against. """
        load(window)

        assert window.matcher.time.iso.startswith('2012-09-22 22:57:44')
        assert window.matcher.location.geodetic.lat.value == pytest.approx(sighting.Latitude,
                                                                          abs=1e-6)

    def test_it_moved_from_wherever_it_started(self, window, sighting):
        """ Guards against the assertions above passing because the default happened to match. """
        before = (window.time.iso, window.location.geodetic.lat.value)

        load(window)

        assert (window.time.iso, window.location.geodetic.lat.value) != before

    def test_the_handler_can_be_called_after_any_blocked_write(self, window):
        """
        The general shape, not just this one caller: write the widgets with signals off, notify,
        and the derived state has to catch up.
        """
        window._block_location_time_signals(True)
        window.set_location(11.5, 22.5, 333.0)
        window._block_location_time_signals(False)

        window.on_location_time_changed()

        assert window.location.geodetic.lat.value == pytest.approx(11.5)
        assert window.location.geodetic.lon.value == pytest.approx(22.5)
