"""
The correction grid, which had been switched off.

`_plot_correction_grid` computed a lattice and threw it away: the line that would have drawn it was
commented out, so `Matcher.position_grid` and `magnitude_grid` were never called by anything and
nobody noticed there were two `unit_grid` functions. Switched on, it is worth a test, because
neither plot takes what it is handed at face value -- the quiver ravels a (R, R, 2) field into u and
v and needs x and y to match, and the imshow asserts its input is (R, R, 1) and square.

Driven through the real window on a real sighting, because the shapes only line up at the point
where the field, the lattice and the artist meet, and nothing smaller than that exercises it.
"""
import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

from vasco.plots import grid

from .base import load_sighting

SIGHTING = 'data/M20120922_225744_AGO__00007.yaml'


@pytest.fixture(scope='session')
def qt_app():
    yield QApplication.instance() or QApplication([])


@pytest.fixture
def window(qt_app):
    """ A window with a sighting loaded and the smoothers built, as a person would have it. """
    import types

    from vasco.mainwindow import MainWindow

    main = MainWindow(types.SimpleNamespace(catalogue=None, sighting=None, projection=None,
                                            debug=False))
    load_sighting(main, SIGHTING)
    main.matcher.update_position_smoother(bandwidth=0.1)
    main.matcher.update_magnitude_smoother(bandwidth=0.1)
    return main


class TestItDrawsSomething:
    def test_the_position_quiver_appears(self, window):
        window.cb_show_grid.setChecked(True)

        window.plot_position_correction_grid()

        assert window.position_correction_plot.quiver_grid is not None
        assert window.position_correction_plot.valid_grid()

    def test_the_magnitude_image_appears(self, window):
        window.cb_show_grid.setChecked(True)

        window.plot_magnitude_correction_grid()

        assert window.magnitude_correction_plot.magnitude_grid is not None
        assert window.magnitude_correction_plot.valid_grid()

    def test_unticking_clears_it(self, window):
        window.cb_show_grid.setChecked(True)
        window.plot_position_correction_grid()

        window.cb_show_grid.setChecked(False)
        window.plot_position_correction_grid()

        # clear_grid draws an empty quiver rather than removing it, so what is asserted is that it
        # went through the clearing path and has no arrows left
        assert window.position_correction_plot.quiver_grid.N == 0

    def test_it_survives_being_drawn_twice(self, window):
        """ Each artist removes its predecessor; the second call is where that shows. """
        window.cb_show_grid.setChecked(True)

        window.plot_position_correction_grid()
        window.plot_magnitude_correction_grid()
        window.plot_position_correction_grid()
        window.plot_magnitude_correction_grid()

        assert window.position_correction_plot.valid_grid()
        assert window.magnitude_correction_plot.valid_grid()

    #: The spinbox's own range: 11 to 201 in steps of two, defaulting to 41. Odd on purpose, so
    #: that a lattice has a point at the centre.
    @pytest.mark.parametrize('resolution', [11, 41, 201])
    def test_at_the_ends_of_the_resolution_the_spinbox_allows(self, window, resolution):
        window.cb_show_grid.setChecked(True)
        window.sb_resolution.setValue(resolution)
        assert window.sb_resolution.value() == resolution, "the spinbox clamps outside 11..201"

        window.plot_position_correction_grid()
        window.plot_magnitude_correction_grid()

        assert window.magnitude_correction_plot.magnitude_grid.get_array().shape == \
            (resolution, resolution)


class TestBothLattices:
    @pytest.mark.parametrize('lattice', sorted(grid.LATTICES))
    def test_the_quiver_takes_either(self, window, lattice):
        """ Which is the point of keeping both: the position plot can switch. """
        window.cb_show_grid.setChecked(True)

        window._plot_correction_grid(window.position_correction_plot,
                                     window.matcher.position_smoother,
                                     masked=True, lattice=lattice)

        assert window.position_correction_plot.quiver_grid.N > 0

    def test_the_triangular_one_is_sparser(self, window):
        """ 89 arrows against 313 at resolution 21, because the shear pushes most of it outside. """
        window.cb_show_grid.setChecked(True)
        window.sb_resolution.setValue(21)
        drawn = {}

        for lattice in ('square', 'triangular'):
            window._plot_correction_grid(window.position_correction_plot,
                                         window.matcher.position_smoother,
                                         masked=True, lattice=lattice)
            xx, _ = grid.lattice(21, masked=True, kind=lattice)
            drawn[lattice] = int(np.ma.count(xx))

        assert drawn['triangular'] < drawn['square']

    def test_the_imshow_cannot_take_the_triangular_one(self, window):
        """
        Not a limitation to work around -- imshow paints a regular grid over an extent, so a sheared
        lattice would be stretched back into a square and drawn in the wrong places. Hence the
        magnitude plot pins itself to `square`, and this is why.
        """
        window.cb_show_grid.setChecked(True)
        window.sb_resolution.setValue(21)

        square, _ = grid.lattice(21, masked=False, kind='square')
        triangular, _ = grid.lattice(21, masked=False, kind='triangular')

        assert square.min() == pytest.approx(-1) and square.max() == pytest.approx(1)
        assert triangular.min() < -1.5, "spans well outside the extent imshow would draw it in"


class TestTheFieldItself:
    def test_the_sampled_position_field_has_two_components(self, window):
        sampled = grid.sample(window.matcher.position_smoother, 21, masked=True)

        assert sampled.shape == (21, 21, 2)

    def test_the_sampled_magnitude_field_has_one(self, window):
        sampled = grid.sample(window.matcher.magnitude_smoother, 21, masked=False, kind='square')

        assert sampled.shape == (21, 21, 1)

    def test_and_it_is_not_all_zero(self, window):
        """ A field of zeros would draw a picture that looked fine and meant nothing. """
        sampled = grid.sample(window.matcher.position_smoother, 21, masked=True)

        assert np.any(np.abs(sampled) > 1e-9)
