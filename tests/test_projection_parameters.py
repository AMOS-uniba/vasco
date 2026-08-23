"""
What happens to a fit between the optimiser and the widgets.

The parameter widgets are not a display: get_projection_parameters() reads them straight back as
the model, so whatever they hold *is* the plate. And a QDoubleSpinBox will not hold a value outside
its range, without raising or warning about it -- so writing an unnormalised fit into them replaced
it, silently, which is the bug these tests exist for.

The widgets here are configured by calling MainWindow.setup_parameters on a stand-in rather than by
repeating the ranges, so that a test cannot pass against a configuration the program does not have.
"""
import math
from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pytest
from demeteor.projections import BorovickaProjection
from demeteor.projections.base import TAU
from PyQt6.QtWidgets import QApplication

from mainwindow import MainWindow
from widgets.qparameterwidget import QParameterWidget

#: In the order BorovickaProjection takes its parameters, which is the order the widgets are in
NAMES = ('x0', 'y0', 'a0', 'A', 'F', 'V', 'S', 'D', 'P', 'Q', 'epsilon', 'E')
A0, F, EPSILON, E = 2, 4, 10, 11

#: A real fitted plate, from the AGO calibration file
KY = (0.014864, -0.014848, 4.4094506287729995, -0.002302, 1.4911492386711016, 0.452072,
      -0.000868, 1.6e-05, 1.074161, -0.002488, 0.013736282265263492, 2.613772659569206)


@pytest.fixture(scope='session')
def qt_app():
    yield QApplication.instance() or QApplication([])


@pytest.fixture
def window(qt_app):
    """
    Enough of a MainWindow to hold parameters, and no more.

    setup_parameters and set_projection_parameters are called unbound on this, so the ranges, the
    wrapping and the degree conversions under test are the ones the real window uses.
    """
    stub = SimpleNamespace(on_projection_parameters_changed=lambda: None)
    for name in NAMES:
        setattr(stub, f'pw_{name}', QParameterWidget())
    stub.param_widgets = OrderedDict((name, getattr(stub, f'pw_{name}')) for name in NAMES)
    stub._block_parameter_signals = MainWindow._block_parameter_signals.__get__(stub)

    MainWindow.setup_parameters(stub)
    return stub


def write(window, values):
    return MainWindow.set_projection_parameters(window, values)


def read(window):
    return MainWindow.get_projection_parameters(window)


def replaced(index, value):
    return KY[:index] + (value,) + KY[index + 1:]


def sky_distance(one, other):
    """ The furthest apart two plates put any point of the sensor, in radians. """
    xs, ys = (axis.ravel() for axis in np.meshgrid(np.linspace(-6, 6, 30), np.linspace(-6, 6, 30)))
    z1, a1 = BorovickaProjection(*one)(xs, ys)
    z2, a2 = BorovickaProjection(*other)(xs, ys)
    return float(np.nanmax(np.hypot(z1 - z2, np.sin(z1) * ((a1 - a2 + math.pi) % TAU - math.pi))))


class TestASpinBoxWillNotHoldWhatItIsGiven:
    """
    The Qt behaviour the rest of this depends on, pinned down so that it is not a belief.

    Both outcomes below are wrong; they differ only in how obviously. That is the reason
    set_projection_parameters exists rather than a wider range or a wrapping flag.
    """
    def test_a_value_below_the_range_does_not_survive(self, window):
        window.pw_a0.set_true_value(-0.05)          # a0 = -2.8648 deg, as a fit may return it

        assert window.pw_a0.true_value != pytest.approx(-0.05)

    def test_it_is_replaced_silently(self, window):
        """ No exception, no warning: the caller has no way to notice. """
        window.pw_a0.set_true_value(-0.05)

        assert window.pw_a0.true_value == pytest.approx(TAU, abs=1e-6)

    def test_wrapping_alone_would_leave_it_wrong(self, window):
        """
        The near miss. With wrapping on, -2.8648 deg comes back as 359.999999 rather than 0 -- an
        error of 2.86 degrees instead of 357, which is far likelier to go unnoticed.
        """
        window.pw_a0.set_true_value(-0.05)
        wanted = math.degrees(-0.05) % 360

        assert abs(window.pw_a0.display_value - wanted) == pytest.approx(2.8648, abs=1e-3)

    def test_the_azimuths_wrap_so_a_person_can_turn_past_the_end(self, window):
        for name in ('a0', 'F', 'E'):
            assert getattr(window, f'pw_{name}').dsb_value.wrapping(), name

    def test_the_zenith_distance_holds_a_whole_half_turn(self, window):
        """ Normalising can produce anything up to pi, and a range of 90 would silently truncate. """
        window.pw_epsilon.set_true_value(math.pi)

        assert window.pw_epsilon.true_value == pytest.approx(math.pi, abs=1e-6)


class TestSetProjectionParameters:
    """ The fix: normalise the whole plate, then write. """
    #: What the widgets quantise to, six decimals of a degree
    TOLERANCE = math.radians(1e-6)

    @pytest.mark.parametrize('values, description', [
        (KY, 'already in range'),
        (replaced(A0, -0.05), 'a0 below zero'),
        (replaced(A0, KY[A0] + 3 * TAU), 'a0 past three turns'),
        (replaced(F, -2.0), 'F below zero'),
        (replaced(E, KY[E] + TAU), 'E past a turn'),
        (replaced(EPSILON, -KY[EPSILON]), 'epsilon negative'),
        (replaced(EPSILON, TAU - KY[EPSILON]), 'epsilon reflex'),
    ])
    def test_the_widgets_end_up_describing_the_plate_that_was_fitted(self, window, values,
                                                                     description):
        write(window, values)

        assert sky_distance(values, read(window)) < 1e-6, description

    @pytest.mark.parametrize('values, description', [
        (replaced(A0, -0.05), 'a0 below zero'),
        (replaced(EPSILON, -KY[EPSILON]), 'epsilon negative'),
    ])
    def test_writing_the_raw_values_would_not(self, window, values, description):
        """
        The control, without which the test above would pass on a broken implementation.

        Only the property is asserted -- that the plate the widgets describe is nowhere near the
        one that was fitted -- because how far wrong it goes is not something worth predicting.
        Measured, the rotation case moves the sky by 0.050 rad and the zenith case by 1.93 rad, and
        the second is large for a reason worth knowing: a small negative epsilon is not a small
        error. It sends the shifter down its general branch with sin(epsilon) negative, which turns
        the azimuth around, so -0.0137 clamped to 0 is most of a half turn away rather than
        0.0137 away.
        """
        for value, widget in zip(values, window.param_widgets.values()):
            widget.set_true_value(value)

        moved = sky_distance(values, read(window))

        assert moved > 10000 * self.TOLERANCE, f"{description}: moved {moved:.4f} rad"

    @pytest.mark.parametrize('values', [KY, replaced(A0, -0.05), replaced(EPSILON, -0.5)])
    def test_what_is_written_is_what_can_be_read_back(self, window, values):
        written = write(window, values)

        assert read(window) == pytest.approx(np.array(written), abs=self.TOLERANCE)

    def test_everything_lands_inside_the_range_its_widget_allows(self, window):
        write(window, replaced(EPSILON, -8.0))

        for name, widget in window.param_widgets.items():
            box = widget.dsb_value
            assert box.minimum() <= widget.display_value <= box.maximum(), name

    def test_a_negative_zenith_distance_turns_its_azimuth_around(self, window):
        written = write(window, replaced(EPSILON, -KY[EPSILON]))

        assert written[EPSILON] == pytest.approx(KY[EPSILON])
        assert written[E] == pytest.approx((KY[E] + math.pi) % TAU)

    def test_the_constants_that_are_not_angles_pass_through(self, window):
        written = write(window, replaced(A0, -0.05))

        for index in (0, 1, 3, 5, 6, 7, 8, 9):
            assert written[index] == pytest.approx(KY[index]), NAMES[index]

    def test_it_returns_what_it_wrote(self, window):
        written = write(window, replaced(A0, KY[A0] - TAU))

        assert written[A0] == pytest.approx(KY[A0])
        assert written == BorovickaProjection(*replaced(A0, KY[A0] - TAU)).normalised().as_tuple()

    def test_writing_twice_settles(self, window):
        once = write(window, replaced(A0, -0.05))
        twice = write(window, once)

        assert twice == pytest.approx(np.array(once))
