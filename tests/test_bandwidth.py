"""
Choosing a bandwidth, and why it cannot be chosen the obvious way.

The trap is worth stating because it is invisible: the obvious score for a smoother is how well it
reproduces the residuals it was built from, and that score is monotone in the bandwidth. It has its
minimum at zero, where the smoother returns each star its own residual and predicts nothing at all
about the space between them -- which is the only place a meteor ever is.

Leave-one-out has a minimum where the field's own scale is. The tests below assert exactly that
difference, on a field whose scale is known because it was made up.
"""
import numpy as np
import pytest
from demeteor.metrics import euclidean

from vasco.correctors import kernels
from vasco.correctors.bandwidth import DEFAULT_MIN, MIN_POINTS, loo_score, select

#: The scale of the made-up field: a sinusoid over the disk with about this wavelength
SCALE = 0.3


def field(points):
    """ A smooth vector field over the unit disk, and nothing like a constant. """
    return np.stack([0.01 * np.sin(points[:, 0] / SCALE),
                     0.01 * np.cos(points[:, 1] / SCALE)], axis=1)


@pytest.fixture
def sample():
    """ 120 points on the disk, the field, and noise a tenth of its amplitude. """
    generator = np.random.default_rng(20260825)
    radius = np.sqrt(generator.uniform(0, 1, 120))
    angle = generator.uniform(0, 2 * np.pi, 120)
    points = np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis=1)
    values = field(points) + generator.normal(0, 0.001, (120, 2))
    return points, values


def in_sample_score(points, values, bandwidth):
    """
    The score that cannot be used, computed the same way but without dropping the point.

    Here so that the next reader can see the difference rather than take it on trust.
    """
    distances = euclidean(np.expand_dims(points, 1), np.expand_dims(points, 0))
    weights = kernels.nexp(distances / bandwidth)
    predicted = (weights.T @ values) / np.sum(weights, axis=0)[:, None]
    return float(np.mean(np.sum(np.square(values - predicted), axis=1)))


class TestLeaveOneOutIsNecessary:
    def test_in_sample_error_falls_all_the_way_to_zero_bandwidth(self, sample):
        """ Which is why it cannot choose one: the best bandwidth by this measure is no smoothing
            at all, and the smoother then says nothing about any point it was not given. """
        points, values = sample
        scores = [in_sample_score(points, values, h) for h in (0.001, 0.01, 0.1, 1.0)]

        assert scores == sorted(scores), "monotone: smaller is always 'better'"

    def test_leave_one_out_does_not(self, sample):
        points, values = sample
        best, score, curve = select(points, values)
        bandwidths = [h for h, _ in curve]

        assert best not in (bandwidths[0], bandwidths[-1]), "an interior minimum, not an edge"
        assert score < loo_score(points, values, bandwidths[0])
        assert score < loo_score(points, values, bandwidths[-1])

    def test_it_finds_the_scale_of_the_field(self, sample):
        """
        Not the exact number -- the minimum is broad and the answer depends on how densely the
        points sample the field -- but the right order of magnitude, which is all a bandwidth is.
        """
        points, values = sample
        best, _, _ = select(points, values)

        assert SCALE / 10 < best < SCALE * 10

    def test_a_bandwidth_of_the_whole_sky_is_no_better_than_no_correction(self, sample):
        """ At a large enough bandwidth every point gets the same global mean, so the score
            approaches the plain variance of the residuals. """
        points, values = sample
        variance = float(np.mean(np.sum(np.square(values - values.mean(axis=0)), axis=1)))

        assert loo_score(points, values, 100.0) == pytest.approx(variance, rel=0.05)


class TestEdges:
    def test_too_few_points_cannot_be_cross_validated(self):
        points = np.random.default_rng(1).uniform(-1, 1, (MIN_POINTS - 1, 2))
        values = np.zeros((MIN_POINTS - 1, 2))

        assert loo_score(points, values, 0.1) == np.inf

    def test_and_select_says_so_and_falls_back(self):
        points = np.random.default_rng(1).uniform(-1, 1, (2, 2))
        best, score, _ = select(points, np.zeros((2, 2)))

        assert score == np.inf
        assert best == DEFAULT_MIN * 10

    def test_a_scalar_field_works_too(self, sample):
        """ The magnitude smoother's values have one component, not two, and get their own
            bandwidth: nothing says the two fields vary over the same distance. """
        points, values = sample
        best, score, _ = select(points, values[:, :1])

        assert np.isfinite(score)
        assert best > 0

    def test_identical_points_do_not_produce_a_nan(self):
        """ Two stars measured at the same place cannot predict each other away, and the score has
            to stay a number. """
        points = np.zeros((MIN_POINTS + 2, 2))
        values = np.ones((MIN_POINTS + 2, 2))

        assert np.isfinite(loo_score(points, values, 0.1))
