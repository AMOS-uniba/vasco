"""
A zero-padded frame number is a decimal frame number.

Every AMOS sighting file writes them padded -- `fno: 015` -- and PyYAML implements YAML 1.1, where
a leading zero means octal. Read that way a trail of frames 15 to 34 comes out as 13, 14, 15, 18,
19, 16, ... : the ones that are valid octal are reinterpreted, and `018` and `019` are not valid
octal at all so PyYAML leaves them as the strings '018' and '019', which int() then turns back
into 18 and 19. So 18 and 19 appear twice and six frames are lost.

Nothing downstream could notice. The dots are in the right places and only their labels are wrong,
so the fit is unaffected and every plot looks right. What breaks is the one thing frame numbers are
for: tying a sky position back to the frame it came from, which is what the server matches a
reduction on.
"""
import io

import pytest

from vasco import yaml_io
from vasco.io import load_sighting

SIGHTING = 'data/M20151105_231201_KNM__00033.yaml'


class TestZeroPaddingIsNotOctal:
    def test_a_padded_scalar_keeps_its_value(self):
        document = yaml_io.load(io.StringIO("a: 015\nb: 020\nc: 0100\nd: 018\n"))

        assert document == dict(a=15, b=20, c=100, d=18)

    def test_a_real_trail_comes_out_in_order_and_without_repeats(self):
        data = load_sighting(SIGHTING)
        fnos = data.meteor.fnos(masked=False).tolist()

        assert fnos == sorted(fnos)
        assert len(set(fnos)) == len(fnos)
        assert fnos == list(range(fnos[0], fnos[0] + len(fnos)))

    def test_pyyaml_would_have_got_it_wrong(self):
        """
        Not a test of vasco, but of the reason for yaml_io: without it this is what happens, and it
        is worth having written down where the next person will find it.
        """
        pyyaml = pytest.importorskip('yaml')
        document = pyyaml.safe_load("a: 015\nb: 020\nd: 018\n")

        assert document['a'] == 13
        assert document['b'] == 16
        # 8 is not an octal digit, so PyYAML cannot read this as a number at all and leaves it a
        # string. int() then turns it back into 18 -- which is why the mangled sequence keeps its
        # eights and nines and only the others slide.
        assert document['d'] == '018'
        assert int(document['d']) == 18


class TestRoundTrip:
    def test_dump_and_load(self):
        data = {'format': 'amos-fit-result/1', 'reductions': [{'fit': {'stars': 58}}]}

        assert yaml_io.load(io.StringIO(yaml_io.dumps(data))) == data

    def test_dump_refuses_a_numpy_scalar(self):
        """
        Which is why the CLI casts everything to float on the way out: a representer error at the
        very end would throw away a fit that had already succeeded.
        """
        numpy = pytest.importorskip('numpy')

        with pytest.raises(Exception):
            yaml_io.dumps({'x': numpy.float64(1.5)})
