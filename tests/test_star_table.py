"""
The star table's columns.

The model renders cells from a `match index.column()` ladder of integer cases, and the column
widths in MainWindow are a positional list. Inserting a column means renumbering both by hand, and
getting it wrong shifts every cell after the insertion point onto the wrong heading -- silently,
and plausibly, since most of these columns are similar-looking numbers.

So each column is checked against a value that could only have come from its own field.
"""
import dotmap
import numpy as np
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from models.qstarmodel import QStarModel


@pytest.fixture(scope='session')
def qt_app():
    yield QApplication.instance() or QApplication([])


@pytest.fixture
def table():
    """ One row, every field a value that cannot be mistaken for another. """
    return dotmap.DotMap(
        px=np.array([11.0]), py=np.array([22.0]),
        x=np.array([0.333333]), y=np.array([0.444444]),
        alt=np.array([70.0]), az=np.array([123.456789]),
        intensity=np.array([1000.0]),
        mask=np.array([True]),
        star=np.array([4242]),
        name=np.array(['Bellatrix']),
        count=1,
        scalar_errors=np.array([0.5]),
        vector_errors=np.array([[0.25, 0.125]]),
        _dynamic=False,
    )


@pytest.fixture
def model(qt_app, table):
    return QStarModel(table)


def cell(model, column, row=0):
    return model.data(model.index(row, column), Qt.ItemDataRole.DisplayRole)


class TestColumns:
    def test_the_count_follows_the_headings(self, model):
        assert model.columnCount() == len(QStarModel.COLUMNS)

    def test_every_column_has_a_heading(self, model):
        for i in range(model.columnCount()):
            assert model.headerData(i, Qt.Orientation.Horizontal,
                                    Qt.ItemDataRole.DisplayRole) is not None, i

    def test_every_column_renders_something(self, model):
        for i in range(model.columnCount()):
            assert cell(model, i) is not None, QStarModel.COLUMNS[i]

    @pytest.mark.parametrize('heading, expected', [
        ('#', '0'),
        ('x [px]', '11.000'),
        ('y [px]', '22.000'),
        ('x [mm]', '0.333333'),
        ('y [mm]', '0.444444'),
        ('alt', '20.000000°'),            # rendered as the zenith distance, 90 - alt
        ('az', '123.456789°'),
        ('star', '4242'),
        ('name', 'Bellatrix'),
        ('alt error', '0.250000°'),
        ('az error', '0.125000°'),
        ('total error', '0.500000°'),
    ])
    def test_each_column_shows_its_own_field(self, model, heading, expected):
        assert cell(model, QStarModel.COLUMNS.index(heading)) == expected

    def test_the_used_column_is_a_tick(self, model):
        assert cell(model, QStarModel.COLUMNS.index('used')) == '✔'


class TestName:
    def test_it_reads_left_to_right_unlike_the_numbers(self, model):
        column = QStarModel.COLUMNS.index('name')
        alignment = model.data(model.index(0, column), Qt.ItemDataRole.TextAlignmentRole)

        assert alignment == (Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

    def test_a_table_without_names_does_not_break_the_row(self, qt_app, table):
        """ build_stars_table always supplies them, but the pairing column has the same guard. """
        table.name = None
        model = QStarModel(table)

        assert cell(model, QStarModel.COLUMNS.index('name')) == '-'
        assert cell(model, QStarModel.COLUMNS.index('total error')) == '0.500000°'

    def test_the_error_colouring_still_lands_on_the_error(self, model):
        """ The colour is keyed by column number, which the insertion moved. """
        column = QStarModel.COLUMNS.index('total error')

        assert model.data(model.index(0, column), Qt.ItemDataRole.ForegroundRole) is not None


class TestColumnWidths:
    def test_mainwindow_sets_one_width_per_column(self):
        """
        The widths are a positional list in update_stars_table, so it has to keep pace with
        COLUMNS. Read out of the source rather than by running the window, which needs a catalogue.
        """
        import ast
        import inspect

        import mainwindow

        source = inspect.getsource(mainwindow.MainWindow.update_stars_table)
        widths = next(node for node in ast.walk(ast.parse(source.strip()))
                      if isinstance(node, ast.List))

        assert len(widths.elts) == len(QStarModel.COLUMNS)


class TestCatalogueNames:
    """
    Turning a pairing index into a name.

    The pairing indexes the catalogue as Catalogue builds it, which is planets first and then
    stars. Nothing in vasco chooses that order and nothing here should assume it silently, so the
    test checks the alignment against vmag() -- the same array, the same order -- rather than only
    that the lengths agree, which an off-by-seven would satisfy.
    """
    SIX_COLUMN = ("#\n"
                  "ra\tdec\tdist\tvmag\tabsmag\tname\n"
                  "101.287215\t-16.716116\t2.6371\t-1.44\tSirius\n")

    @pytest.fixture
    def catalogue(self, tmp_path):
        """ A catalogue of three stars, written here so the test needs no data files. """
        from demeteor.catalogue import Catalogue

        path = tmp_path / 'tiny.tsv'
        path.write_text(
            "#\n"
            "ra\tdec\tdist\tvmag\tabsmag\tname\n"
            "101.287215\t-16.716116\t2.6371\t-1.44\t1.454\tSirius\n"
            "95.987925\t-52.69566\t94.7867\t-0.62\t-5.504\tCanopus\n"
            "213.91545\t19.18241\t11.2575\t-0.05\t-0.307\tunnamed\n",
            encoding='utf-8')
        return Catalogue(path)

    @pytest.fixture
    def matcher(self, catalogue):
        import datetime

        import astropy.units as u
        from astropy.coordinates import EarthLocation
        from astropy.time import Time
        from models.matcher import Matcher

        where = EarthLocation(17.27 * u.deg, 48.37 * u.deg, 531 * u.m)
        when = Time(datetime.datetime(2024, 9, 25, 21, 56, 37, tzinfo=datetime.UTC))
        return Matcher(where, when, catalogue=catalogue)

    def test_there_is_one_name_per_catalogue_entry(self, matcher):
        assert len(matcher.catalogue_names()) == matcher.catalogue.count

    def test_the_planets_come_first(self, matcher):
        from demeteor.catalogue import Catalogue

        names = matcher.catalogue_names()

        assert list(names[:len(Catalogue.PLANETS)]) == [p.title() for p in Catalogue.PLANETS]

    def test_the_stars_follow(self, matcher):
        from demeteor.catalogue import Catalogue

        names = matcher.catalogue_names()

        assert list(names[len(Catalogue.PLANETS):]) == ['Sirius', 'Canopus', 'unnamed']

    def test_it_lines_up_with_the_magnitudes(self, matcher):
        """
        The alignment that matters: an off-by-seven passes a length check and fails this.

        Note that the brightest object is not the brightest star -- Venus outshines Sirius by
        two and a half magnitudes -- which is the planets being genuinely in the array rather
        than an artefact of the indexing.
        """
        from demeteor.catalogue import Catalogue

        names = matcher.catalogue_names()
        vmags = matcher.catalogue.vmag(matcher.location, matcher.time, masked=False)
        planets = len(Catalogue.PLANETS)

        assert len(names) == len(vmags)
        assert names[vmags.argmin()] in [p.title() for p in Catalogue.PLANETS]
        assert names[planets + vmags[planets:].argmin()] == 'Sirius'

    def test_an_index_resolves_to_the_object_at_that_index(self, matcher):
        names = matcher.catalogue_names()
        stars = matcher.catalogue.stars.name.to_numpy()

        for i, name in enumerate(stars):
            assert names[len(matcher.catalogue.planets) + i] == name
