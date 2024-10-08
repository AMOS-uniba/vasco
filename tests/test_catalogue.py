import pytest

from vasco.models.sensordata import SensorData
from vasco.models.catalogue import Catalogue


@pytest.fixture
def hyg30():
    cat = Catalogue.load('catalogues/HYG30.tsv')
    return cat


@pytest.fixture
def sd():
    return SensorData.load_YAML('data/20220531_055655.yaml')


class TestCatalogue:
    def test_dimensions(self, hyg30):
        assert hyg30.skycoord.size == 5068


class TestSensorData:
    def test_dimensions(self, sd):
        assert sd.stars.count == 702
