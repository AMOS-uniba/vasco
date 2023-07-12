import pytest
import yaml
import dotmap

from models.sensordata import SensorData
from models.catalogue import Catalogue


@pytest.fixture
def hyg30():
    cat = Catalogue()
    cat.load('catalogues/HYG30.tsv')
    return cat


@pytest.fixture
def sd():
    contents = dotmap.DotMap(yaml.safe_load(open('data/20220531_055655.yaml', 'r')))
    return SensorData.load(contents)


class TestCatalogue():
    def test_dimensions(self, hyg30):
        assert hyg30.skycoord.size == 5068


class TestSensorData():
    def test_dimensions(self, sd):
        assert sd.stars.count == 702
