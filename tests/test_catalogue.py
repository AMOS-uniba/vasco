import pytest
from models.sensordata import SensorData
from models.catalogue import Catalogue


@pytest.fixture
def hyg30():
    return Catalogue('catalogue/HYG30.tsv')


@pytest.fixture
def sd():
    return SensorData('data/20220531_055655.yaml')


class TestCatalogue():
    def test_dimensions(self, hyg30):
        assert hyg30.skycoords.shape == (5068,)


class TestSensorData():
    def test_dimensions(self):
        pass
