import pytest
from sensordata import SensorData


@pytest.fixture
def hyg30():
    return SensorData('catalogue/HYG30.tsv')


class TestSensorData():
    def test_dimensions(self, hyg30):
        assert hyg30.points.shape == (1615, 2)
