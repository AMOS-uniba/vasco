import pytest

from vasco.models.sensordata import SensorData


@pytest.fixture
def sd():
    return SensorData.load_YAML('data/20220531_055655.yaml')


class TestSensorData:
    def test_dimensions(self, sd):
        assert sd.stars.count == 702
