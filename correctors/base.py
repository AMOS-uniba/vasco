from abc import ABCMeta, abstractmethod


class BaseCorrector(metaclass=ABCMeta):
    def __init__(self, points, values):
        self.points = points
        self.values = values

    @abstractmethod
    def __call__(self, nodes):
        """ Estimate values of the function at nodes """
