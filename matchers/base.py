from abc import ABCMeta, abstractmethod


class StarMatcher(metaclass=abc.ABCMeta):
    @abstractmethod
    def minimize(self):
        pass
