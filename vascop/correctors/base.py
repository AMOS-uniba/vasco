class BaseCorrector():
    def __init__(self, points, values):
        self.points = points
        self.values = values

    def __call__(self):
        raise NotImplementedError("The __call__ method of a corrector must be implemented")
