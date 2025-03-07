from .base import Exporter


class CSVExporter(Exporter):
    """
    CSV based meteor exporter.
    Currently a semi-hardcoded mess but works for typical use cases.
    """

    def export(self, filename):
        with open(filename, 'w') as file:
            file.write(self.matcher.print_meteor(self._projection, self._calibration))

    def print_meteor(self, data):
        df = super()._get_meteor()
        return None # mockup, finish
