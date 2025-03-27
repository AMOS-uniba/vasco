from .base import Exporter


class DSVExporter(Exporter):
    """
    DSV based meteor exporter.
    """

    def export(self, filename):
        with open(filename, 'w') as file:
            file.write(self.matcher.print_meteor(self._projection, self._calibration))

    def print_meteor(self, data):
        df = super()._get_meteor()
        return df.to_csv()
