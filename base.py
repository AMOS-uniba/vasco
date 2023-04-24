import dotmap


from PyQt6.QtWidgets import QMainWindow
from main_ui import Ui_MainWindow

from matplotlib import pyplot as plt

from photometry import LogCalibration
from amos import AMOS


class MainWindowBase(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        plt.style.use('dark_background')
        self.position_errors = None
        self.magnitude_errors = None
        self.location = None
        self.time = None
        self.projection = None
        self.calibration = None
        self.matcher = None

        self.setupUi(self)
        self.param_widgets = [
            (self.dsb_x0, 'x0'), (self.dsb_y0, 'y0'), (self.dsb_a0, 'a0'), (self.dsb_A, 'A'), (self.dsb_F, 'F'),
            (self.dsb_V, 'V'), (self.dsb_S, 'S'), (self.dsb_D, 'D'), (self.dsb_P, 'P'), (self.dsb_Q, 'Q'),
            (self.dsb_eps, 'eps'), (self.dsb_E, 'E')
        ]

        self.settings = dotmap.DotMap(dict(
            resolution=dict(left=-1, bottom=-1, right=1, top=1)
        ))

        self.calibration = LogCalibration(4000)
        self.populateStations()
        self.updateProjection()

    def connectSignalSlots(self):
        self.ac_load_sighting.triggered.connect(self.loadSighting)
        self.ac_load_catalogue.triggered.connect(self.loadCatalogue)
        self.ac_load_constants.triggered.connect(self.importProjectionConstants)
        self.ac_save_constants.triggered.connect(self.exportProjectionConstants)
        self.ac_export_meteor.triggered.connect(self.exportCorrectedMeteor)
        self.ac_mask_unmatched.triggered.connect(self.maskSensor)
        self.ac_create_pairing.triggered.connect(self.pair)
        self.ac_about.triggered.connect(self.displayAbout)

        for widget, param in self.param_widgets:
            widget.valueChanged.connect(self.onParametersChanged)

        self.dt_time.dateTimeChanged.connect(self.updateTime)
        self.dt_time.dateTimeChanged.connect(self.onTimeChanged)

        self.dsb_lat.valueChanged.connect(self.onLocationChanged)
        self.dsb_lon.valueChanged.connect(self.onLocationChanged)

        self.pb_optimize.clicked.connect(self.minimize)
        self.pb_pair.clicked.connect(self.pair)
        self.pb_export.clicked.connect(self.exportProjectionConstants)
        self.pb_import.clicked.connect(self.importProjectionConstants)

        self.pb_mask_unidentified.clicked.connect(self.maskSensor)
        self.pb_mask_distant.clicked.connect(self.maskCatalogueDistant)
        self.pb_mask_faint.clicked.connect(self.maskCatalogueFaint)
        self.pb_reset.clicked.connect(self.resetValid)
        self.dsb_error_limit.valueChanged.connect(self.onErrorLimitChanged)

        self.hs_bandwidth.actionTriggered.connect(self.onBandwidthSettingChanged)
        self.hs_bandwidth.sliderMoved.connect(self.onBandwidthSettingChanged)
        self.hs_bandwidth.actionTriggered.connect(self.onBandwidthChanged)
        self.hs_bandwidth.sliderReleased.connect(self.onBandwidthChanged)
        self.sb_arrow_scale.valueChanged.connect(self.onArrowScaleChanged)
        self.sb_resolution.valueChanged.connect(self.onResolutionChanged)

        self.cb_show_errors.clicked.connect(self.plotPositionCorrectionErrors)
        self.cb_show_grid.clicked.connect(self.plotPositionCorrectionGrid)
        self.cb_interpolation.currentIndexChanged.connect(self.plotMagnitudeCorrectionGrid)

        self.tw_charts.currentChanged.connect(self.updatePlots)

    def populateStations(self):
        for name, station in AMOS.stations.items():
            self.cb_stations.addItem(station.name)

        self.cb_stations.currentIndexChanged.connect(self.selectStation)
