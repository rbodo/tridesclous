from .myqt import QT
import pyqtgraph as pg
from .base import WidgetBase
import numpy as np

from example.cbnu.utils import get_trigger_times, get_spiketrains, get_interval


class MyViewBox(pg.ViewBox):
    doubleclicked = QT.pyqtSignal()

    # noinspection PyPep8Naming
    def mouseDoubleClickEvent(self, ev):
        self.doubleclicked.emit()
        ev.accept()


class PSTH(WidgetBase):

    _params = [
        {'name': 'bin_method', 'type': 'list', 'value': 'auto', 'values':
            ['manual', 'auto', 'fd', 'doane', 'scott', 'stone',
             'rice', 'sturges', 'sqrt']},
        {'name': 'num_bins', 'type': 'int', 'value': 100, 'step': 10},
        {'name': 'pre', 'type': 'float', 'value': 1, 'step': 0.5,
         'suffix': 's', 'siPrefix': True},
        {'name': 'post', 'type': 'float', 'value': 1, 'step': 0.5,
         'suffix': 's', 'siPrefix': True}]

    def __init__(self, controller=None, parent=None):
        WidgetBase.__init__(self, parent, controller)

        self.catalogueconstructor = controller.cc
        self.canvas = pg.GraphicsLayoutWidget()
        self.layout = QT.QVBoxLayout()
        self.setLayout(self.layout)
        self.layout.addWidget(self.canvas)

        self.trigger_times = get_trigger_times(
            self.catalogueconstructor.cbnu.filepath)
        self.initialize_plot()

        self.tree_params.setWindowTitle("PSTH settings")
        self.params.param('num_bins').setLimits((1, 1e6))
        self.params.param('pre').setLimits((0, 1e3))
        self.params.param('post').setLimits((0, 1e3))
        self.params.param('bin_method').sigTreeStateChanged.connect(
            self.on_method_change)
        self.on_method_change()

    def initialize_plot(self):

        # Return if there are no triggers.
        if len(self.trigger_times) == 0:
            return

        us_per_tick = int(1e6 / self.catalogueconstructor.dataio.sample_rate)

        start = int(self.catalogueconstructor.cbnu.config['start_time'] * 1e6)
        spiketrains = get_spiketrains(self.catalogueconstructor, us_per_tick,
                                      start)

        pre = int(self.params['pre'] * 1e6)
        post = int(self.params['post'] * 1e6)
        bin_method = self.params['bin_method']
        num_bins = self.params['num_bins'] if bin_method == 'manual' \
            else bin_method

        histograms = {}
        ylim = 0
        for cluster_label, cluster_trains in spiketrains.items():
            spike_times_section = get_interval(cluster_trains,
                                               self.trigger_times[0] - pre,
                                               self.trigger_times[-1] + post)

            spike_times_zerocentered = []

            for trigger_time in self.trigger_times:
                t_pre = trigger_time - pre
                t_post = trigger_time + post

                x = get_interval(spike_times_section, t_pre, t_post)
                if len(x):
                    x -= trigger_time
                spike_times_zerocentered += list(x)

            cluster_counts, bin_edges = np.histogram(spike_times_zerocentered,
                                                     num_bins)
            histograms[cluster_label] = (bin_edges / 1e6, cluster_counts)
            # Update common plot range for y axis.
            max_count = np.max(cluster_counts)
            if max_count > ylim:
                ylim = max_count

        n = 2
        if len(histograms) > n * n:
            print("WARNING: Only {} out of {} available PSTH plots can be "
                  "shown.".format(n * n, len(histograms)))

        viewboxes = []
        for i in range(n):
            for j in range(n):
                if len(histograms) == 0:
                    return
                viewboxes.append(MyViewBox())
                plt = self.canvas.addPlot(row=i, col=j, viewBox=viewboxes[-1])
                cluster_label, (x, y) = histograms.popitem()
                color = self.controller.qcolors.get(cluster_label,
                                                    QT.QColor('white'))
                plt.plot(x, y, stepMode=True, fillLevel=0, brush=color)
                txt = pg.TextItem(str(cluster_label), color)
                txt.setPos(0, ylim)
                plt.addItem(txt)
                plt.setYRange(0, ylim)

                viewboxes[-1].doubleclicked.connect(self.open_settings)

    def on_method_change(self):
        if self.params['bin_method'] == 'manual':
            self.params.param('num_bins').show()
        else:
            self.params.param('num_bins').hide()
        self.refresh()

    def refresh(self):
        self.canvas.clear()
        self.initialize_plot()
