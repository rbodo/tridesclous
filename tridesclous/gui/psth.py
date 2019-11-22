from .myqt import QT
import pyqtgraph as pg
from .base import WidgetBase
import numpy as np

from example.cbnu.cbnu_spikesorter import get_spiketrains, get_trigger_times


class PSTH(WidgetBase):

    def __init__(self, controller=None, parent=None, filepath=None):
        WidgetBase.__init__(self, parent, controller)

        self.catalogueconstructor = controller.cc
        self.canvas = pg.GraphicsLayoutWidget()
        self.layout = QT.QVBoxLayout()
        self.setLayout(self.layout)
        self.layout.addWidget(self.canvas)

        self.initialize_plot(filepath)

    def initialize_plot(self, filepath, num_bins=100):

        us_per_tick = int(1e6 / self.catalogueconstructor.dataio.sample_rate)

        spiketrains = get_spiketrains(self.catalogueconstructor, us_per_tick)

        trigger_times = get_trigger_times(filepath)

        num_triggers = len(trigger_times)

        # Return if there are no triggers.
        if num_triggers == 0:
            return

        # binsize = np.max(spiketrains) // num_bins
        bin_edges = None
        histograms = {}
        ylim = 0
        for cluster_label, cluster_trains in spiketrains.items():
            cluster_counts = np.zeros(num_bins)
            for t in range(num_triggers - 1):
                counts, bin_edges = np.histogram(cluster_trains, bins=num_bins,
                                                 range=(trigger_times[t],
                                                        trigger_times[t + 1]))
                cluster_counts += counts
            # No point in normalizing here because we've only counted some
            # subset of all the spikes (the catalogue constructor does not
            # assign all peaks to waveforms; this is the job of the peeler).
            # cluster_counts //= (binsize * num_triggers)
            histograms[cluster_label] = cluster_counts
            # Update common plot range for y axis.
            max_count = np.max(cluster_counts)
            if max_count > ylim:
                ylim = max_count
        bin_edges -= bin_edges[0]  # Shift to zero.
        bin_edges /= 1e6  # microseconds to seconds.

        n = 2
        if len(histograms) > n * n:
            print("WARNING: Only {} out of {} available PSTH plots can be "
                  "shown.".format(n * n, len(histograms)))

        for i in range(n):
            for j in range(n):
                if len(histograms) == 0:
                    return
                plt = self.canvas.addPlot(row=i, col=j)
                cluster_label, cluster_counts = histograms.popitem()
                color = self.controller.qcolors.get(cluster_label,
                                                    QT.QColor('white'))
                plt.plot(bin_edges, cluster_counts, stepMode=True, fillLevel=0,
                         brush=color)
                txt = pg.TextItem(str(cluster_label), color)
                txt.setPos(0, ylim)
                plt.addItem(txt)
                plt.setYRange(0, ylim)

    def on_spike_selection_changed(self):
        pass

    def on_spike_label_changed(self):
        pass

    def on_colors_changed(self):
        pass

    def on_cluster_visibility_changed(self):
        pass

    def on_params_changed(self):
        self.refresh()

    def refresh(self):
        pass
