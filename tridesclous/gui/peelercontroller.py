from .myqt import QT
import pyqtgraph as pg

import numpy as np
import seaborn as sns


from .base import ControllerBase


class PeelerController(ControllerBase):
    def __init__(self, parent=None, dataio=None, catalogue=None):
        ControllerBase.__init__(self, parent=parent)
        self.dataio=dataio
        self.catalogue=catalogue
        
        self.chan_grp = catalogue['chan_grp']
        self.nb_channel = self.dataio.nb_channel(self.chan_grp)
        
        
        self.init_plot_attributes()
    
    def init_plot_attributes(self):
        #concatenate all spikes for all segments
        self.spikes = []
        
        for i in range(self.dataio.nb_segment):
            local_spikes = self.dataio.get_spikes(seg_num=i, chan_grp=self.chan_grp)
            _dtype_spike = [('index', 'int64'), ('label', 'int64'), ('jitter', 'float64'),]
            _dtype_complement = [('segment', 'int64'), ('visible', 'bool'), ('selected', 'bool')]
            spikes = np.zeros(local_spikes.shape, dtype=_dtype_spike+_dtype_complement)
            for k, _ in _dtype_spike:
                spikes[k] = local_spikes[k]
            spikes['segment'] = i
            spikes['visible'] = True
            spikes['selected'] = False
            self.spikes.append(spikes)
        self.spikes = np.concatenate(self.spikes)
        
        self.nb_spike = int(self.spikes.size)
        self.cluster_labels = np.unique(self.spikes['label'])
        
        self.cluster_count = { k:np.sum(self.spikes['label']==k) for k in self.cluster_labels}
        
        
        self.cluster_visible = {k:True for k  in self.cluster_labels}
        self.refresh_colors(reset=True)
    
    def check_plot_attributes(self):
        for k in self.cluster_labels:
            if k not in self.cluster_visible:
                self.cluster_visible[k] = True
        
        for k in list(self.cluster_visible.keys()):
            if k not in self.cluster_labels:
                self.cluster_visible.pop(k)
        
        self.refresh_colors(reset=False)
    
    @property
    def spike_selection(self):
        return self.spikes['selected']
    
    def refresh_colors(self, reset=True, palette = 'husl'):
        if reset:
            self.colors = {}
        
        n = self.cluster_labels.size
        color_table = sns.color_palette(palette, n)
        for i, k in enumerate(self.cluster_labels):
            if k not in self.colors:
                self.colors[k] = color_table[i]
        
        self.colors[-1] = (.4, .4, .4)
        
        self.qcolors = {}
        for k, color in self.colors.items():
            r, g, b = color
            self.qcolors[k] = QT.QColor(r*255, g*255, b*255)
    
    def get_threshold(self):
        threshold = self.catalogue['params_peakdetector']['relative_threshold']
        if self.catalogue['params_peakdetector']['peak_sign']=='-':
            threshold = -threshold
        return threshold

    def get_max_on_channel(self, label):
        cluster_idx = self.catalogue['label_to_index'][label]
        chan = self.catalogue['max_on_channel'][cluster_idx]
        return chan

    def update_visible_spikes(self):
        visibles = np.array([k for k, v in self.cluster_visible.items() if v ])
        self.spikes['visible'][:] = np.in1d(self.spikes['label'], visibles)

    def on_cluster_visibility_changed(self):
        self.update_visible_spikes()
        ControllerBase.on_cluster_visibility_changed(self)
