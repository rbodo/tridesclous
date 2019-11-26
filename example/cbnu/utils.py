import os

import numpy as np


def get_trigger_times(filepath):
    dirname, basename = os.path.split(filepath)
    basename, _ = os.path.splitext(basename)
    trigger_path = os.path.join(dirname, basename + '_trigger.npz')
    return np.load(trigger_path)['arr_0']


def get_spiketrains(catalogueconstructor, time_per_tick=None):
    spike_times = np.array(catalogueconstructor.all_peaks['index'])
    spike_labels = np.array(catalogueconstructor.all_peaks['cluster_label'])

    if time_per_tick is not None:
        spike_times = spike_times * time_per_tick

    spiketrains = {}
    for cluster_label in catalogueconstructor.positive_cluster_labels:
        spiketrains[cluster_label] = spike_times[spike_labels == cluster_label]

    return spiketrains
