import os

import numpy as np


def get_interval(timedata, t0, t1):
    mask = np.logical_and(np.greater_equal(timedata, t0),
                          np.less(timedata, t1))
    return timedata[mask]


def get_trigger_times(filepath):
    dirname, basename = os.path.split(filepath)
    basename, _ = os.path.splitext(basename)
    trigger_path = os.path.join(dirname, basename + '_stimulus.npz')
    return np.load(trigger_path)['times']


def get_spiketrains(catalogueconstructor, time_per_tick, start_time=0):
    spike_ticks = np.array(catalogueconstructor.all_peaks['index'])
    spike_labels = np.array(catalogueconstructor.all_peaks['cluster_label'])

    spike_times = spike_ticks * time_per_tick + start_time

    spiketrains = {}
    for cluster_label in catalogueconstructor.positive_cluster_labels:
        spiketrains[cluster_label] = spike_times[spike_labels == cluster_label]

    return spiketrains
