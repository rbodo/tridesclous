# coding=utf-8

import os
import numpy as np
import matplotlib.pyplot as plt
from example.cbnu.utils import get_interval
from scipy.io import loadmat
from scipy.signal import find_peaks
from sklearn.cluster import k_means
import pandas as pd
import seaborn as sns
sns.set()

num_trials = 40
num_delays = 11
step_size_delays = 5
target_delays = step_size_delays * np.arange(num_delays)
data_path = 'C:\\Users\\bodor\\Documents\\Korea\\experiment\\' \
            'alternating_pulses_in_corners\\5uA_1ms_1Hz_cathodic'
trigger_path0 = os.path.join(data_path, 'Stim_Location_Green(ch61)_Blue(ch57)')
cell_path0 = os.path.join(trigger_path0, 'spiketimes')
trigger_path1 = os.path.join(data_path, 'Stim_Location_Green(ch61)_Blue(ch77)')
cell_path1 = os.path.join(trigger_path1, 'spiketimes')
pre = 0.01
post = 0.09

threshold_sweep = [1, 1.5, 2]
bin_sweep = [16, 20, 25, 33]  # 6.25 ms, 5 ms, 4 ms, 3 ms

output_path = os.path.join(data_path, 'plots')
if not os.path.exists(output_path):
    os.makedirs(output_path)

spike_times0 = {}
for cell_name in os.listdir(cell_path0):
    spike_times0[cell_name[:-4]] = loadmat(os.path.join(cell_path0, cell_name),
                                           squeeze_me=True)['timestamps']

trigger_times0 = []
for filename in os.listdir(trigger_path0):
    if 'trigger_times' in filename:
        trigger_times0.append(
            np.loadtxt(os.path.join(trigger_path0, filename)) / 1e6)

spike_times1 = {}
for cell_name in os.listdir(cell_path1):
    spike_times1[cell_name[:-4]] = loadmat(os.path.join(cell_path1, cell_name),
                                           squeeze_me=True)['timestamps']

trigger_times1 = []
for filename in os.listdir(trigger_path1):
    if 'trigger_times' in filename:
        trigger_times1.append(
            np.loadtxt(os.path.join(trigger_path1, filename)) / 1e6)

spike_times_list = [spike_times0, spike_times1]
trigger_times_list = [trigger_times0, trigger_times1]


def get_peaks(_spike_times, _trigger_times, path, _delay, _cell_name,
              save_plot, _threshold, _num_bins, _pre, _post):

    spike_times_section = get_interval(_spike_times, _trigger_times[0] - _pre,
                                       _trigger_times[-1] + _post)

    spike_times_zerocentered = []

    for trigger_time in _trigger_times:
        t_pre = trigger_time - _pre
        t_post = trigger_time + _post

        x = get_interval(spike_times_section, t_pre, t_post)
        if len(x):
            x -= trigger_time  # Zero-center
            x *= 1e3  # Seconds to ms
        spike_times_zerocentered.append(x)

    sns_fig = sns.distplot(np.concatenate(spike_times_zerocentered),
                           _num_bins, hist=True, rug=True, kde=True,
                           hist_kws={'align': 'left'})
    bin_edges, counts = sns_fig.get_lines()[0].get_data()

    sns_fig.set_xlabel("Time [ms]")

    median = np.median(counts)
    mad = np.median(np.abs(counts - median))
    min_height = median + _threshold * mad
    # mean = np.mean(counts)
    # std = np.std(counts)
    # min_height = mean + _threshold * std
    peak_idxs, _ = find_peaks(counts, min_height)
    peak_heights = counts[peak_idxs]
    sort_idxs = np.argsort(peak_heights)
    max_peak_idxs = peak_idxs[sort_idxs][-2:]

    ymax = 0.1  # axes.get_ylim()[1]
    peak_times = []
    if len(max_peak_idxs) > 0:
        peak_time = bin_edges[max_peak_idxs[0]]
        sns_fig.vlines(peak_time, 0, ymax, color='g')
        peak_times.append(peak_time)
    if len(max_peak_idxs) > 1:
        peak_time = bin_edges[max_peak_idxs[1]]
        sns_fig.vlines(peak_time, 0, ymax, color='b')
        peak_times.append(peak_time)

    if save_plot:
        pre_ms = 1e3 * _pre
        post_ms = 1e3 * _post
        filepath = os.path.join(path,
                                'PSTH_{}_{}.png'.format(_cell_name, _delay))
        sns_fig.set_xlim(-pre_ms, post_ms)
        sns_fig.vlines(0, 0, ymax, color='r')
        sns_fig.hlines(min_height, -pre_ms, post_ms, color='y')
        sns_fig.get_figure().savefig(os.path.join(filepath))
        plt.clf()

    return peak_times


def run_single(path, save_plots, _threshold, _num_bins):
    _peaks = [[] for _ in range(num_delays)]
    _peak_diffs = [[] for _ in range(num_delays)]
    for stim_id, (spike_times, trigger_times) in enumerate(zip(
            spike_times_list, trigger_times_list)):
        _path = os.path.join(path, 'stim{}'.format(stim_id))
        if not os.path.exists(_path):
            os.makedirs(_path)
        for _cell_name, cell_spikes in spike_times.items():
            for i in range(num_delays):
                delay = step_size_delays * i
                window = slice(i * num_trials, (i + 1) * num_trials)
                cell_peaks = get_peaks(cell_spikes, trigger_times[0][window],
                                       _path, delay, _cell_name, save_plots,
                                       _threshold, _num_bins, pre, post)
                if len(cell_peaks) == 2:
                    _peak_diffs[i].append(np.abs(cell_peaks[1] -
                                                 cell_peaks[0]))
                elif len(cell_peaks) == 1:
                    if i == 0:
                        _peak_diffs[i].append(cell_peaks[0])  # Or 0.
                    # if i == 1:
                    #     peak_diffs[i].append(cell_peaks[0])
                else:
                    _peak_diffs[i].append(-1)  # Dummy value.
                _peaks[i] += cell_peaks

    return _peaks, _peak_diffs


def plot_peak_diffs(_peak_diffs, path):
    sns_fig = sns.violinplot(data=_peak_diffs, inner='points', scale='count',
                             color='b')
    sns_fig.plot([0, num_delays - 1], [0, step_size_delays * (num_delays - 1)])
    sns_fig.set_xticklabels(step_size_delays * np.arange(num_delays))
    sns_fig.set_xlabel("Stimulus delay [ms]")
    sns_fig.set_ylabel("Response delay [ms]")
    medians = [np.median(d) for d in _peak_diffs]
    mse = np.sum(np.subtract(medians, target_delays) ** 2)
    sns_fig.set_title('{:.2f}'.format(mse))
    sns_fig.plot(medians, '--', color='b')
    sns_fig.get_figure().savefig(os.path.join(path, 'delay_diffs'))
    plt.clf()


def plot_peaks(_peaks, path):
    colors = ['b', 'orange']
    cluster_means = [[], []]
    clusters = []
    delays = []
    cluster_ids = []
    for i, delay_peaks in enumerate(_peaks):
        if len(delay_peaks) == 0:
            cluster_means[0].append(0)
            cluster_means[1].append(0)
            continue
        target_delay = i * step_size_delays
        delay_peaks = np.array(delay_peaks)
        weights = np.ones_like(delay_peaks)
        weights[delay_peaks > target_delay + 30] = 0.1
        weights[delay_peaks < 0] = 0
        if i == 0:
            mean, _, _ = k_means(np.expand_dims(delay_peaks, -1), 1, weights,
                                 np.array([[0]]), n_init=1, n_jobs=-1)
            mean = mean[0, 0]  # Remove empty axes.
            cluster_means[0].append(mean)
            cluster_means[1].append(mean)
            clusters += list(delay_peaks)
            delays += [i] * len(delay_peaks)
            cluster_ids += [0] * len(delay_peaks)
        else:
            means, labels, _ = k_means(np.expand_dims(delay_peaks, -1), 2,
                                       weights,
                                       np.array([[0], [target_delay]]),
                                       n_init=1, n_jobs=-1)
            means_sorted = np.sort(means, 0)
            if not np.array_equal(means, means_sorted):
                labels = np.logical_not(labels)
                means = means_sorted
            for cluster_id in [0, 1]:
                cluster = delay_peaks[labels == cluster_id]
                mean = means[cluster_id, 0]  # Second axis is empty.
                cluster_means[cluster_id].append(mean)
                clusters += list(cluster)
                delays += [i] * len(cluster)
                cluster_ids += [cluster_id] * len(cluster)

    clusters_ = pd.DataFrame(
        {'clusters': clusters, 'delays': delays, 'cluster_ids': cluster_ids})

    sns_fig = sns.violinplot(x='delays', y='clusters', hue='cluster_ids',
                             data=clusters_, inner='point', split=True,
                             scale='count', scale_hue=True)
    sns_fig.set_xticks(np.arange(num_delays))
    sns_fig.set_xticklabels(step_size_delays * np.arange(num_delays))
    sns_fig.set_xlabel("Stimulus delay [ms]")
    sns_fig.set_ylabel("Response times [ms]")
    sns_fig.plot(target_delays, colors[1])
    sns_fig.plot(cluster_means[0], colors[0], linestyle='--')
    sns_fig.plot(cluster_means[1], colors[1], linestyle='--')
    offset = np.mean(cluster_means[0])
    means_norm0 = np.array(cluster_means[0]) - offset
    means_norm1 = np.array(cluster_means[1]) - offset
    sns_fig.plot(means_norm0, colors[0], linestyle=':')
    sns_fig.plot(means_norm1, colors[1], linestyle=':')
    mse = np.sum(means_norm0 ** 2) + np.sum((means_norm1 - target_delays) ** 2)
    sns_fig.set_title('{:.2f}'.format(mse))
    sns_fig.hlines(0, 0, num_delays - 1, colors[0])
    sns_fig.legend_.remove()
    sns_fig.get_figure().savefig(os.path.join(path, 'peaks'))
    plt.clf()


for num_bins in bin_sweep:
    for threshold in threshold_sweep:
        path_sweep = os.path.join(output_path, 'bins{}_threshold{}'.format(
            num_bins, threshold))
        print(path_sweep)
        if not os.path.exists(path_sweep):
            os.makedirs(path_sweep)
        peaks, peak_diffs = run_single(path_sweep, True, threshold, num_bins)

        plot_peak_diffs(peak_diffs, path_sweep)
        plot_peaks(peaks, path_sweep)
