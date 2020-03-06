# coding=utf-8

import os
import numpy as np
import matplotlib.pyplot as plt
from example.cbnu.utils import get_interval
from scipy.signal import find_peaks
import pandas as pd
from collections import OrderedDict
import seaborn as sns
sns.set()


def remove_nan(array):
    return array[~np.isnan(array)]


def init_peak_times():
    _peak_times = {}
    for _key1 in heights:
        _peak_times[_key1] = {}
        for _key2 in widths:
            _peak_times[_key1][_key2] = {}
            for _key0 in ['Cathodic', 'Anodic']:
                _peak_times[_key1][_key2][_key0] = []
    return _peak_times


widths = [0.5, 1, 2, 4]
heights = [10, 30, 50]
data_path = 'C:\\Users\\bodor\\Documents\\Korea\\experiment\\' \
            'alternating_pulses_in_corners\\5uA_1ms_1Hz_cathodic'
pre = 0.01
post = 0.09

threshold_sweep = [1, 1.5, 2]
bin_sweep = [10, 20, 25, 33]  # 10 ms, 5 ms, 4 ms, 3 ms

# If true, fit a kernel-density estimate on PSTH and get the peak time from its
# maximum. Otherwise, peak time is the first time where a bin count exceeds
# threshold.
use_kde = False

# Select neighbors of stimulus electrode 55.
# cells_to_plot = []
cells_to_plot = ['ch_45a', 'ch_46a', 'ch_54a', 'ch_54b', 'ch_56a', 'ch_64a',
                 'ch_64b', 'ch_64c', 'ch_64d', 'ch_65a', 'ch_65b', 'ch_65c',
                 'ch_66a', 'ch_66b']

output_path = os.path.join(data_path, 'plots')
if not os.path.exists(output_path):
    os.makedirs(output_path)

input_path = 'C:\\Users\\bodor\\Documents\\Korea\\experiment\\stimulus_sweep'
times_filepath = os.path.join(input_path, 'Data.xlsx')
output_path = os.path.join(input_path, 'plots')
if not os.path.exists(output_path):
    os.makedirs(output_path)

trigger_sheet = pd.read_excel(times_filepath, sheet_name=0, header=[0, 1],
                              index_col=0, skiprows=0)

trigger_times = init_peak_times()
for key0 in ['Cathodic', 'Anodic']:
    for key1 in heights:
        for key2 in widths:
            key12 = '{}uA_{}ms'.format(key1, key2)
            trigger_times[key1][key2][key0] = remove_nan(
                    trigger_sheet[key0][key12].to_numpy())

spike_sheet = pd.read_excel(times_filepath, sheet_name=1, header=0)

spike_times = OrderedDict()
for cell_name, cell_data in spike_sheet.items():
    if 'ch_' not in cell_name:
        continue
    if len(cells_to_plot) and cell_name not in cells_to_plot:
        continue
    spike_times[cell_name] = remove_nan(cell_data.to_numpy())


def get_peak(_spike_times, _trigger_times, path, _cell_name, height, width,
             polarity, save_plot, _threshold, _num_bins, _pre, _post):

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

    spike_times_zerocentered = np.concatenate(spike_times_zerocentered)

    if len(spike_times_zerocentered) < 10:
        return

    sns_fig = sns.distplot(spike_times_zerocentered, _num_bins, hist=True,
                           rug=True, kde=use_kde, hist_kws={'align': 'left'})
    if use_kde:
        bin_edges, counts = sns_fig.get_lines()[0].get_data()
    else:
        bin_edges = np.array([patch.xy[0] for patch in sns_fig.patches])
        counts = np.array([patch.get_height() for patch in sns_fig.patches])

    sns_fig.set_xlabel("Time [ms]")

    median = np.median(counts)
    mad = np.median(np.abs(counts - median))
    min_height = median + _threshold * mad
    # The mad can be zero if there are few spikes. In that case, use mean:
    if min_height == 0:
        mean = np.mean(counts)
        std = np.std(counts)
        min_height = mean + _threshold * std
    if use_kde:
        peak_idxs, _ = find_peaks(counts, min_height)
        if len(peak_idxs) == 0:
            return
        peak_heights = counts[peak_idxs]
        max_peak_idx = peak_idxs[np.argmax(peak_heights)]
        peak_time = bin_edges[max_peak_idx]
    else:
        # Set pre-stimulus counts to zero so they are not considered when
        # finding peak.
        counts[bin_edges <= 0] = 0
        peak_idxs = np.flatnonzero(counts >= min_height)
        if len(peak_idxs) == 0:
            return
        peak_time = bin_edges[peak_idxs[0]]

    if save_plot:
        filepath = os.path.join(path, 'PSTH_{}_{}_{}_{}.png'.format(
            _cell_name, height, width, polarity))
        pre_ms = 1e3 * _pre
        post_ms = 1e3 * _post
        ymax = 0.1 if use_kde else sns_fig.get_ylim()[1]
        sns_fig.set_xlim(-pre_ms, post_ms)
        sns_fig.vlines(peak_time, 0, ymax, color='g')
        sns_fig.vlines(0, 0, ymax, color='r')
        sns_fig.hlines(min_height, -pre_ms, post_ms, color='y')
        sns_fig.get_figure().savefig(filepath)
        plt.clf()

    return peak_time


def run_single(path, save_plots, _threshold, _num_bins):
    _peak_times = init_peak_times()
    for _key0, section0 in trigger_times.items():
        for _key1, section1 in section0.items():
            for _key2, section2 in section1.items():
                for _cell_name, cell_spikes in spike_times.items():
                    # if '56a' not in _cell_name:
                    #     continue
                    peak = get_peak(cell_spikes, section2, path, _cell_name,
                                    _key0, _key1, _key2, save_plots,
                                    _threshold, _num_bins, pre, post)
                    if peak is not None:
                        _peak_times[_key0][_key1][_key2].append(peak)

    return _peak_times


def plot_peaks(peaks, path):
    data = {'peak_times': [], 'heights': [], 'widths': [], 'polarity': []}
    medians = {'peak_times': [], 'heights': [], 'widths': [], 'polarity': []}
    for _key0, _section0 in peaks.items():  # Height
        for _key1, _section1 in _section0.items():  # Width
            for _key2, _section2 in _section1.items():  # Polarity
                data['peak_times'].extend(_section2)
                data['heights'].extend([_key0] * len(_section2))
                data['widths'].extend([_key1] * len(_section2))
                data['polarity'].extend([_key2] * len(_section2))
                medians['peak_times'].append(np.median(_section2))
                medians['heights'].append(_key0)
                medians['widths'].append(_key1)
                medians['polarity'].append(_key2)

    data = pd.DataFrame(data)
    medians = pd.DataFrame(medians)

    for _key0 in peaks.keys():  # Height
        data_heights = data.query('heights == {}'.format(_key0))
        sns_fig = sns.violinplot(x='widths', y='peak_times', hue='polarity',
                                 data=data_heights, inner='point', split=True,
                                 scale='count', scale_hue=True)
        sns_fig.set_xticks(np.arange(len(widths)))
        sns_fig.set_xticklabels(widths)
        sns_fig.set_xlabel("Stimulus width [ms]")
        sns_fig.set_ylabel("Response times [ms]")
        sns_fig.set_ylim(- pre * 1e3, post * 1e3)
        sns_fig.legend_.remove()
        sns_fig.get_figure().savefig(os.path.join(path,
                                                  'peaks_{}'.format(_key0)))
        plt.clf()

    sns_fig = sns.lineplot(x='widths', y='peak_times', hue='heights',
                           style='polarity', data=medians, legend='full')
    sns_fig.set_ylim(0, 50)
    sns_fig.set_xticks(widths)
    sns_fig.set_xticklabels(widths)
    sns_fig.set_xlabel('Pulse width [ms]')
    sns_fig.set_xlabel('Peak response time [ms]')
    sns_fig.get_figure().savefig(os.path.join(path, 'medians'))
    plt.clf()

    # # Enable this when looking at a single cell:
    # _peak_times_cathodic = {}
    # _peak_times_anodic = {}
    # for _key0, _section0 in peaks.items():  # Height
    #     _peak_times_cathodic[_key0] = []
    #     _peak_times_anodic[_key0] = []
    #     for _key1, _section1 in _section0.items():  # Width
    #         for _key2, _section2 in _section1.items():  # Polarity
    #             if _key2 == "Cathodic":
    #                 _peak_times_cathodic[_key0] += _section2
    #             else:
    #                 _peak_times_anodic[_key0] += _section2
    # for k, v in _peak_times_cathodic.items():
    #     plt.plot(v, label=k)
    # plt.legend()
    # plt.show()
    # plt.clf()
    # for k, v in _peak_times_anodic.items():
    #     plt.plot(v, label=k)
    # plt.legend()
    # plt.show()


for num_bins in bin_sweep:
    for threshold in threshold_sweep:
        path_sweep = os.path.join(output_path, 'bins{}_threshold{}'.format(
            num_bins, threshold))

        print(path_sweep)

        if not os.path.exists(path_sweep):
            os.makedirs(path_sweep)

        peak_times = run_single(path_sweep, True, threshold, num_bins)

        plot_peaks(peak_times, path_sweep)
