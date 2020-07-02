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

PRE = 0.1
POST = 0.1


def remove_nan(array):
    return array[~np.isnan(array)]


def decode_header(header):

    height, width = header.split('_')

    height = height.replace('uA', '')
    width = width.replace('ms', '')

    return float(height), float(width)


def get_spiketimes_zerocentered(_spike_times, _trigger_times):
    spike_times_section = get_interval(_spike_times, _trigger_times[0] - PRE,
                                       _trigger_times[-1] + POST)

    spike_times_zerocentered = []

    for trigger_time in _trigger_times:
        t_pre = trigger_time - PRE
        t_post = trigger_time + POST

        x = get_interval(spike_times_section, t_pre, t_post)
        if len(x):
            x -= trigger_time  # Zero-center
            x *= 1e3  # Seconds to ms
        spike_times_zerocentered.append(x)

    return np.concatenate(spike_times_zerocentered)


def get_data(path, _use_nn):
    _data = []
    for experiment in os.listdir(os.path.join(path, 'Data')):
        stimulus_electrode = int(experiment[-2:])
        cells_to_use = []
        if _use_nn:
            for a in [-1, 1, -10, 10, -11, 11, -9, 9]:
                for b in ['a', 'b', 'c', 'd']:
                    cell_name = 'ch_{}{}'.format(stimulus_electrode + a, b)
                    cells_to_use.append(cell_name)

        subexperiments = os.listdir(os.path.join(path, 'Data', experiment))
        for subexperiment in subexperiments:
            trigger_times = {}
            filepath = os.path.join(path, 'Data', experiment, subexperiment)
            polarity = pd.read_excel(filepath, sheet_name=0, usecols=[1],
                                     nrows=1, squeeze=True).values[0]
            polarity = polarity.lower()  # Unify capital case.
            trigger_sheet = pd.read_excel(filepath, sheet_name=0, header=1,
                                          index_col=0, skiprows=1)
            column_labels = trigger_sheet.keys()
            for column_label in column_labels:
                height, width = decode_header(column_label)
                if height not in trigger_times:
                    trigger_times[height] = {}
                if width not in trigger_times[height]:
                    trigger_times[height][width] = {}
                trigger_times[height][width][polarity] = remove_nan(
                    trigger_sheet[column_label].to_numpy())

            spike_sheet = pd.read_excel(filepath, sheet_name=1, header=0)
            spike_times = OrderedDict()
            for cell_name, cell_data in spike_sheet.items():
                if len(cells_to_use):
                    if not np.any([c in cell_name for c in cells_to_use]):
                        continue
                elif 'ch_' not in cell_name:
                    continue
                spike_times[cell_name] = remove_nan(cell_data.to_numpy())

            _data.append((trigger_times, spike_times))

    return _data


def run(_data, _bin_sweep, _threshold_sweep):
    for num_bins in _bin_sweep:
        for threshold in _threshold_sweep:
            path_sweep = os.path.join(output_path, 'bins{}_threshold{}'.format(
                num_bins, threshold))

            print(path_sweep)

            if not os.path.exists(path_sweep):
                os.makedirs(path_sweep)

            peaks = run_single(path_sweep, True, threshold, num_bins, _data)

            plot_peaks(peaks, path_sweep)


def run_single(path, save_plots, _threshold, _num_bins, _data):
    peaks = {'peak_times': [], 'heights': [], 'widths': [], 'polarity': []}
    all_spikes = {}
    for i, (_trigger_times, _spike_times) in enumerate(_data):
        for _key0, section0 in _trigger_times.items():
            for _key1, section1 in section0.items():
                for _key2, section2 in section1.items():
                    for _cell_name, cell_spikes in _spike_times.items():
                        spike_times_zerocentered = get_spiketimes_zerocentered(
                            cell_spikes, section2)
                        keys = (_key0, _key1, _key2)
                        if keys not in all_spikes:
                            all_spikes[keys] = []
                        all_spikes[keys].extend(list(spike_times_zerocentered))
                        peak = get_peak(spike_times_zerocentered, path, i,
                                        _cell_name, _key0, _key1, _key2,
                                        save_plots, _threshold, _num_bins)
                        if peak is not None:
                            peaks['peak_times'].append(peak)
                            peaks['heights'].append(_key0)
                            peaks['widths'].append(_key1)
                            peaks['polarity'].append(_key2)

    peaks2 = {'peak_times': [], 'heights': [], 'widths': [], 'polarity': []}
    for (_key0, _key1, _key2), spikes in all_spikes.items():
        peak = get_peak(spikes, path, '', 'all', _key0, _key1, _key2,
                        save_plots, _threshold, _num_bins, False)
        peaks2['peak_times'].append(peak)
        peaks2['heights'].append(_key0)
        peaks2['widths'].append(_key1)
        peaks2['polarity'].append(_key2)
    peaks2 = pd.DataFrame(peaks2)

    plt.clf()
    sns_fig = sns.lineplot(x='widths', y='peak_times', hue='heights',
                           style='polarity', data=peaks2, legend='full')
    sns_fig.set(xscale='log')
    sns_fig.set_ylim(0, 50)
    widths = np.unique(peaks2['widths'].values)
    sns_fig.set_xticks(widths)
    sns_fig.set_xticklabels(widths)
    sns_fig.set_xlabel('Pulse width [ms] (log scale)')
    sns_fig.set_ylabel('Peak response time [ms]')
    sns_fig.get_figure().savefig(os.path.join(path, 'from_combined_PSTH'))
    plt.clf()

    return pd.DataFrame(peaks)


def get_peak(_spike_times, path, experiment_idx, _cell_name,
             height, width, polarity, save_plot, _threshold, _num_bins,
             use_kde=False):
    """
    :param _spike_times:
    :param path:
    :param experiment_idx:
    :param _cell_name:
    :param height:
    :param width:
    :param polarity:
    :param save_plot:
    :param _threshold:
    :param _num_bins:
    :param use_kde: If true, fit a kernel-density estimate on PSTH and get the
    peak time from its maximum. Otherwise, peak time is the first time where a
    bin count exceeds threshold.
    :return:
    """

    if len(_spike_times) < 10:
        return

    plt.clf()
    sns_fig = sns.distplot(_spike_times, _num_bins, hist=True, rug=True,
                           kde=use_kde)
    if use_kde:
        bin_edges, counts = sns_fig.get_lines()[0].get_data()
    else:
        bin_edges = np.array([patch.xy[0] for patch in sns_fig.patches])
        counts = np.array([patch.get_height() for patch in sns_fig.patches])

    sns_fig.set_xlabel("Time [ms]")

    counts_nonzero = counts[np.flatnonzero(counts)]
    if counts_nonzero.size == 0:
        return
    else:
        median = np.median(counts_nonzero)
        mad = np.median(np.abs(counts_nonzero - median))
        min_height = median + _threshold * mad

    # Set pre-stimulus counts to zero so they are not considered when finding
    # peak.
    counts[bin_edges <= 0] = 0

    if use_kde:
        peak_idxs, _ = find_peaks(counts, min_height)
        if len(peak_idxs) == 0:
            return
        peak_heights = counts[peak_idxs]
        max_peak_idx = peak_idxs[np.argmax(peak_heights)]
        peak_time = bin_edges[max_peak_idx]
    else:
        min_height = max(min_height, 5)  # Want at least 5 spikes in a bin.
        peak_idxs = np.flatnonzero(counts >= min_height)
        if len(peak_idxs) == 0:
            return
        peak_time = bin_edges[peak_idxs[0]]

    if save_plot:
        filepath = os.path.join(path, 'PSTH_({})_{}_{}_{}_{}.png'.format(
            experiment_idx, _cell_name, height, width, polarity))
        pre_ms = 1e3 * PRE
        post_ms = 1e3 * POST
        ymax = sns_fig.get_ylim()[1]
        sns_fig.set_xlim(-pre_ms, post_ms)
        sns_fig.vlines(peak_time, 0, ymax, color='g')
        sns_fig.vlines(0, 0, ymax, color='r')
        sns_fig.hlines(min_height, -pre_ms, post_ms, color='y')
        sns_fig.get_figure().savefig(filepath)

    return peak_time


def plot_peaks(peaks, path):

    heights = np.unique(peaks['heights'].values)
    widths = np.unique(peaks['widths'].values)
    polarities = np.unique(peaks['polarity'].values)
    for height in heights:
        data_heights = peaks.query('heights == {}'.format(height))
        try:
            sns_fig = sns.violinplot(x='widths', y='peak_times',
                                     hue='polarity', data=data_heights,
                                     inner='point', split=True, scale='count')
        except ValueError:  # Can't use ``split`` if only cathodic xor anodic.
            sns_fig = sns.violinplot(x='widths', y='peak_times',
                                     hue='polarity', data=data_heights,
                                     inner='point', scale='count')
        sns_fig.set_xticks(np.arange(len(widths)))
        sns_fig.set_xticklabels(widths)
        sns_fig.set_xlabel("Stimulus width [ms]")
        sns_fig.set_ylabel("Response times [ms]")
        sns_fig.set_ylim(- PRE * 1e3, POST * 1e3)
        sns_fig.legend_.remove()
        sns_fig.get_figure().savefig(os.path.join(path, 'peaks_{}.png'
                                                        ''.format(height)))
        plt.clf()

    medians = {'peak_times': [], 'heights': [], 'widths': [], 'polarity': []}
    for height in heights:
        for width in widths:
            for polarity in polarities:
                peak_times = peaks['peak_times'][
                    (peaks['heights'] == height) &
                    (peaks['widths'] == width) &
                    (peaks['polarity'] == polarity)].values
                medians['peak_times'].append(np.median(peak_times))
                medians['heights'].append(height)
                medians['widths'].append(width)
                medians['polarity'].append(polarity)
    medians = pd.DataFrame(medians)

    sns_fig = sns.lineplot(x='widths', y='peak_times', hue='heights',
                           style='polarity', data=medians, legend='full')
    sns_fig.set(xscale='log')
    sns_fig.set_ylim(0, 50)
    sns_fig.set_xticks(widths)
    sns_fig.set_xticklabels(widths)
    sns_fig.set_xlabel('Pulse width [ms] (log scale)')
    sns_fig.set_ylabel('Peak response time [ms]')
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


if __name__ == '__main__':

    threshold_sweep = [1, 2, 3]
    bin_sweep = [20, 50, 100]  # 5 ms, 4 ms, 2 ms, 1 ms

    # If true, use only spikes from cells recorded at nearest neighbors of the
    # stimulation electrode.
    use_nn = True

    base_path = \
        'C:\\Users\\bodor\\Documents\\Korea\\experiment\\stimulus_sweep\\wt'
    output_path = os.path.join(base_path, 'plots')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    data = get_data(base_path, use_nn)

    run(data, bin_sweep, threshold_sweep)
