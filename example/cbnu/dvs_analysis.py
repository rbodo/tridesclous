# coding=utf-8

import os
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from example.cbnu.utils import get_interval
from scipy.io import loadmat
from scipy.signal import find_peaks
import pandas as pd
import seaborn as sns
sns.set()

data_path = 'C:\\Users\\bodor\\Documents\\Korea\\experiment\\' \
            'dvs\\30uA_1ms_1Hz_cathodic'
exp_paths = ['Stim_Location_ch54', 'Stim_Location_ch61', 'Stim_Location_ch77']
trigger_filenames = [
    'dvs_2020-01-09T17-21-35_trigger_times_manually_extracted.txt',
    'dvs_2020-01-09T17-24-45_trigger_times_manually_extracted.txt',
    'dvs_2020-01-09T17-29-02_trigger_times_manually_extracted.txt']
pre = 0.01
post = 0.09

threshold_sweep = [1, 1.5, 2]
bin_sweep = [16, 20, 25, 33]  # 6.25 ms, 5 ms, 4 ms, 3 ms

output_path = os.path.join(data_path, 'plots')
if not os.path.exists(output_path):
    os.makedirs(output_path)

spike_times_list = []
trigger_times_list = []
for exp_path, trigger_filename in zip(exp_paths, trigger_filenames):
    cell_path = os.path.join(data_path, exp_path, 'spiketimes')

    spike_times = {}
    for cell_name in os.listdir(cell_path):
        spike_times[cell_name[:-4]] = loadmat(
            os.path.join(cell_path, cell_name), squeeze_me=True)['timestamps']
    spike_times_list.append(spike_times)

    trigger_times_list.append(np.loadtxt(os.path.join(data_path, exp_path,
                                                      trigger_filename)) / 1e6)

num_trials = np.min([len(t) for t in trigger_times_list])
for i, trigger_times in enumerate(trigger_times_list):
    trigger_times_list[i] = trigger_times[:num_trials]

relative_trigger_times = trigger_times_list[0] - trigger_times_list[0][0]


def run_single(path, save_plots, _threshold, _num_bins):
    spike_times_population = [[] for _ in range(num_trials)]
    for _spike_times, _trigger_times in zip(spike_times_list,
                                            trigger_times_list):
        for cell_spikes in _spike_times.values():
            for t, trigger_time in enumerate(_trigger_times):
                x = get_interval(cell_spikes, trigger_time - pre,
                                 trigger_time + post)
                if len(x):
                    x -= trigger_time  # Zero-center
                    x *= 1e3  # Seconds to ms

                spike_times_population[t] += list(x)

    _peaks = []
    for trigger_idx, _spike_times in enumerate(spike_times_population):
        figure = Figure()
        canvas = FigureCanvas(figure)
        axes = figure.subplots(1, 1)
        axes.set_xlabel("Time [ms]")
        counts, bin_edges, _ = axes.hist(_spike_times, _num_bins, align='left',
                                         histtype='stepfilled', facecolor='k')

        # median = np.median(counts)
        # mad = np.median(np.abs(counts - median))
        # min_height = median + 5 * mad
        mean = np.mean(counts)
        std = np.std(counts)
        min_height = mean + _threshold * std
        peak_idxs, _ = find_peaks(counts, min_height)

        if len(peak_idxs) == 0:
            _peaks.append(-1)
            continue

        peak_heights = counts[peak_idxs]
        max_peak_idx = peak_idxs[np.argmax(peak_heights)]
        peak_time = bin_edges[max_peak_idx]
        # Convert peak_time from ms to s.
        _peaks.append(peak_time / 1e3 + relative_trigger_times[trigger_idx])

        if save_plots:
            ymax = axes.get_ylim()[1]
            axes.vlines(peak_time, 0, ymax, color='g')
            pre_ms = 1e3 * pre
            post_ms = 1e3 * post
            axes.set_xlim(-pre_ms, post_ms)
            axes.vlines(0, 0, ymax, color='r')
            axes.hlines(min_height, -pre_ms, post_ms, color='y')
            figure.subplots_adjust(wspace=0, hspace=0)
            filepath = os.path.join(path, 'PSTH_{}.png'.format(trigger_idx))
            canvas.print_figure(filepath, bbox_inches='tight', dpi=100)

    return _peaks


def plot_peaks(_peaks, path):
    figure = Figure()
    canvas = FigureCanvas(figure)
    axes = figure.subplots(1, 1)
    axes.scatter(relative_trigger_times, _peaks)
    pearson = np.corrcoef(relative_trigger_times, _peaks)[0, 1]
    axes.set_title("Pearson correlation: {:.4f}".format(pearson))
    axes.set_xlabel("Stimulus time [s]")
    axes.set_ylabel("Response time [s]")
    canvas.print_figure(os.path.join(path, 'peaks'), bbox_inches='tight',
                        dpi=100)


def plot_residuals(_peaks, path):
    figure = Figure()
    canvas = FigureCanvas(figure)
    axes = figure.subplots(1, 1)
    data = np.subtract(_peaks, relative_trigger_times) * 1e3
    axes.plot(data, '.')
    pearson = np.corrcoef(_peaks, relative_trigger_times)[0, 1]
    axes.set_title("Pearson correlation: {:.4f}".format(pearson))
    xlabel = "Stimulus index"
    axes.set_xlabel(xlabel)
    ylabel = "Response delay [ms]"
    axes.set_ylabel(ylabel)
    canvas.print_figure(os.path.join(path, 'residuals'), bbox_inches='tight',
                        dpi=100)
    data = pd.DataFrame({xlabel: np.arange(len(data)), ylabel: data})
    sns_fig = sns.regplot(xlabel, ylabel, data=data)
    sns_fig.get_figure().savefig(os.path.join(path, 'residuals'))


for num_bins in bin_sweep:
    for threshold in threshold_sweep:
        path_sweep = os.path.join(output_path, 'bins{}_threshold{}'.format(
            num_bins, threshold))
        print(path_sweep)
        if not os.path.exists(path_sweep):
            os.makedirs(path_sweep)
        peaks = run_single(path_sweep, False, threshold, num_bins)

        plot_peaks(peaks, path_sweep)
        plot_residuals(peaks, path_sweep)
