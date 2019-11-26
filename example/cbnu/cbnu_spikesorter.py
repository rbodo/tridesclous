import os
import time
import tkinter as tk
import tkinter.ttk as ttk
from functools import partial
from tkinter import filedialog, messagebox

import matplotlib.pyplot as plt
import numpy as np
import pyqtgraph as pg
from scipy.io import savemat

import tridesclous as tdc


class ElectrodeSelector:

    def __init__(self, root):
        self.root = root

        style = ttk.Style()
        style.theme_use('winnative')  # 'clam', 'alt', 'classic', 'default'
        self.filetypes = ['.mcd', '.msrd', '.h5']

        self.num_rows = 8
        self.num_columns = 8
        self.to_skip = {(0, 0),
                        (0, self.num_columns - 1),
                        (self.num_rows - 1, 0),
                        (self.num_rows - 1, self.num_columns - 1)}
        self.to_disable = {(4, 0)}  # Reference electrode 15.

        # Parameters
        self.config = {'highpass_frequency': 100,
                       'lowpass_frequency': 5000,
                       'relative_threshold': 4,
                       'duration': 100,
                       'waveform_left_ms': -2,
                       'waveform_right_ms': 3,
                       'feature_extractor': 'pca_by_channel',
                       'n_components_by_channel': 4,
                       'clustering_method': 'gmm',
                       'n_clusters': 3}

        self.plot_types = ['raster', 'waveform', 'psth', 'isi']
        self.plot_format = 'png'

        self.wait_window = None
        self.filepath = None
        self.dataio = None
        self.output_path = None
        self.geometry = None
        self.electrodes = []
        self.statusbar = None

        self.main_container = ttk.Frame(root)
        self.main_container.pack()

        self.button_frame_left = ttk.Frame(self.main_container)
        self.button_frame_left.pack(side='left')
        self.button_frame_right = ttk.Frame(self.main_container)
        self.button_frame_right.pack(side='right')

        self.dataset_button()
        self.toggle_electrode_selection()
        self.save_button()
        for plot_type in self.plot_types:
            self.plot_button(plot_type)
        self.electrode_selection_frame()
        self.status_widgets()

        # Set x and y coordinates for the Tk root window.
        ws = root.winfo_screenwidth()  # width of the screen
        hs = root.winfo_screenheight()  # height of the screen
        self.initial_position = '+{w}+{h}'.format(w=(ws // 3), h=(hs // 3))
        self.root.geometry(self.initial_position)

    def top_level_menu(self):
        """Top level menu."""

        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Quit", command=self.quit)
        menubar.add_cascade(label="File", menu=filemenu)

    def dataset_button(self):
        """Button for loading the dataset."""

        ttk.Button(self.button_frame_left, text="Load dataset",
                   command=self.load_dataset).pack(
            fill='both', expand=True, padx=[10, 10], pady=[10, 10])

    def toggle_electrode_selection(self):

        def toggle_all():
            for electrode in self.electrodes:
                electrode.set(not electrode.get())

        ttk.Button(self.button_frame_left, text="Toggle all",
                   command=toggle_all).pack(fill='both', expand=True,
                                            padx=[10, 10], pady=[10, 10])

    def save_button(self):

        ttk.Button(self.button_frame_left, text="Save spike times",
                   command=self.save_spiketimes).pack(
            fill='both', expand=True, padx=[10, 10], pady=[10, 10])

    def plot_button(self, plot_type):
        """Button for showing plots."""

        ttk.Button(self.button_frame_right,
                   text="Show {} plots".format(plot_type),
                   command=partial(self.show_plots, plot_type=plot_type)).pack(
            fill='both', expand=True, padx=[10, 10], pady=[10, 10])

    def status_widgets(self):
        from tkinter.scrolledtext import ScrolledText
        self.statusbar = ScrolledText(height=6, width=64, wrap=tk.WORD)
        self.statusbar.insert('end', "TOOLTIP: Start by loading dataset.")
        self.statusbar.pack(fill='both', expand=True, side='bottom')

    def quit(self):
        """Quit GUI."""

        self.root.destroy()
        self.root.quit()

    def log(self, message):
        # if self.wait_label is not None:
        #     self.wait_label['text'] = message
        self.statusbar.insert('end', '\n\n' + message)
        self.statusbar.see('end')
        self.root.update()

    def show_wait_window(self):
        self.wait_window = tk.Toplevel()
        self.wait_window.transient(self.root)
        self.wait_window.title("INFO")
        self.wait_window.lift()
        self.wait_window.attributes('-alpha', 1)
        self.wait_window.geometry('200x100' + self.initial_position)
        ttk.Label(self.wait_window, text="Working on it... Please wait."
                  ).place(relx=0.1, rely=0.3)

        # Disable close button.
        self.wait_window.protocol('WM_DELETE_WINDOW', lambda: None)

    def close_wait_window(self):
        self.wait_window.destroy()
        self.wait_window.update()

    def electrode_selection_frame(self):
        """Window for electrode selection."""

        electrode_window = ttk.Frame(self.main_container)
        electrode_window.pack(expand=True, side='right',
                              padx=[10, 10], pady=[10, 10])

        self.geometry = []
        interelectrode_distance = 200  # micrometer

        for c in range(self.num_columns):
            for r in range(self.num_rows):
                if (r, c) in self.to_skip:
                    continue

                self.geometry.append((r * interelectrode_distance,
                                      c * interelectrode_distance))

                label = "{}{}".format(c + 1, r + 1)

                self.electrodes.append(tk.BooleanVar(value=True))

                if (r, c) in self.to_disable:
                    self.electrodes[-1].set(False)

                tk.Checkbutton(electrode_window, text=label,
                               variable=(self.electrodes[-1]),
                               indicatoron=False).grid(
                    row=r, column=c, padx=1, pady=1)

    def load_dataset(self):
        """Load dataset."""

        filepath = filedialog.askopenfilename(
            title="Select dataset.", initialdir='/',
            filetypes=[('MEA files', self.filetypes), ('all files', '*.*')])
        self.filepath = filepath

        has_file = self.check_dataset_filepath(filepath)

        if not has_file:
            return

        basepath, extension = os.path.splitext(filepath)
        self.output_path = os.path.join(os.path.dirname(basepath),
                                        'tdc_output',
                                        time.strftime("%Y%m%d-%H%M%S"))
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        self.dataio = tdc.DataIO(self.output_path)

        self.show_wait_window()
        self.dataio.set_data_source(type=extension[1:],
                                    filenames=[filepath], gui=self)
        self.close_wait_window()

        self.log("TOOLTIP: Click on channel number to start spike sorter.")

    def check_dataset_filepath(self, path):

        if path in {None, ''}:
            return False

        if not os.path.exists(path):
            msg = "File not found:\n{}".format(path)
            messagebox.showerror("Error", msg)
            return False

        basepath, extension = os.path.splitext(path)
        if extension not in self.filetypes:
            msg = "File format {} not supported. Must be one of {}." \
                  "".format(extension, self.filetypes)
            messagebox.showerror("Error", msg)
            return False

        if extension == '.msrd' and (os.path.basename(basepath) + '.msrs'
                                     not in os.listdir(os.path.dirname(path))):
            msg = "File not found: {}.msrs".format(basepath)
            messagebox.showerror("Error", msg)
            return False

        return True

    def get_catalogueconstructor(self, channel_rel, label):
        if not self.check_finished_loading():
            return

        label = 'ch' + label

        # Get absolute row index of data array.
        channel_abs = self.dataio.datasource.channel_names.index(label)

        channel_groups = {label: {
            'channels': [channel_abs],
            'geometry': {channel_abs: self.geometry[channel_rel]}}}
        path_probe = os.path.join(self.output_path, 'electrode_selection.prb')
        with open(path_probe, 'w') as f:
            f.write("channel_groups = {}".format(channel_groups))

        self.dataio.set_probe_file(path_probe)

        return tdc.CatalogueConstructor(self.dataio)

    def run_spikesorter(self, catalogueconstructor):

        self.show_wait_window()
        self.log("Processing channel {}...".format(
            catalogueconstructor.chan_grp[2:]))

        catalogueconstructor.set_preprocessor_params(
            highpass_freq=self.config['highpass_frequency'],
            lowpass_freq=self.config['lowpass_frequency'],
            relative_threshold=self.config['relative_threshold'])
        # Using the GPU only makes sense when processing large workloads
        # (many channels in parallel)
        # signalpreprocessor_engine='opencl', peakdetector_engine='opencl')

        # Median and MAD per channel
        catalogueconstructor.estimate_signals_noise()

        # Signal preprocessing and peak detection
        catalogueconstructor.run_signalprocessor(
            duration=self.config['duration'])

        # Extract a few waveforms
        catalogueconstructor.extract_some_waveforms(
            wf_left_ms=self.config['waveform_left_ms'],
            wf_right_ms=self.config['waveform_right_ms'])

        # Remove outlier spikes
        catalogueconstructor.clean_waveforms()

        # Feature extraction
        catalogueconstructor.extract_some_features(
            method=self.config['feature_extractor'],
            n_components_by_channel=self.config['n_components_by_channel'])

        # Clustering
        catalogueconstructor.find_clusters(
            method=self.config['clustering_method'],
            n_clusters=self.config['n_clusters'])

        self.close_wait_window()

    def run_spikesorter_on_channel(self, channel_rel, label):

        catalogueconstructor = self.get_catalogueconstructor(channel_rel,
                                                             label)

        # The catalogueconstructor tries to load processed data from disk,
        # stored during a previous run. If successful, skip processing here.
        if 'clusters' not in catalogueconstructor.arrays.keys():
            self.run_spikesorter(catalogueconstructor)

        return catalogueconstructor

    def open_tridesclous_gui(self, channel_rel, label):
        cc = self.run_spikesorter_on_channel(channel_rel, label)
        gui = pg.mkQApp()
        win = tdc.CatalogueWindow(cc, filepath=self.filepath)
        win.show()
        gui.exec_()

    def check_finished_loading(self):
        if self.dataio is None:
            msg = "Load dataset before running spike sorter."
            messagebox.showerror("Error", msg)
            return False
        if not hasattr(self.dataio, 'nb_segment'):
            # This handles the case where the user clicks on an electrode while
            # the dataset is loading.
            return False
        return True

    def save_spiketimes(self):
        if not self.check_finished_loading():
            return

        path = os.path.join(self.output_path, 'spiketimes')
        if not os.path.exists(path):
            os.makedirs(path)

        d = {i: j for i, j in enumerate(['A', 'B', 'C', 'D', 'E', 'F', 'G'])}

        channel_idx = 0
        for c in range(self.num_columns):
            for r in range(self.num_rows):
                if (r, c) in self.to_skip:
                    continue

                if not self.electrodes[channel_idx].get():
                    channel_idx += 1
                    continue

                label = "{}{}".format(c + 1, r + 1)

                catalogueconstructor = self.run_spikesorter_on_channel(
                    channel_idx, label)

                s_per_tick = 1 / self.dataio.sample_rate

                spiketrains = get_spiketrains(catalogueconstructor, s_per_tick)

                for cell_label, spike_times in spiketrains.items():
                    filename = 'ch{}_{}.mat'.format(label, d[cell_label])
                    filepath = os.path.join(path, filename)
                    savemat(filepath, {'timestamps': spike_times})

                channel_idx += 1

    def show_plots(self, plot_type):

        if not self.check_finished_loading():
            return

        window = tk.Toplevel()
        window.transient(self.root)
        window.title("{} plots".format(plot_type))
        window.lift()

        path_image_blank = os.path.join(self.output_path, 'blank.png')
        if not os.path.exists(path_image_blank):
            plt.figure(figsize=(4, 4))
            plt.savefig(path_image_blank, bbox_inches='tight', pad_inches=0)
            plt.close()

        channel_idx = 0
        for c in range(self.num_columns):
            for r in range(self.num_rows):
                if (r, c) in self.to_skip:
                    continue

                label = "{}{}".format(c + 1, r + 1)
                path_plots = os.path.join(self.output_path,
                                          'channel_group_ch{}'.format(label),
                                          'plots')
                if not os.path.exists(path_plots):
                    os.makedirs(path_plots)
                path_image = os.path.join(path_plots,
                                          plot_type + '.' + self.plot_format)

                if not os.path.exists(path_image):
                    if self.electrodes[channel_idx].get():
                        cc = self.run_spikesorter_on_channel(channel_idx,
                                                             label)
                        self.create_plots(plot_type, path_image, cc)
                    else:
                        path_image = path_image_blank

                image = tk.PhotoImage(file=path_image)
                image = image.subsample(3)
                command = partial(self.open_tridesclous_gui,
                                  channel_rel=channel_idx, label=label)
                b = tk.Button(window, image=image, compound='center',
                              width=image.width(), height=image.height(),
                              command=command, text='ch' + label)
                b.grid(row=r, column=c, padx=2, pady=2, sticky='NSEW')
                b.image = image

                channel_idx += 1

    def create_plots(self, plot_type, path, catalogueconstructor):
        if not hasattr(catalogueconstructor, 'colors'):
            # Colors are not restored when loading catalogue from disk.
            catalogueconstructor.refresh_colors()

        if plot_type == 'raster':
            self.plot_raster(catalogueconstructor)
        elif plot_type == 'waveform':
            self.plot_waveform(catalogueconstructor)
        elif plot_type == 'psth':
            self.plot_psth(catalogueconstructor)
        elif plot_type == 'isi':
            self.plot_isi(catalogueconstructor)
        else:
            raise NotImplementedError

        plt.savefig(path, bbox_inches='tight', pad_inches=0)
        plt.close()

    def plot_raster(self, catalogueconstructor):

        us_per_tick = int(1e6 / self.dataio.sample_rate)

        spiketrains = get_spiketrains(catalogueconstructor, us_per_tick)

        trigger_times = get_trigger_times(self.filepath)

        num_clusters = len(spiketrains)

        num_triggers = len(trigger_times)

        if num_triggers == 0:
            duration = self.config['duration']
            num_triggers = int(np.sqrt(duration))
            trigger_times = np.linspace(0, duration, num_triggers,
                                        endpoint=False, dtype=int)
            trigger_times *= int(1e6)  # Seconds to microseconds.
        else:
            # Rasterplot should start at t=0.
            trigger_times = np.insert(trigger_times, 0, 0)
            num_triggers += 1

        n = 2
        if num_clusters > n * n:
            print("WARNING: Only {} out of {} available plots can be shown."
                  "".format(n * n, num_clusters))

        fig, axes = plt.subplots(2, 2, figsize=(5, 5))
        for i in range(n):
            for j in range(n):
                if len(spiketrains) == 0:
                    axes[i, j].plot(0, 0)
                else:
                    cluster_label, raster = spiketrains.popitem()
                    color = catalogueconstructor.colors[cluster_label]
                    for t in range(num_triggers - 1):
                        mask = np.logical_and(
                            np.greater_equal(raster, trigger_times[t]),
                            np.less(raster, trigger_times[t + 1]))
                        x = raster[mask]
                        if len(x):
                            x -= x[0]
                        y = (t + 1) * np.ones_like(x)
                        axes[i, j].scatter(x, y, s=5, linewidths=0,
                                           c=np.expand_dims(color, 0))
                        axes[i, j].text(0, 0.4 * num_triggers, cluster_label,
                                        color=color, fontsize=28)
                axes[i, j].axis('off')

        fig.subplots_adjust(wspace=0, hspace=0)

    @staticmethod
    def plot_waveform(catalogueconstructor):
        waveforms = {cl: np.squeeze(w) for cl, w in
                     zip(catalogueconstructor.cluster_labels,
                         catalogueconstructor.centroids_median)}

        ymin = np.min(list(waveforms.values()))
        ymax = np.max(list(waveforms.values()))

        n = 2
        if len(waveforms) > n * n:
            print("WARNING: Only {} out of {} available plots can be shown."
                  "".format(n * n, len(waveforms)))

        fig, axes = plt.subplots(2, 2, figsize=(5, 5))
        for i in range(n):
            for j in range(n):
                if len(waveforms) == 0:
                    axes[i, j].plot(0, 0)
                else:
                    cluster_label, waveform = waveforms.popitem()
                    color = catalogueconstructor.colors[cluster_label]
                    axes[i, j].plot(waveform, color='k', linewidth=2)
                    axes[i, j].plot(waveform, color=color, linewidth=1)
                    axes[i, j].text(0, 0.9 * ymax,
                                    cluster_label, color=color, fontsize=28)
                axes[i, j].axis('off')
                axes[i, j].set_ylim(ymin, ymax)

        fig.subplots_adjust(wspace=0, hspace=0)

    def plot_psth(self, catalogueconstructor, num_bins=100):
        """Plot PSTH of spiketrains."""

        us_per_tick = int(1e6 / self.dataio.sample_rate)

        spiketrains = get_spiketrains(catalogueconstructor, us_per_tick)

        trigger_times = get_trigger_times(self.filepath)

        num_triggers = len(trigger_times)

        # Return if there are no triggers.
        if num_triggers == 0:
            print("No trigger data available; aborting PSTH plot.")
            return

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
            histograms[cluster_label] = cluster_counts
            # Update common plot range for y axis.
            max_count = np.max(cluster_counts)
            if max_count > ylim:
                ylim = max_count
        bin_edges -= bin_edges[0]  # Shift to zero.
        bin_edges /= 1e6  # microseconds to seconds.

        n = 2
        if len(histograms) > n * n:
            print("WARNING: Only {} out of {} available plots can be shown."
                  "".format(n * n, len(histograms)))

        fig, axes = plt.subplots(2, 2, figsize=(5, 5))
        for i in range(n):
            for j in range(n):
                if len(histograms) == 0:
                    axes[i, j].plot(0, 0)
                else:
                    cluster_label, cluster_counts = histograms.popitem()
                    color = catalogueconstructor.colors[cluster_label]
                    x = np.ravel([bin_edges[:-1], bin_edges[1:]], 'F')
                    y = np.ravel([cluster_counts, cluster_counts], 'F')
                    axes[i, j].fill_between(x, 0, y, facecolor=color,
                                            edgecolor='k')
                    axes[i, j].text(0.7 * bin_edges[-1], 0.7 * ylim,
                                    cluster_label, color=color, fontsize=28)
                axes[i, j].axis('off')
                axes[i, j].set_ylim(0, ylim)

        fig.subplots_adjust(wspace=0, hspace=0)

    def plot_isi(self, catalogueconstructor, num_bins=100):
        """Plot ISI of spiketrains."""

        us_per_tick = int(1e6 / self.dataio.sample_rate)

        spiketrains = get_spiketrains(catalogueconstructor, us_per_tick)

        num_clusters = len(spiketrains)

        n = 2
        if num_clusters > n * n:
            print("WARNING: Only {} out of {} available plots can be shown."
                  "".format(n * n, num_clusters))

        fig, axes = plt.subplots(2, 2, figsize=(5, 5))
        for i in range(n):
            for j in range(n):
                if len(spiketrains) == 0:
                    axes[i, j].plot(0, 0)
                else:
                    cluster_label, cluster_trains = spiketrains.popitem()
                    intervals = np.diff(cluster_trains)
                    counts, bin_edges = np.histogram(intervals, bins=num_bins)
                    color = catalogueconstructor.colors[cluster_label]
                    x = np.ravel([bin_edges[:-1], bin_edges[1:]], 'F')
                    y = np.ravel([counts, counts], 'F')
                    axes[i, j].fill_between(x, 0, y, facecolor=color,
                                            edgecolor='k')
                    axes[i, j].text(0.7 * bin_edges[-1], 0.7 * np.max(counts),
                                    cluster_label, color=color, fontsize=28)
                axes[i, j].axis('off')

        fig.subplots_adjust(wspace=0, hspace=0)


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


def main():
    # Open window for data loading and electrode selection.
    tk_root = tk.Tk()
    tk_root.title("CBNU SpikeSorter")
    gui = ElectrodeSelector(tk_root)
    tk_root.protocol('WM_DELETE_WINDOW', gui.quit)
    tk_root.mainloop()


if __name__ == '__main__':
    main()
