import os
import shutil
import time
import tkinter as tk
import tkinter.ttk as ttk
from functools import partial
from tkinter import filedialog, messagebox

import win32api
import matplotlib.pyplot as plt
import numpy as np
import pyqtgraph as pg
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, \
    NavigationToolbar2Tk
from matplotlib.figure import Figure
from scipy.io import savemat

import tridesclous as tdc
from tridesclous.gui import gui_params
from tridesclous.gui.tools import ParamDialog, MethodDialog, \
    get_dict_from_group_param
from example.cbnu.utils import get_trigger_times, get_spiketrains, get_interval


class ElectrodeSelector:

    def __init__(self, root):
        self.root = root

        style = ttk.Style()
        style.theme_use('winnative')  # 'clam', 'alt', 'classic', 'default'
        self.filetypes = ['.mcd', '.msrd', '.h5']

        self.plot_types = ['raster', 'waveform', 'psth', 'isi']
        self.plot_format = 'png'

        self.wait_window = None
        self.filepath = None
        self.dataio = None
        self.output_path = None
        self.electrodes = {}
        self.statusbar = None
        self.interelectrode_distance = None  # micrometer
        self.to_disable = {}  # Reference electrode 15.
        self.labels = []
        self.geometry = {}
        self.electrode_window = None
        self.plot_canvas = None
        self.plot_window = None
        self.generate_default_geometry()

        # This list contains references to all catalogueconstructors in use.
        # Its only purpose is to allow closing the memmaps so we can move
        # files.
        self._cc = {}

        # Settings
        self.config = {}
        self.settings_version = 0
        self._has_time_changed = False
        self.preprocessing_dialog = None
        self.feature_dialog = None
        self.cluster_dialog = None
        self.init_settings_dialog()
        self.plot_settings_window = None
        self.plot_settings = {'pre': tk.DoubleVar(value=1),
                              'post': tk.DoubleVar(value=1),
                              'num_bins': tk.IntVar(value=100),
                              'bin_method': tk.StringVar(value='auto')}
        self._old_values = {}

        self.main_container = ttk.Frame(self.root)
        self.main_container.pack()

        self.button_frame_left = ttk.Frame(self.main_container)
        self.button_frame_left.pack(side='left')
        self.button_frame_right = ttk.Frame(self.main_container)
        self.button_frame_right.pack(side='right')

        # self.geometry_button()  # Enable when needed.
        self.top_level_menu()
        self.dataset_button()
        self.stimulus_button()
        self.toggle_electrode_selection()
        self.save_button()
        for plot_type in self.plot_types:
            self.plot_button(plot_type)
        self.electrode_selection_frame()
        self.status_widgets()

        # Get coordinates of secondary monitor
        monitors = win32api.EnumDisplayMonitors()
        i = max(0, len(monitors) - 1)
        x0, y0, x1, y1 = monitors[i][2]  # Top left and bottom right corners
        # Set x and y coordinates for the Tk root window.
        ws = root.winfo_screenwidth()  # width of the screen
        hs = root.winfo_screenheight()  # height of the screen
        self.initial_position = '+{w}+{h}'.format(w=(ws // 3 + x0),
                                                  h=(hs // 3 + y0))
        self.root.geometry(self.initial_position)

    def top_level_menu(self):
        """Top level menu."""

        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Settings", command=self.settings_dialog)
        filemenu.add_command(label="Plot settings",
                             command=self.show_plot_settings)
        filemenu.add_command(label="Quit", command=self.quit)
        menubar.add_cascade(label="File", menu=filemenu)

    def geometry_button(self):
        """Button for loading the electrode layout."""

        ttk.Button(self.button_frame_left, text="Load geometry",
                   command=self.load_geometry).pack(
            fill='both', expand=True, padx=[10, 10], pady=[10, 10])

    def stimulus_button(self):
        """Button for showing the stimulus over time."""

        ttk.Button(self.button_frame_left, text="Show stimulus",
                   command=self.show_stimulus).pack(
            fill='both', expand=True, padx=[10, 10], pady=[10, 10])

    def settings_button(self):
        """Button to modify default settings."""

        ttk.Button(self.button_frame_left, text="Settings",
                   command=self.settings_dialog).pack(
            fill='both', expand=True, padx=[10, 10], pady=[10, 10])

    def dataset_button(self):
        """Button for loading the dataset."""

        ttk.Button(self.button_frame_left, text="Load dataset",
                   command=self.load_dataset).pack(
            fill='both', expand=True, padx=[10, 10], pady=[10, 10])

    def toggle_electrode_selection(self):

        def toggle_all():
            for electrode in self.electrodes.values():
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
        self.wait_window.geometry('200x100' + self.initial_position)
        self.wait_window.transient(self.root)
        self.wait_window.title("INFO")
        self.wait_window.lift()
        self.wait_window.grab_set()
        ttk.Label(self.wait_window, text="Working on it... Please wait."
                  ).place(relx=0.1, rely=0.3)

        # Disable close button.
        self.wait_window.protocol('WM_DELETE_WINDOW', lambda: None)

    def close_wait_window(self):
        self.wait_window.destroy()
        self.wait_window.update()

    def electrode_selection_frame(self):
        """Window for electrode selection."""

        self.electrode_window = ttk.Frame(self.main_container)
        self.electrode_window.pack(expand=True, side='right',
                                   padx=[10, 10], pady=[10, 10])

        for channel_idx, (r, c) in self.geometry.items():

            self.electrodes[channel_idx] = tk.BooleanVar(value=True)

            r //= self.interelectrode_distance
            c //= self.interelectrode_distance

            if (r, c) in self.to_disable:
                self.electrodes[channel_idx].set(False)

            tk.Checkbutton(self.electrode_window,
                           text=self.labels[channel_idx],
                           variable=self.electrodes[channel_idx],
                           indicatoron=False).grid(row=r, column=c,
                                                   padx=1, pady=1)

    def generate_default_geometry(self):
        self.to_disable = {(4, 0)}  # Reference electrode 15.
        self.interelectrode_distance = 200

        num_rows = 8
        num_columns = 8
        to_skip = {(0, 0), (0, num_columns - 1), (num_rows - 1, 0),
                   (num_rows - 1, num_columns - 1)}

        for c in range(num_columns):
            for r in range(num_rows):
                if (r, c) in to_skip:
                    continue
                i = len(self.labels)
                self.labels.append("{}{}".format(c + 1, r + 1))
                self.geometry[i] = (r * self.interelectrode_distance,
                                    c * self.interelectrode_distance)

    def load_geometry(self):
        filepath = filedialog.askopenfilename(title="Select probe file.",
                                              initialdir='/')

        if filepath in {None, ''}:
            return

        d = {}
        with open(filepath) as f:
            exec(f.read(), None, d)
        channel_groups = d['channel_groups']
        if len(channel_groups) > 1:
            self.log("WARNING: Probe geometry contains more than one "
                     "channel group. Only the first is used.")
        channel_group = channel_groups[0]
        if 'geometry' not in channel_group:
            messagebox.showerror("Error", "Probe file contains no geometry.")
            return
        if 'channels' not in channel_group:
            messagebox.showerror("Error", "Probe file contains no channel "
                                          "list.")
            return

        self.labels = channel_group['channels']
        self.geometry = channel_group['geometry']

        # Estimate interelectrode distance. Only used for arranging plots in
        # electrode layout.
        coordinates = sorted(np.unique(list(self.geometry.values())))
        diff = np.diff(coordinates)
        self.interelectrode_distance = np.min(diff)

        # Update layout.
        self.electrode_window.destroy()
        self.electrode_window.update()
        self.electrode_selection_frame()

    def show_stimulus(self):

        if self.filepath in {None, ''}:
            msg = "Load dataset before plotting stimulus."
            messagebox.showerror("Info", msg)
            return

        dirname, basename = os.path.split(self.filepath)
        basename, _ = os.path.splitext(basename)
        trigger_path = os.path.join(dirname, basename + '_stimulus.npz')

        if not os.path.exists(trigger_path):
            msg = "File not found:\n{}".format(trigger_path)
            messagebox.showerror("Error", msg)
            return

        trigger_data = np.load(trigger_path)['data']

        window = tk.Toplevel()
        window.geometry(self.initial_position)
        window.title("Stimulus")
        window.lift()

        figure = Figure(figsize=(5, 5), dpi=100)
        axis = figure.add_subplot(111)
        axis.plot(trigger_data)
        axis.set_xlabel("Time [s]")
        xticks = np.linspace(0, len(trigger_data), 10)
        xticklabels = np.round(xticks / self.dataio.sample_rate +
                               self.config['start_time'], 2)
        axis.set_xticks(xticks)
        axis.set_xticklabels(xticklabels)
        axis.format_coord = lambda x, y: 'Time: {:.6f} s'.format(
            x / self.dataio.sample_rate + self.config['start_time'])
        canvas = FigureCanvasTkAgg(figure, window)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, window)
        toolbar.update()

    def init_settings_dialog(self):

        pg.mkQApp()

        start = 0
        stop = 100

        params = gui_params.fullchain_params
        params.insert(0, {'name': 'stop_time', 'type': 'float', 'value': stop,
                          'suffix': 's', 'siPrefix': True})
        params.insert(0, {'name': 'start_time', 'type': 'float',
                          'value': start, 'suffix': 's', 'siPrefix': True})
        self.preprocessing_dialog = ParamDialog(params,
                                                "Preprocessing settings")
        self.preprocessing_dialog.resize(350, 500)
        p = self.preprocessing_dialog.params
        p.param('duration').setValue(stop - start)
        p.param('duration').hide()
        p.param('preprocessor').param('highpass_freq').setValue(100)
        p.param('preprocessor').param('lowpass_freq').setValue(5000)
        p.param('peak_detector').param('relative_threshold').setValue(4)
        p.param('start_time').setLimits((0, 1e6))
        p.param('stop_time').setLimits((0, 1e6))
        p.param('start_time').sigTreeStateChanged.connect(self.on_time_change)
        p.param('stop_time').sigTreeStateChanged.connect(self.on_time_change)

        params = gui_params.features_params_by_methods
        self.feature_dialog = MethodDialog(
            params, "Feature extractor settings",
            selected_method='pca_by_channel')
        self.feature_dialog.all_params['pca_by_channel'].param(
            'n_components_by_channel').setValue(4)

        params = gui_params.cluster_params_by_methods
        self.cluster_dialog = MethodDialog(params, "Clustering settings",
                                           selected_method='kmeans')
        self.cluster_dialog.all_params['kmeans'].param(
            'n_clusters').setValue(3)

        self.update_config()

    def on_time_change(self):
        self._has_time_changed = True

    def settings_dialog(self):

        has_changed = False
        has_changed += self.preprocessing_dialog.exec_()
        has_changed += self.feature_dialog.exec_()
        has_changed += self.cluster_dialog.exec_()

        if not has_changed:
            return

        self.update_config()

        # If user has changed settings before loading dataset, we do not need
        # to continue here.
        if self.output_path is None:
            return

        # Move all saved results to a backup folder so the program runs spike
        # sorter again with new settings.
        path = os.path.join(self.output_path, '_old_settings_{}'.format(
            self.settings_version))
        os.mkdir(path)

        # Open memmap files in catalogueconstructor may prevent moving.
        self.close_memmap()

        for sub_dir in os.listdir(self.output_path):
            if 'channel_group_ch' in sub_dir:
                shutil.move(os.path.join(self.output_path, sub_dir), path)

        self.settings_version += 1

        if self._has_time_changed:
            self.dataio = None
            self.load_dataset(self.filepath)
            self._has_time_changed = False

    def show_plot_settings(self):
        self.plot_settings_window = tk.Toplevel()
        self.plot_settings_window.geometry(self.initial_position)
        self.plot_settings_window.title("Plot settings")
        self.plot_settings_window.transient(self.root)
        self.plot_settings_window.lift()
        self.plot_settings_window.grab_set()
        self.plot_settings_window.protocol('WM_DELETE_WINDOW',
                                           self.confirm_settings)

        self._old_values = {k: v.get() for k, v in self.plot_settings.items()}

        ttk.Label(self.plot_settings_window, text="bin_method").grid(
            row=0, column=0, padx=(10, 10), pady=(10, 10))
        methods = ['', 'manual', 'auto', 'fd', 'doane', 'scott', 'stone',
                   'rice', 'sturges', 'sqrt']
        ttk.OptionMenu(self.plot_settings_window,
                       self.plot_settings['bin_method'], *methods).grid(
            row=0, column=1, padx=(10, 10), pady=(10, 10))
        ttk.Label(self.plot_settings_window, text="num_bins").grid(
            row=1, column=0, padx=(10, 10), pady=(10, 10))
        ttk.Spinbox(self.plot_settings_window, from_=1, to=1e6, width=5,
                    textvariable=self.plot_settings['num_bins']).grid(
            row=1, column=1, padx=(10, 10), pady=(10, 10))
        ttk.Label(self.plot_settings_window, text="pre").grid(
            row=2, column=0, padx=(10, 10), pady=(10, 10))
        ttk.Spinbox(self.plot_settings_window, from_=0, to=1e6, increment=0.1,
                    width=5, textvariable=self.plot_settings['pre']).grid(
            row=2, column=1, padx=(10, 10), pady=(10, 10))
        ttk.Label(self.plot_settings_window, text="post").grid(
            row=3, column=0, padx=(10, 10), pady=(10, 10))
        ttk.Spinbox(self.plot_settings_window, from_=0, to=1e6, increment=0.1,
                    width=5, textvariable=self.plot_settings['post']).grid(
            row=3, column=1, padx=(10, 10), pady=(10, 10))

        ttk.Button(self.plot_settings_window, text="OK",
                   command=self.confirm_settings).grid(
            row=4, column=0, columnspan=2, padx=(10, 10), pady=(10, 10))

    def confirm_settings(self):

        has_changed = 0
        for k, v in self.plot_settings.items():
            has_changed += self._old_values[k] != v.get()

        if has_changed and self.output_path is not None:
            for filename in os.listdir(self.output_path):
                path = os.path.join(self.output_path, filename)
                if os.path.isdir(path) and 'channel_group_ch' in filename:
                    self.remove_plots(filename[-2:])

        self.plot_settings_window.destroy()
        self.plot_settings_window.update()

    def update_config(self):

        p = self.preprocessing_dialog.params
        start = p['start_time']
        stop = p['stop_time']
        if start > stop:
            messagebox.showwarning("Error", "Selected start_time must be "
                                            "smaller than stop_time.")
        p.param('duration').setValue(stop - start)

        config = self.preprocessing_dialog.get()

        config['feature_method'] = self.feature_dialog.param_method['method']
        config['feature_kargs'] = get_dict_from_group_param(
            self.feature_dialog.all_params[config['feature_method']],
            cascade=True)

        config['cluster_method'] = self.cluster_dialog.param_method['method']
        config['cluster_kargs'] = get_dict_from_group_param(
            self.cluster_dialog.all_params[config['cluster_method']],
            cascade=True)

        self.config.update(config)

    def load_dataset(self, filepath=None):
        """Load dataset."""

        if filepath is None:
            filepath = filedialog.askopenfilename(
                title="Select dataset.", initialdir='/',
                filetypes=[('MEA files', self.filetypes),
                           ('all files', '*.*')])
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

        self.log("TOOLTIP: Select which channels to process.")

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

        label = 'ch{}'.format(label)

        # Get absolute row index of data array.
        channel_abs = self.dataio.datasource.channel_names.index(label)

        channel_groups = {label: {
            'channels': [channel_abs],
            'geometry': {channel_abs: self.geometry[channel_rel]}}}
        path_probe = os.path.join(self.output_path, 'electrode_selection.prb')
        with open(path_probe, 'w') as f:
            f.write("channel_groups = {}".format(channel_groups))

        self.dataio.set_probe_file(path_probe)

        cc = tdc.CatalogueConstructor(self.dataio, cbnu=self)
        cc.info.update(self.config)
        cc.flush_info()

        self._cc[len(self._cc)] = cc

        return cc

    def run_spikesorter(self, catalogueconstructor):

        self.show_wait_window()
        self.log("Processing channel {}...".format(
            catalogueconstructor.chan_grp[2:]))

        params = {}
        params.update(self.config['preprocessor'])
        params.update(self.config['peak_detector'])
        catalogueconstructor.set_preprocessor_params(**params)

        # Median and MAD per channel
        catalogueconstructor.estimate_signals_noise()

        # Signal preprocessing and peak detection
        catalogueconstructor.run_signalprocessor(self.config['duration'])

        # Extract a few waveforms
        catalogueconstructor.extract_some_waveforms(
            **self.config['extract_waveforms'])

        # Remove outlier spikes
        catalogueconstructor.clean_waveforms(**self.config['clean_waveforms'])

        catalogueconstructor.extract_some_noise(**self.config['noise_snippet'])

        # Feature extraction
        catalogueconstructor.extract_some_features(
            self.config['feature_method'], **self.config['feature_kargs'])

        # Clustering
        catalogueconstructor.find_clusters(self.config['cluster_method'],
                                           **self.config['cluster_kargs'])

        catalogueconstructor.order_clusters()

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
        win = tdc.CatalogueWindow(cc)
        self.plot_window.withdraw()
        win.show()
        gui.exec_()
        self.plot_window.deiconify()

        # Remove any plots, because the user may have modified the clusters.
        # The plots will be re-created on demand.
        self.remove_plots(label)

    def remove_plots(self, label):
        """Remove plots from disk."""
        path = os.path.join(self.output_path,
                            'channel_group_ch{}'.format(label), 'plots')
        if os.path.exists(path):
            for filename in os.listdir(path):
                os.remove(os.path.join(path, filename))

    def close_memmap(self):
        for cc in self._cc.values():
            for array in list(cc.arrays.keys()):
                cc.arrays.detach_array(array, mmap_close=True)
            for arrays in cc.dataio.arrays.get(cc.chan_grp, []):
                for array in list(arrays.keys()):
                    arrays.detach_array(array, mmap_close=True)
        self._cc = {}

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

        saved_channels = []
        for channel_idx, electrode in self.electrodes.items():
            if not electrode.get():
                continue

            label = self.labels[channel_idx]

            saved_channels.append(label)

            catalogueconstructor = self.run_spikesorter_on_channel(channel_idx,
                                                                   label)

            s_per_tick = 1 / self.dataio.sample_rate

            spiketrains = get_spiketrains(catalogueconstructor, s_per_tick,
                                          self.config['start_time'])

            for cell_label, spike_times in spiketrains.items():
                filename = 'ch{}_{}.mat'.format(label, d[cell_label])
                filepath = os.path.join(path, filename)
                savemat(filepath, {'timestamps': spike_times})

        self.log("Spike times of channels {} saved to {}.".format(
            saved_channels, path))

    def show_plots(self, plot_type):

        if not self.check_finished_loading():
            return

        self.plot_window = tk.Toplevel()
        self.plot_window.geometry(self.initial_position)
        self.plot_window.transient(self.root)
        self.plot_window.title("{} plots".format(plot_type))
        self.plot_window.lift()
        self.plot_window.grab_set()

        hsb = tk.Scrollbar(self.plot_window, orient='horizontal')
        vsb = tk.Scrollbar(self.plot_window, orient='vertical')
        hsb.grid(row=1, column=0, sticky='ew')
        vsb.grid(row=0, column=1, sticky='ns')
        self.plot_canvas = tk.Canvas(
            self.plot_window, borderwidth=0, width=1000, height=800,
            xscrollcommand=hsb.set, yscrollcommand=vsb.set)
        self.plot_canvas.grid(row=0, column=0, sticky='nsew')
        hsb.config(command=self.plot_canvas.xview)
        vsb.config(command=self.plot_canvas.yview)

        self.plot_window.grid_rowconfigure(0, weight=1)
        self.plot_window.grid_columnconfigure(0, weight=1)

        table = ttk.Frame(self.plot_canvas)
        self.plot_canvas.create_window((0, 0), window=table, anchor='nw',
                                       tags='table')

        table.bind('<Configure>', lambda e: self.on_frame_configure())

        table.bind("<MouseWheel>", self.on_mousewheel)

        path_image_blank = os.path.join(self.output_path, 'blank.png')
        if not os.path.exists(path_image_blank):
            plt.figure(figsize=(4, 4))
            plt.savefig(path_image_blank, bbox_inches='tight', pad_inches=0)
            plt.close()

        for channel_idx, electrode in self.electrodes.items():

            label = self.labels[channel_idx]
            path_plots = os.path.join(self.output_path,
                                      'channel_group_ch{}'.format(label),
                                      'plots')
            if not os.path.exists(path_plots):
                os.makedirs(path_plots)
            path_image = os.path.join(path_plots,
                                      plot_type + '.' + self.plot_format)

            if not os.path.exists(path_image):
                if electrode.get():
                    cc = self.run_spikesorter_on_channel(channel_idx, label)
                    self.create_plots(plot_type, path_image, cc)
                else:
                    path_image = path_image_blank

            image = tk.PhotoImage(file=path_image)
            image = image.subsample(3)
            command = partial(self.open_tridesclous_gui,
                              channel_rel=channel_idx, label=label)
            b = tk.Button(table, image=image, compound='center',
                          width=image.width(), height=image.height(),
                          command=command, text='ch' + label)
            r, c = self.geometry[channel_idx]
            r //= self.interelectrode_distance
            c //= self.interelectrode_distance
            b.grid(row=r, column=c, padx=2, pady=2, sticky='NSEW')
            b.bind("<MouseWheel>", self.on_mousewheel)
            b.image = image

    def on_mousewheel(self, event):
        self.plot_canvas.yview_scroll(- event.delta // 120, 'units')

    def on_frame_configure(self):
        self.plot_canvas.config(scrollregion=self.plot_canvas.bbox('all'))

    def create_plots(self, plot_type, path, catalogueconstructor):
        if not hasattr(catalogueconstructor, 'colors'):
            # Colors are not restored when loading catalogue from disk.
            catalogueconstructor.refresh_colors()

        if plot_type == 'raster':
            canvas = self.plot_raster(catalogueconstructor)
        elif plot_type == 'waveform':
            canvas = self.plot_waveform(catalogueconstructor)
        elif plot_type == 'psth':
            canvas = self.plot_psth(catalogueconstructor)
        elif plot_type == 'isi':
            canvas = self.plot_isi(catalogueconstructor)
        else:
            raise NotImplementedError

        canvas.print_figure(path, bbox_inches='tight', pad_inches=0)
        plt.close()

    def plot_raster(self, catalogueconstructor):

        us_per_tick = int(1e6 / self.dataio.sample_rate)

        spiketrains = get_spiketrains(catalogueconstructor, us_per_tick,
                                      int(self.config['start_time'] * 1e6))

        trigger_times = get_trigger_times(self.filepath)

        num_clusters = len(spiketrains)

        num_triggers = len(trigger_times)

        if num_triggers == 0:
            duration = self.config['duration']
            num_triggers = int(np.sqrt(duration))
            trigger_times = np.linspace(0, duration, num_triggers,
                                        endpoint=False, dtype=int)
            trigger_times *= int(1e6)  # Seconds to microseconds.

        pre = int(self.plot_settings['pre'].get() * 1e6)
        post = int(self.plot_settings['post'].get() * 1e6)

        n = 2
        if num_clusters > n * n:
            print("WARNING: Only {} out of {} available plots can be shown."
                  "".format(n * n, num_clusters))

        fig = Figure(figsize=(5, 5))
        canvas = FigureCanvas(fig)
        axes = fig.subplots(2, 2)
        for i in range(n):
            for j in range(n):
                axes[i, j].axis('off')
                if len(spiketrains) == 0:
                    axes[i, j].plot(0, 0)
                    continue
                cluster_label, cluster_trains = spiketrains.popitem()
                if len(cluster_trains) == 0:
                    continue

                spike_times_section = get_interval(cluster_trains,
                                                   trigger_times[0] - pre,
                                                   trigger_times[-1] + post)
                spike_times_zerocentered = []
                color = catalogueconstructor.colors[cluster_label]
                for trigger_time in trigger_times:
                    x = get_interval(spike_times_section, trigger_time - pre,
                                     trigger_time + post)
                    if len(x):
                        x -= trigger_time
                    spike_times_zerocentered.append(x)
                axes[i, j].eventplot(spike_times_zerocentered, color=color,
                                     linewidths=0.5, lineoffsets=-1)
                axes[i, j].text(0, 0.5, cluster_label, fontsize=28,
                                color=color, transform=axes[i, j].transAxes)
                axes[i, j].vlines(0, *axes[i, j].get_ylim(), linewidth=1,
                                  alpha=0.9)
                axes[i, j].set_xlim(-pre, post)

        fig.subplots_adjust(wspace=0, hspace=0)
        return canvas

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

        fig = Figure(figsize=(5, 5))
        canvas = FigureCanvas(fig)
        axes = fig.subplots(2, 2)
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
        return canvas

    def plot_psth(self, catalogueconstructor):
        """Plot PSTH of spiketrains."""

        us_per_tick = int(1e6 / self.dataio.sample_rate)

        spiketrains = get_spiketrains(catalogueconstructor, us_per_tick,
                                      int(self.config['start_time'] * 1e6))

        trigger_times = get_trigger_times(self.filepath)

        # Return if there are no triggers.
        if len(trigger_times) == 0:
            print("No trigger data available; aborting PSTH plot.")
            return

        pre = int(self.plot_settings['pre'].get() * 1e6)
        post = int(self.plot_settings['post'].get() * 1e6)
        bin_method = self.plot_settings['bin_method'].get()
        num_bins = self.plot_settings['num_bins'].get() \
            if bin_method == 'manual' else bin_method

        histograms = {}
        ylim = 0
        for cluster_label, cluster_trains in spiketrains.items():
            spike_times_section = get_interval(cluster_trains,
                                               trigger_times[0] - pre,
                                               trigger_times[-1] + post)

            spike_times_zerocentered = []

            for trigger_time in trigger_times:
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
            print("WARNING: Only {} out of {} available plots can be shown."
                  "".format(n * n, len(histograms)))

        fig = Figure(figsize=(5, 5))
        canvas = FigureCanvas(fig)
        axes = fig.subplots(2, 2)
        for i in range(n):
            for j in range(n):
                if len(histograms) == 0:
                    axes[i, j].plot(0, 0)
                else:
                    cluster_label, (x, y) = histograms.popitem()
                    color = catalogueconstructor.colors[cluster_label]
                    axes[i, j].fill_between(x[:-1], 0, y, facecolor=color,
                                            edgecolor='k')
                    axes[i, j].text(0.7 * x[-1], 0.7 * ylim, cluster_label,
                                    color=color, fontsize=28)
                    axes[i, j].vlines(0, 0, ylim)
                axes[i, j].axis('off')
                axes[i, j].set_ylim(0, ylim)
                axes[i, j].set_xlim(-pre/1e6, post/1e6)

        fig.subplots_adjust(wspace=0, hspace=0)
        return canvas

    def plot_isi(self, catalogueconstructor, num_bins=100):
        """Plot ISI of spiketrains."""

        us_per_tick = int(1e6 / self.dataio.sample_rate)

        spiketrains = get_spiketrains(catalogueconstructor, us_per_tick,
                                      int(self.config['start_time'] * 1e6))

        num_clusters = len(spiketrains)

        n = 2
        if num_clusters > n * n:
            print("WARNING: Only {} out of {} available plots can be shown."
                  "".format(n * n, num_clusters))

        fig = Figure(figsize=(5, 5))
        canvas = FigureCanvas(fig)
        axes = fig.subplots(2, 2)
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
                    # When there is an extremely long ISI, the label causes
                    # the figure to be too large.
                    # axes[i, j].text(0.7 * bin_edges[-1],
                    #                 0.7 * np.max(counts),
                    #                 cluster_label, color=color, fontsize=28)
                axes[i, j].axis('off')

        fig.subplots_adjust(wspace=0, hspace=0)
        return canvas


def main():
    # Open window for data loading and electrode selection.
    tk_root = tk.Tk()
    tk_root.title("CBNU SpikeSorter")
    gui = ElectrodeSelector(tk_root)
    tk_root.protocol('WM_DELETE_WINDOW', gui.quit)
    tk_root.mainloop()


if __name__ == '__main__':
    main()
