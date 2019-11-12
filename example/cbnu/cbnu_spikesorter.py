import os
import time
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog, messagebox

import pyqtgraph as pg

import tridesclous as tdc


class ElectrodeSelector:

    def __init__(self, root):
        self.root = root

        style = ttk.Style()
        style.theme_use('clam')  # 'winnative', 'alt', 'classic', 'default'
        self.filetypes = ['.mcd', '.msrd', '.h5']

        self.wait_window = None
        self.wait_label = None
        self.dataio = None
        self.output_path = None
        self.electrode_window = None
        self.electrodes = []
        self.geometry = None
        self.statusbar = None

        self.main_container = ttk.Frame(root)
        self.main_container.pack()
        # self.top_level_menu()
        self.dataset_button()
        self.electrode_selection_button()
        self.processing_button()
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
        """Widgets for loading the dataset."""

        # Button for loading dataset.
        ttk.Button(self.main_container,
                   text="Load dataset",
                   command=self.load_dataset).pack(fill='x',
                                                   expand=True,
                                                   side='left')

    def electrode_selection_button(self):
        """Widgets for selecting the electrodes."""

        ttk.Button(self.main_container,
                   text="Select electrodes",
                   command=self.electrode_selection_window).pack(fill='x',
                                                                 expand=True,
                                                                 side='left')

    def processing_button(self):
        """Widgets for processing data."""

        # Button for loading dataset.
        ttk.Button(self.main_container,
                   text="Start processing",
                   command=self.start_processing).pack(fill='x',
                                                       expand=True,
                                                       side='left')

    def status_widgets(self):
        from tkinter.scrolledtext import ScrolledText
        self.statusbar = ScrolledText(height=6, wrap=tk.WORD)
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
        self.wait_window.geometry(self.initial_position)
        ttk.Label(self.wait_window,
                  text="Working... Please wait.").pack()

        # Disable close button.
        self.wait_window.protocol('WM_DELETE_WINDOW', lambda: None)

    def close_wait_window(self):
        self.wait_window.destroy()
        self.wait_window.update()

    def load_dataset(self):
        """Load dataset."""

        filepath = filedialog.askopenfilename(
            title="Select dataset.", initialdir='/',
            filetypes=[('MEA files', self.filetypes), ('all files', '*.*')])

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

        self.log("TOOLTIP: Optionally, toggle electrodes to use, or start "
                 "processing all.")

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

    def electrode_selection_window(self):
        """Popup for electrode selection."""

        self.electrode_window = tk.Toplevel()
        self.electrode_window.transient(self.root)
        self.electrode_window.title("Select electrodes to process.")
        self.electrode_window.lift()
        self.electrode_window.geometry(self.initial_position)

        self.geometry = {}
        interelectrode_distance = 200  # micrometer
        num_rows = 8
        num_columns = 8
        to_skip = {(0, 0),
                   (0, num_columns - 1),
                   (num_rows - 1, 0),
                   (num_rows - 1, num_columns - 1)}
        to_disable = {(4, 0)}  # Reference electrode 15.
        init_electrodes = len(self.electrodes) == 0

        for c in range(num_columns):
            for r in range(num_rows):
                if (r, c) in to_skip:
                    continue

                if init_electrodes:
                    self.electrodes.append(tk.BooleanVar(value=True))

                i = len(self.geometry)
                if (r, c) in to_disable:
                    self.electrodes[i].set(False)

                label = "{}{}".format(c + 1, r + 1)
                ttk.Checkbutton(self.electrode_window, text=label,
                                variable=(self.electrodes[i])).grid(
                    row=r, column=c, padx=1, pady=1)

                self.geometry[i] = (r * interelectrode_distance,
                                    c * interelectrode_distance)

        half_width = num_columns // 2
        ttk.Button(self.electrode_window,
                   text="Toggle all",
                   command=self.toggle_electrode_selection).grid(
            row=num_rows, column=0, columnspan=half_width, pady=10)

        ttk.Button(self.electrode_window,
                   text="Apply selection",
                   command=self.apply_electrode_selection).grid(
            row=num_rows, column=half_width, columnspan=half_width, pady=10)

    def toggle_electrode_selection(self):

        for electrode in self.electrodes:
            electrode.set(not electrode.get())

    def apply_electrode_selection(self):

        if self.dataio is None:
            msg = "Load dataset before applying electrode selection."
            messagebox.showerror("Error", msg)
            return

        for i, electrode in enumerate(self.electrodes):
            if not electrode.get():
                self.geometry.pop(i)

        channels = list(self.geometry.keys())
        channel_groups = {0: {'channels': channels, 'geometry': self.geometry}}

        path_probe = os.path.join(self.output_path, 'electrode_selection.prb')
        with open(path_probe, 'w') as f:
            f.write("channel_groups = {}".format(channel_groups))

        self.dataio.set_probe_file(path_probe)

        self.electrode_window.destroy()
        self.root.update()

        self.log("Electrode selection applied. You can start processing now.")

    def start_processing(self):
        if self.dataio is None:
            msg = "Load dataset before starting spike sorter."
            messagebox.showerror('Error', msg)
            return

        # Apply default electrode selection if user has not changed it.
        if self.dataio.info['probe_filename'] in {'default.prb', '', None}:
            self.log("Applying default electrode selection.")
            self.electrode_selection_window()
            self.apply_electrode_selection()

        self.run_tridesclous()

    def run_tridesclous(self):
        # Parameters
        highpass_frequency = 100
        lowpass_frequency = 5000
        relative_threshold = 4
        duration = 100
        waveform_left_ms = -2
        waveform_right_ms = 3
        feature_extractor = 'pca_by_channel'
        n_components_by_channel = 4
        clustering_method = 'gmm'
        n_clusters = 3

        self.show_wait_window()
        self.log("Running spike sorter.\n"
                 "The GUI (tridesclous) will open shortly.")

        # CatalogueConstructor
        catalogueconstructor = tdc.CatalogueConstructor(self.dataio)

        catalogueconstructor.set_preprocessor_params(
            highpass_freq=highpass_frequency, lowpass_freq=lowpass_frequency,
            relative_threshold=relative_threshold)

        # Median and MAD per channel
        catalogueconstructor.estimate_signals_noise()

        # Signal preprocessing and peak detection
        catalogueconstructor.run_signalprocessor(duration=duration)

        # Extract a few waveforms
        catalogueconstructor.extract_some_waveforms(
            wf_left_ms=waveform_left_ms, wf_right_ms=waveform_right_ms)

        # Remove outlier spikes
        catalogueconstructor.clean_waveforms()

        # Feature extraction
        catalogueconstructor.extract_some_features(
            method=feature_extractor,
            n_components_by_channel=n_components_by_channel)

        # Clustering
        catalogueconstructor.find_clusters(method=clustering_method,
                                           n_clusters=n_clusters)

        self.close_wait_window()
        self.quit()

        # Visual check in CatalogueWindow
        gui = pg.mkQApp()
        win = tdc.CatalogueWindow(catalogueconstructor)
        win.show()
        gui.exec_()


def main():

    # Open window for data loading and electrode selection.
    tk_root = tk.Tk()
    tk_root.title("CBNU SpikeSorter")
    gui = ElectrodeSelector(tk_root)
    tk_root.protocol('WM_DELETE_WINDOW', gui.quit)
    tk_root.mainloop()


if __name__ == '__main__':
    main()
