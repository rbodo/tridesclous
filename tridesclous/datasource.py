import os
import subprocess

import numpy as np
import re
from collections import OrderedDict


data_source_classes = OrderedDict()

possible_modes = ['one-file', 'multi-file', 'one-dir', 'multi-dir', 'other']


import neo



class DataSourceBase:
    gui_params = None
    def __init__(self):
        # important total_channel != nb_channel because nb_channel=len(channels)
        self.total_channel = None 
        self.sample_rate = None
        self.nb_segment = None
        self.dtype = None
        self.bit_to_microVolt = None
    
    def get_segment_shape(self, seg_num):
        raise NotImplementedError
    
    def get_channel_names(self):
        raise NotImplementedError
    
    def get_signals_chunk(self, seg_num=0, i_start=None, i_stop=None):
        raise NotImplementedError

    def load(self):
        pass


def get_channel_names(stream):
    channel_names_of_file = -np.ones(len(stream.channel_infos), int)
    ref_label = '15'
    for info in stream.channel_infos.values():
        channel_id = info.channel_id
        label = stream.channel_infos[channel_id].label
        if label == 'Ref':
            all_labels = [info.label for info in stream.channel_infos.values()]
            assert ref_label not in all_labels, \
                "Reference electrode was assumed to be {}. But this label " \
                "is already in use.".format(ref_label)
            label = ref_label
        channel_names_of_file[channel_id] = int(label)
    assert np.all(channel_names_of_file > 0)
    return ['ch{}'.format(i) for i in channel_names_of_file]


def get_all_channel_data(stream):
    num_channels, num_timesteps = stream.channel_data.shape
    # Data type is fixed to float because ``stream.get_channel_in_range``
    # returns traces in Volt. Otherwise, could use
    # ``stream.channel_data.dtype``, which would be 'int32'.
    dtype = 'float32'

    id_map = {}
    scales = []
    offsets = []
    sample_rates = []
    for k in stream.channel_infos.keys():
        info = stream.channel_infos[k]
        id_map[info.channel_id] = info.row_index
        scales.append(info.adc_step.magnitude)
        offsets.append(info.get_field('ADZero'))
        sample_rates.append(int(info.sampling_frequency.magnitude))

    assert np.array_equiv(sample_rates, sample_rates[0]), \
        "Recording contains different sample rates."

    is_parallelizable = (np.array_equiv(scales, scales[0]) and
                         np.array_equiv(offsets, offsets[0]))

    if is_parallelizable:
        channel_data_permuted = np.array(stream.channel_data)
        channel_data = np.empty_like(channel_data_permuted)
        channel_data[list(id_map.keys())] = \
            channel_data_permuted[list(id_map.values())]
        channel_data = np.transpose(channel_data)
        channel_data = (channel_data - offsets[0]) * scales[0]
    else:
        import sys
        channel_data = np.empty((num_timesteps, num_channels), dtype)
        for i_channel in range(num_channels):
            channel_data[:, i_channel] = stream.get_channel_in_range(
                i_channel, 0, num_timesteps - 1)[0]
            status = (i_channel + 1) / num_channels
            sys.stdout.write('\r{:>7.2%}'.format(status))
            sys.stdout.flush()
        print('')

    return channel_data, sample_rates[0]


class H5DataSource(DataSourceBase):
    """DataSource from h5 files."""

    mode = 'multi-file'

    def __init__(self, filenames, gui=None):
        DataSourceBase.__init__(self)

        self.log = print if gui is None else gui.log

        self.filenames = [filenames] if isinstance(filenames, str) \
            else filenames
        for filename in self.filenames:
            assert os.path.exists(filename), \
                "File {} could not be found.".format(filename)

        self.nb_segment = len(self.filenames)
        self.array_sources = []
        self.channel_names = None
        self.bit_to_microVolt = 1  # Signal is already in uV.

    def load(self):
        from McsPy.McsData import RawData

        sample_rates = []
        dtypes = []
        num_channels = []
        channel_names = []
        for filename in self.filenames:
            self.log("Loading .h5 file: {}".format(filename))
            data = RawData(filename)
            assert len(data.recordings) == 1, \
                "Can only handle a single recording per file."

            electrode_data = None
            stream_id = None
            analog_streams = data.recordings[0].analog_streams
            for stream_id, stream in analog_streams.items():
                if stream.data_subtype == 'Electrode':
                    electrode_data = stream
                    break
            assert electrode_data is not None, "Electrode data not found."

            traces, sample_rate = get_all_channel_data(electrode_data)
            self.array_sources.append(traces)
            sample_rates.append(sample_rate)
            dtypes.append(traces.dtype)
            num_channels.append(traces.shape[1])
            channel_names_of_file = get_channel_names(electrode_data)
            channel_names.append(channel_names_of_file)

            trigger_times = None
            event_streams = data.recordings[0].event_streams
            if event_streams is not None:
                if len(event_streams) > 1:
                    print("WARNING: Multiple stimulus streams detected!")
                for d in event_streams[0].event_entity.values():
                    if d.info.label in {'STG 1 Single Pulse Start',
                                        'Digital Event Detector Event'}:
                        trigger_times = d.data[0]
                        break
            else:
                analog_stream_id = (stream_id + 1) % 2
                trigger_data = analog_streams[analog_stream_id].channel_data[0]
                trigger_times = np.flatnonzero(np.diff(trigger_data) >
                                               np.abs(np.min(trigger_data)))
                us_per_tick = int(1e6 / sample_rate)
                trigger_times *= us_per_tick

            assert trigger_times is not None, "Trigger data not found."

            dirname, basename = os.path.split(filename)
            basename, _ = os.path.splitext(basename)
            np.savez_compressed(os.path.join(
                dirname, basename + '_trigger'), trigger_times)

        # Make sure that every file uses the same sample rate, dtype, etc.
        assert np.array_equiv(sample_rates, sample_rates[0]), \
            "Recording contains different sample rates."

        assert np.array_equiv(dtypes, dtypes[0]), \
            "Recording contains different dtypes."

        assert np.array_equiv(num_channels, num_channels[0]), \
            "Recording contains different number of channels."

        assert np.array_equiv(channel_names, channel_names[0]), \
            "Recording contains different channel names."

        self.total_channel = num_channels[0]
        self.sample_rate = sample_rates[0]
        self.dtype = dtypes[0]
        self.channel_names = channel_names[0]

        self.log("Finished initializing DataSource.")

    def get_segment_shape(self, seg_num):
        return self.array_sources[seg_num].shape

    def get_signals_chunk(self, seg_num=0, i_start=None, i_stop=None):
        return self.array_sources[seg_num][i_start:i_stop, :]

    def get_channel_names(self):
        from McsPy.McsData import RawData

        if self.channel_names is None:
            channel_names = []
            for filename in self.filenames:
                data = RawData(filename)
                stream = data.recordings[0].analog_streams[0]
                channel_names.append(get_channel_names(stream))
            assert np.array_equiv(channel_names, channel_names[0]), \
                "Recording contains different channel names."
            self.channel_names = channel_names[0]

        return self.channel_names


class MsrdDataSource(H5DataSource):
    """DataSource from MCS2100."""

    def __init__(self, filenames, gui=None):

        log = print if gui is None else gui.log

        if isinstance(filenames, str):
            filenames = [filenames]

        filenames_h5 = []
        for filename in filenames:
            assert os.path.exists(filename), \
                "File {} could not be found.".format(filename)
            msg = "Converting file from .msrd to .h5: {}".format(filename)
            log(msg)
            basename = os.path.splitext(filename)[0]
            filenames_h5.append(basename + '.h5')
            subprocess.run(["MCDataConv", "-t", "hdf5", basename + '.msrs'])
        log("Done converting.")

        H5DataSource.__init__(self, filenames_h5, gui)


class McdDataSource(H5DataSource):
    """DataSource from MCS1060."""

    def __init__(self, filenames, gui=None):

        log = print if gui is None else gui.log

        if isinstance(filenames, str):
            filenames = [filenames]

        filenames_h5 = []
        for filename in filenames:
            assert os.path.exists(filename), \
                "File {} could not be found.".format(filename)
            msg = "Converting file from .mcd to .h5: {}".format(filename)
            log(msg)
            filenames_h5.append(os.path.splitext(filename)[0] + '.h5')
            subprocess.run(["MCDataConv", "-t", "hdf5", filename])
        log("Done converting.")

        H5DataSource.__init__(self, filenames_h5, gui)


class InMemoryDataSource(DataSourceBase):
    """
    DataSource in memory numpy array.
    This is for debugging  or fast testing.
    """
    mode = 'other'
    def __init__(self, nparrays=[], sample_rate=None):
        DataSourceBase.__init__(self)
        
        self.nparrays = nparrays
        self.nb_segment = len(self.nparrays)
        self.total_channel = self.nparrays[0].shape[1]
        self.sample_rate = sample_rate
        self.dtype = self.nparrays[0].dtype
    
    def get_segment_shape(self, seg_num):
        full_shape = self.nparrays[seg_num].shape
        return full_shape
    
    def get_signals_chunk(self, seg_num=0, i_start=None, i_stop=None):
            data = self.nparrays[seg_num][i_start:i_stop, :]
            return data
            
    def get_channel_names(self):
        return ['ch{}'.format(i) for i in range(self.total_channel)]

data_source_classes['InMemory'] = InMemoryDataSource





class RawDataSource(DataSourceBase):
    """
    DataSource from raw binary file. Easy case.
    """
    mode = 'multi-file'
    gui_params = [
        {'name': 'dtype', 'type': 'list', 'values':['int16', 'uint16', 'float32', 'float64']},
        {'name': 'total_channel', 'type': 'int', 'value':1},
        {'name': 'sample_rate', 'type': 'float', 'value':10000., 'step': 1000., 'suffix': 'Hz', 'siPrefix': True},
        {'name': 'offset', 'type': 'int', 'value':0},
    ]
    
    def __init__(self, filenames=[], dtype='int16', total_channel=0,
                        sample_rate=0., offset=0, bit_to_microVolt=None, channel_names=None):
        DataSourceBase.__init__(self)
        
        self.filenames = filenames
        if isinstance(self.filenames, str):
            self.filenames = [self.filenames]
        assert all([os.path.exists(f) for f in self.filenames]), 'files does not exist'
        self.nb_segment = len(self.filenames)

        self.total_channel = total_channel
        self.sample_rate = sample_rate
        self.dtype = np.dtype(dtype)
        
        if bit_to_microVolt == 0.:
            bit_to_microVolt = None
        self.bit_to_microVolt = bit_to_microVolt
        
        if channel_names is None:
            channel_names = ['ch{}'.format(i) for i in range(self.total_channel)]
        self.channel_names = channel_names
        

        self.array_sources = []
        for filename in self.filenames:
            data = np.memmap(filename, dtype=self.dtype, mode='r', offset=offset)
            #~ data = data[:-(data.size%self.total_channel)]
            data = data.reshape(-1, self.total_channel)
            self.array_sources.append(data)
    
    def get_segment_shape(self, seg_num):
        full_shape = self.array_sources[seg_num].shape
        return full_shape
    
    def get_signals_chunk(self, seg_num=0, i_start=None, i_stop=None):
            data = self.array_sources[seg_num][i_start:i_stop, :]
            return data

    def get_channel_names(self):
        return self.channel_names

data_source_classes['RawData'] = RawDataSource






import neo.rawio

io_gui_params = {
    'RawBinarySignal':[
                {'name': 'dtype', 'type': 'list', 'values':['int16', 'uint16', 'float32', 'float64']},
                {'name': 'nb_channel', 'type': 'int', 'value':1},
                {'name': 'sampling_rate', 'type': 'float', 'value':10000., 'step': 1000., 'suffix': 'Hz', 'siPrefix': True},
                {'name': 'bytesoffset', 'type': 'int', 'value':0},
    ],
}


# hook for some neo.rawio that have problem with TDC (multi sampling rate or default params)
neo_rawio_hooks = {}

class Intan(neo.rawio.IntanRawIO):
    def _parse_header(self):
        neo.rawio.IntanRawIO._parse_header(self)
        sig_channels = self.header['signal_channels']
        sig_channels = sig_channels[sig_channels['group_id']==0]
        self.header['signal_channels'] = sig_channels

    def _get_signal_size(self, block_index, seg_index, channel_indexes):
        if channel_indexes is None:
            channel_indexes = slice(None)
        assert np.unique(self.header['signal_channels'][channel_indexes]['group_id']).size == 1
        channel_names = self.header['signal_channels'][channel_indexes]['name']
        chan_name = channel_names[0]
        size = self._raw_data[chan_name].size
        return size

neo_rawio_hooks['Intan'] = Intan



class NeoRawIOAggregator(DataSourceBase):
    """
    wrappe and agregate several neo.rawio in the class.
    """
    gui_params = None
    rawio_class = None
    def __init__(self, **kargs):
        DataSourceBase.__init__(self)
        
        self.rawios = []
        if  'filenames' in kargs:
            filenames= kargs.pop('filenames') 
            self.rawios = [self.rawio_class(filename=f, **kargs) for f in filenames]
        elif 'dirnames' in kargs:
            dirnames= kargs.pop('dirnames') 
            self.rawios = [self.rawio_class(dirname=d, **kargs) for d in dirnames]
        else:
            raise(ValueError('Must have filenames or dirnames'))
            
        
        self.sample_rate = None
        self.total_channel = None
        self.sig_channels = None
        nb_seg = 0
        self.segments = {}
        for rawio in self.rawios:
            rawio.parse_header()
            assert not rawio._several_channel_groups, 'several sample rate for signals'
            assert rawio.block_count() ==1, 'Multi block RawIO not implemented'
            for s in range(rawio.segment_count(0)):
                #nb_seg = absolut seg index and s= local seg index
                self.segments[nb_seg] = (rawio, s)
                nb_seg += 1
            
            if self.sample_rate is None:
                self.sample_rate = rawio.get_signal_sampling_rate()
            else:
                assert self.sample_rate == rawio.get_signal_sampling_rate(), 'bad joke different sample rate!!'
            
            sig_channels = rawio.header['signal_channels']
            if self.sig_channels is None:
                self.sig_channels = sig_channels
                self.total_channel = len(sig_channels)
            else:
                assert np.all(sig_channels==self.sig_channels), 'bad joke different channels!'
            
        self.nb_segment = len(self.segments)
        
        self.dtype = np.dtype(self.sig_channels['dtype'][0])
        units = sig_channels['units'][0]
        #~ assert 'V' in units, 'Units are not V, mV or uV'
        if units =='V':
            self.bit_to_microVolt = self.sig_channels['gain'][0]*1e-6
        elif units =='mV':
            self.bit_to_microVolt = self.sig_channels['gain'][0]*1e-3
        elif units =='uV':
            self.bit_to_microVolt = self.sig_channels['gain'][0]
        else:
            self.bit_to_microVolt = None
        
    def get_segment_shape(self, seg_num):
        rawio, s = self.segments[seg_num]
        l = rawio.get_signal_size(0, s)
        return l, self.total_channel
    
    def get_channel_names(self):
        return self.sig_channels['name'].tolist()
    
    def get_signals_chunk(self, seg_num=0, i_start=None, i_stop=None):
        rawio, s = self.segments[seg_num]
        return rawio.get_analogsignal_chunk(block_index=0, seg_index=s, 
                        i_start=i_start, i_stop=i_stop)

#Construct the list with taking local class with hooks dict
rawiolist = []
for rawio_class in neo.rawio.rawiolist:
    name = rawio_class.__name__.replace('RawIO', '')
    if name in neo_rawio_hooks:
        rawio_class = neo_rawio_hooks[name]
    rawiolist.append(rawio_class)

if neo.rawio.RawBinarySignalRawIO in rawiolist:
    # to avoid bug in readthe doc with moc
    RawBinarySignalRawIO = rawiolist.pop(rawiolist.index(neo.rawio.RawBinarySignalRawIO))
#~ rawiolist.insert(0, RawBinarySignalRawIO)

for rawio_class in rawiolist:
    name = rawio_class.__name__.replace('RawIO', '')
    class_name = name+'DataSource'
    datasource_class = type(class_name,(NeoRawIOAggregator,), { })
    datasource_class.rawio_class = rawio_class
    if rawio_class.rawmode in ('multi-file', 'one-file'):
        #multi file in neo have another meaning
        datasource_class.mode = 'multi-file'
    elif rawio_class.rawmode in ('one-dir', ):
        datasource_class.mode = 'multi-dir'
    else:
        continue
    
    #gui stuffs
    if name in io_gui_params:
        datasource_class.gui_params = io_gui_params[name]
        
    data_source_classes[name] = datasource_class
    #~ print(datasource_class, datasource_class.mode )

data_source_classes['mcd'] = McdDataSource
data_source_classes['msrd'] = MsrdDataSource
data_source_classes['h5'] = H5DataSource
    
#TODO implement KWIK and OpenEphys
#https://open-ephys.atlassian.net/wiki/display/OEW/Data+format
# https://github.com/open-ephys/analysis-tools/tree/master/Python3


