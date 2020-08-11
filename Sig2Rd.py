# coding: utf-8
# Author: Red

import io
import time
import copy
import numpy as np
import xml.etree.ElementTree as ET

DEFAULT_AMPLITUDE = 1000

HDR_TMPL = """<?xml version="1.0" encoding="utf-8"?>
<sig>
    <magic>283310943</magic>
    <version>6</version>
    <timestamp>0</timestamp>
    <header_size>0</header_size>
    <hw>
        <sn>{}</sn>
        <sku>0</sku>
        <family>{}</family>
    </hw>
    <sampling>
        <start_time>{}</start_time>
        <start_time_ns>0</start_time_ns>
        <amplitude>500</amplitude>
        <sample_rate>1000</sample_rate>
        <sample_size>8</sample_size>
        <frame_size>1024</frame_size>
        <trigger_mode>0</trigger_mode>
        <channel_mode>1</channel_mode>
        <block_size>0</block_size>
        <segment>1</segment>
        <data_stamp>1</data_stamp>
        <rf>
            <if>0</if>
            <info_list>
                <rf_info>
                    <center_freq>0</center_freq>
                    <bandwidth>0</bandwidth>
                    <gain>0</gain>
                    <spec_order>0</spec_order>
                </rf_info>
            </info_list>
        </rf>
    </sampling>
    <udm>
        <udm_mode>0</udm_mode>
        <udm_data_size>32</udm_data_size>
    </udm>
    <store>
        <number_data_files>1</number_data_files>
        <data_files>
            <data_file>
                <id>0</id>
                <size>0</size>
            </data_file>
        </data_files>
    </store>
    <customer>
        <id>0</id>
        <data>22 serialization::archive 10 128 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        </data>
    </customer>
</sig>
"""


def create_element(elem, formats=None):
    assert(len(elem) > 0)

    invalid_tag = ['if']

    attrs = []
    for e in elem:
        tag = e.tag if e.tag not in invalid_tag else ('_' + e.tag)

        if len(e):
            attrs.append((tag, create_element(e)))
        else:
            def getter(self, e=e):
                try:
                    v = int(e.text) # Try convert all value to int
                except ValueError:
                    v = e.text

                if formats is not None and tag in formats:
                    return formats[tag].format(v)
                else:
                    return v

            def setter(self, v, e=e):
                e.text = str(v)

            attrs.append((tag, property(getter, setter)))

    attrs = dict(attrs)
    attrs['_elem'] = elem
    attrs['dump'] = lambda _: ET.dump(elem)
    return type("Element", (), dict(attrs))()



class Header(object):
    """
    access header as following:
        print(self.hw.sn)
        self.hw.sn = 0xfabac0000
        print(self.sampling.sample_rate)
        self.sampling.sample_rate = 1900 # Ksps
    """

    MAGIC = 0x10e2fb5f


    def __init__(self, header=None):
        if header is None:
            self.root = ET.fromstring(self.__create_default_header())
        else:
            self.root = ET.fromstring(header)


        self.hw = create_element(
                self.root.find('hw'),
                formats={'sn':'{:016X}', 'sku':'{:02X}', 'family':'{:04X}'}
                )
        self.sampling = create_element(self.root.find('sampling'))
        self.store = create_element(self.root.find('store')) # private
        self.is_setup_rfc = False


    def __create_default_header(self):
        return HDR_TMPL.format(
                0xff0000000000, # sn
                0xff00, # family
                int(time.time()) - 1381680000, # start time - offset
                )


    def __get_start_time(self):
        return self.sampling.start_time


    def __set_start_time(self, start_time):
        self.sampling.start_time = start_time


    def __get_amplitude(self):
        if not hasattr(self.sampling, 'amplitude'):
            return DEFAULT_AMPLITUDE
        return self.sampling.amplitude


    def __set_amplitude(self, amplitude):
        self.sampling.amplitude = amplitude


    def __get_sample_rate(self):
        """
        get sample rate in sps
        """
        return self.sampling.sample_rate * 1e3


    def __set_sample_rate(self, srate):
        self.sampling.sample_rate = int(srate/1e3)


    def __get_data_size(self):
        return self.store.data_files.data_file.size


    def __set_data_size(self, size):
        """
        size int bytes
        """
        self.store.data_files.data_file.size = size


    def __get_sample_size(self):
        return self.sampling.sample_size


    def __set_sample_size(self, ssize):
        self.sampling.sample_size = ssize


    def __get_channel_mode(self):
        return self.sampling.channel_mode


    def __set_channel_mode(self, cmode):
        self.sampling.channel_mode = cmode


    def __get_frame_size(self):
        return self.sampling.frame_size


    def __set_frame_size(self, fsize):
        assert((fsize&(fsize-1)) == 0)
        self.sampling.frame_size = fsize


    def __get_block_size(self):
        return self.sampling.block_size


    def __get_dtype(self):
        if self.sample_size > 16:
            return np.int32
        elif self.sample_size > 8:
            return np.int16
        else:
            return np.int8


    def __get_channels(self):
        cm = self.channel_mode
        n = 0
        for i in range(32):
            if (cm>>i)&0x1:
                n += 1
        return n


    def __set_channels(self, nchn):
        self.channel_mode = (1<<nchn) - 1


    def setup_rf(self, _if, cf, bw, gain=0, spec_order=0):
        """
        _if: IF
        cf : rf center frequency
        gain: in db
        spec_order:
        """
        self.is_setup_rfc = True
        assert(self.channels == 1) # FIXME: only support one channel right now
        self.sampling.rf._if = int(_if)
        self.sampling.rf.info_list.rf_info.center_freq = int(cf)
        self.sampling.rf.info_list.rf_info.bandwidth = int(bw)
        self.sampling.rf.info_list.rf_info.gain = int(gain*1e3)
        self.sampling.rf.info_list.rf_info.spec_order = spec_order


    def __get_block_size(self):
        return self.sampling.block_size


    def dump_rf(self):
        if not hasattr(self.sampling, 'rf'):
            return {
                    'if': 0,
                    'center_freq': 0,
                    'bandwidth': 0,
                    'gain': 0,
                    'spec_order': 0
                    }
        else:
            return {
                    'if': self.sampling.rf._if,
                    'center_freq': self.sampling.rf.info_list.rf_info.center_freq,
                    'bandwidth': self.sampling.rf.info_list.rf_info.bandwidth,
                    'gain': self.sampling.rf.info_list.rf_info.gain,
                    'spec_order': self.sampling.rf.info_list.rf_info.spec_order,
                    }


    def dump_raw(self):
        if not self.is_setup_rfc:
            for rfc_node in self.root.findall('sampling.rf'):
                self.root.remove(rfc_node)
        return ET.tostring(self.root)


    def dump(self):
        return {
                'hw': {
                    'sn': self.hw.sn,
                    'family': self.hw.family,
                    'sku': self.hw.sku
                    },
                'sampling': {
                    'start_time': self.start_time,
                    'amplitude': self.amplitude,
                    'sample_rate': self.sample_rate,
                    'sample_size': self.sample_size,
                    'channel_mode': self.channel_mode,
                    'channels': self.channels,
                    'dtype': self.dtype,
                    'block_size': self.block_size,
                    }
                }

    start_time = property(__get_start_time, __set_start_time)

    sample_rate = property(__get_sample_rate, __set_sample_rate)

    data_size = property(__get_data_size, __set_data_size)

    amplitude = property(__get_amplitude, __set_amplitude)

    sample_size = property(__get_sample_size, __set_sample_size)

    channel_mode = property(__get_channel_mode, __set_channel_mode)

    frame_size = property(__get_frame_size, __set_frame_size)

    block_size = property(__get_block_size)

    dtype = property(__get_dtype)

    channels = property(__get_channels, __set_channels)

    block_size = property(__get_block_size)


class Reader(object):
    def __init__(self, fn):
        if fn[-4:] == ".sig":
            hstr, hsize, dfn = self._sig_init(fn)
        else:
            hstr, hsize, dfn = self._rd_init(fn)

        self.header = Header(hstr)
        self.hsize = hsize
        self.dfd = open(dfn, "rb")


    def _rd_init(self, fn):
        with open(fn, 'rb') as fp:
            fp.seek(32768)
            buf = fp.read(8)
            hdr = np.frombuffer(buf, np.int32)
            return fp.read(hdr[1]), 65536, fn


    def _sig_init(self, fn):
        with open(fn, 'rb') as fp:
            return fp.read(), 0, fn[:-4] + "-0.dat" #assert only one data file


    def __len__(self):
        sbytes = np.dtype(self.header.dtype).itemsize
        return int((self.header.data_size-self.hsize)/self.header.channels/sbytes)


    def __enter__(self):
        return self


    def __exit__(self, type, value, trace):
        self.close()


    def close(self):
        self.dfd.close()


    def read(self, chn, pos, size):
        nread = 0
        if self.header.block_size == 0:
            psize = self.header.frame_size
        else:
            psize = self.header.block_size

        sbytes = np.dtype(self.header.dtype).itemsize
        pbytes = psize * sbytes

        arrs = []
        while nread < size:
            pidx = int(pos/psize)
            ppos = pos%psize

            ncpy = min(size-nread, psize-ppos)
            fpos = self.hsize + pidx*pbytes*self.channels + pbytes*chn + ppos*sbytes
            self.dfd.seek(fpos)
            arrs.append(np.fromfile(self.dfd, self.header.dtype, count = ncpy))
            nread += ncpy
            pos += ncpy

        arr = np.hstack(arrs).flatten()
        return arr


    @property
    def channels(self):
        return self.header.channels


    @property
    def dtype(self):
        return self.header.dtype


class Writer(object):
    def __init__(self, fn):
        if fn[-3:] != '.rd':
            fn += '.rd'

        self.fn = fn
        self.header = Header()
        self.hsize = 65536
        self.dfd = open(fn, "wb")
        self.dfd.write(np.zeros(65536, dtype=np.int8).tobytes())


    def __enter__(self):
        return self


    def __exit__(self, type, value, trace):
        self.close()


    def __len__(self):
        sbytes = np.dtype(self.header.dtype).itemsize
        return int(self.header.data_size/self.header.channels/sbytes)


    def write_header(self):
        self.dfd.seek(32768)

        xml = self.header.dump_raw()

        hdr = np.array([Header.MAGIC, len(xml)], dtype=np.int32)
        self.dfd.write(hdr.tobytes())
        self.dfd.write(xml)


    def write(self, data):
        if data.dtype is self.header.dtype:
            raise Exception("Invalid data type")

        if not ((data.ndim == 1 and self.header.channels == 1) \
                or (data.ndim == 2 and data.shape[0] == self.header.channels)):
            raise Exception("Need {} channels data".format(self.header.channels))

        if self.header.channels > 1 and data.shape[1] != self.header.frame_size:
            raise Exception("Invalid frame size, {} != {}".format(
                self.header.frame_size, data.shape[1]))

        self.dfd.seek(0, io.SEEK_END)
        self.dfd.write(data.tobytes())
        self.header.data_size = self.dfd.tell()


    def close(self):
        self.write_header()
        self.dfd.close()



if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        raise RuntimeError("Sig file path and save rd file path must be provided")

    sig_path = sys.argv[1]
    rd_path = sys.argv[2]
    if not sig_path.endswith('.sig'):
        raise RuntimeError("Data file must be .sig file.")    

    sig_data_path = sig_path[:-4] + "-0.dat"
    print(sig_data_path)
    with Writer(rd_path) as writer:
        with Reader(sig_path) as reader:
            writer.header = reader.header

        file_r = open(sig_data_path,"rb")
        file_w = open(rd_path,"wb")
        file_w.seek(64*1024)

        while True:
            data = file_r.read(1024*1024)
            if len(data) == 0:
                break
            file_w.write(data)

        writer.header.data_size = file_w.tell()


