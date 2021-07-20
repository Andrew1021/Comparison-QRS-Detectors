# -*- coding: utf-8 -*-
"""
comparison_qrs_detectors.comparison_qrs_detectors
---------------
A toolbox for comparison of the Pan-Tompkins, Engelse-Zeelenberg and Christov QRS detector written in Python.
:copyright: (c) 2021 by Open Innovation Lab
---------------
This module provides several methods for the Pan-Tompkins, Engelse-Zeelenberg and Christov QRS detectors.

:license: BSD 3-clause, see LICENSE for more details.
"""

# Imports
# 3rd party
import h5py
import numpy as np
import pylab as pl
from biosppy.signals import ecg

# local
from . import read_mit_bit_files


def test():
    """Main function - print quick test example
    """
    print("Example Quick Test")
    # Quick test
    # download ECG data from mit-bit database
    read_mit_bit_files()
    # load raw ECG signal
    signal = np.loadcsv('.data/ECGs.csv')

    # process it and plot
    out = ecg.ecg(signal=signal, sampling_rate=1000., show=True)

def test_hamilton():
    """Test Hamilton algorithm - print results
    """
    # load raw ECG signal
    # Open hdf5 file
    name = '../../data/processed/mit_bit_ecg.hdf5'
    with h5py.File(name, 'r') as hf:
        ecg_dset = hf['ECG/Data/Samples']  # Dataset pointed to ssd memory, gone when hf closed
        ecg = ecg_dset  # numpy/pandas array
        rpeaks_dset = hf['ECG/Annotations/Rpeaks']  # Dataset pointed to ssd memory, gone when hf closed
        rpeaks = rpeaks_dset  # numpy/pandas array
        sf_dset = hf.attrs['Sampling Frequency']
        sampling_frequency = sf_dset    # or fs
    print('Opend hdf5 file {0} with status {1}'.format(name, hf.status_code))

    # Plot the raw ECG signal for record 1 of MIT-BIH Arrhythmia Database
    N = len(ecg[0])  # number of samples
    T = (N - 1) / sampling_frequency  # duration
    ts = np.linspace(0, T, N, endpoint=False)  # relative timestamps
    pl.plot(ts, ecg[0], lw=2)
    pl.grid()
    pl.show()

    # Process raw ECG signal and plot results
    out = ecg.ecg(signal=ecg, sampling_rate=sampling_frequency, show=True)


if __name__ == "__main__":
    test()
