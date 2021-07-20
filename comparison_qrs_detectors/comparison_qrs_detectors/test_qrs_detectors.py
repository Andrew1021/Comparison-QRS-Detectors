# -*- coding: utf-8 -*-
"""
comparison_qrs_detectors
-------
A toolbox for comparison of the Pan-Tompkins, Engelse-Zeelenberg and Christov QRS detector written in Python.
:copyright: (c) 2021 by Open Innovation Lab
:license: BSD 3-clause, see LICENSE for more details.
"""
# Test database download function file
from . import read_mit_bit_files
from . import read_ludb_files
from config import ludb
import numpy as np
from biosppy.signals import ecg

def test_qrs_detection():
    """Main function - print quick test example
    """
    try:
        print("Example MIT-BIT Quick Test")
        # Quick test
        # load ECG data from ludb database into pandas dataframes
        ecg, rpeaks = read_mit_bit_files()
        print(ecg)
        # load raw ECG signal
        signal = np.loadcsv(ludb / 'ludb.csv')
        # process it and plot
        out = ecg.ecg(signal=signal, sampling_rate=500., show=True)
        return 0
    except:
        return 1

if __name__ == "__Test_QRS__":

    mit_bit_test = test_qrs_detection()

    print("MIT-BIT download test executed with {0}\n."
          "LUDB download test executed with {1}".format(mit_bit_test, ludb_test))
    if (mit_bit_test & ludb_test):
        print("All tests executed well!")
    elif(mit_bit_test):
        print("MIT-BIT download test failed!")
    elif(ludb_test):
        print("LUDB download test failed!")
