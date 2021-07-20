# -*- coding: utf-8 -*-
"""
comparison_qrs_detectors
-------
A toolbox for comparison of the Pan-Tompkins, Engelse-Zeelenberg and Christov QRS detector written in Python.
:copyright: (c) 2021 by Open Innovation Lab
:license: BSD 3-clause, see LICENSE for more details.
"""
# Test config.py function file
from download_db import download_url
from config import mit_db_url, mit_bih, ludb_url, ludb
import numpy as np
from biosppy.signals import ecg


def test_download(test_url, test_root):
    """test function - print quick test example
    """
    try:
        # filename
        download_url(test_url, test_root, filename=None)
        return 0
    except:
        return 1


#if __name__ == "__Download_Test__":

# Test MIT-BIH download
url = mit_db_url
root = mit_bih
mit_bih_test = test_download(url, root)

# Test LUDB download
url = ludb_url
root = ludb
ludb_test = test_download(ludb_url, ludb)

print("MIT-BIH download test executed with {0}\n"
      "LUDB download test executed with {1}".format(mit_bih_test, ludb_test))
if (not (mit_bih_test and ludb_test)):
    print("All tests executed well!")
elif (not mit_bih_test):
    print("MIT-BIH download test failed!")
elif (not ludb_test):
    print("LUDB download test failed!")
