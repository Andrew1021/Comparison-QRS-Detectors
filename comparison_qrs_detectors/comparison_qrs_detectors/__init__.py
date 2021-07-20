# -*- coding: utf-8 -*-
"""
comparison_qrs_detectors
-------
A toolbox for comparison of the Pan-Tompkins, Engelse-Zeelenberg and Christov QRS detector written in Python.
:copyright: (c) 2021 by Open Innovation Lab
:license: BSD 3-clause, see LICENSE for more details.
"""

# get version
from .__version__ import __version__

# compat
from __future__ import absolute_import, division, print_function, unicode_literals

# global variables allow easy loading
# path variables
from .config import mit_raw_data_path, ludb_raw_data_path, mit_processed_data_path, \
                    ludb_processed_data_path, mit_db_url, ludb_url, mit_bih, ludb

# comparison_qrs_detectors.py
from .comparison_qrs_detectors import test, test_hamilton

# download databases from website
from .download_db import download_url

# preprocess ecgs from databases
from .preprocess_mit import read_mit_bit_files, save_mit_hdf5, open_mit_hdf5, get_mit_hdf5, get_mit_keyes
from .preprocess_ludb import read_ludb_files, save_ludb_hdf5, open_ludb_hdf5, get_ludb_hdf5, get_ludb_keyes

# detectors
from .qrs_detectors import process_ecg


