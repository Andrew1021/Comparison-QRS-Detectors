# -*- coding: utf-8 -*-
"""
comparison_qrs_detectors
-------
A toolbox for comparison of the Pan-Tompkins, Engelse-Zeelenberg and Christov QRS detector written in Python.
:copyright: (c) 2021 by Open Innovation Lab
:license: BSD 3-clause, see LICENSE for more details.
"""

# Imports
# 3rd party
import pathlib
from pathlib import Path

# directories
file_path = pathlib.Path(__file__).resolve()
package_dir = file_path.parent
setup_dir = package_dir.parent
project_dir = package_dir.parent.parent
data_dir = package_dir.parent.parent / Path('data')
notebook_dir = package_dir.parent.parent / Path('notebooks')

# or
#import inspect, os.path
#filename = inspect.getframeinfo(inspect.currentframe()).filename
#path = os.path.dirname(os.path.abspath(filename))

# raw data paths
mit_bih = data_dir / 'raw/mit-bih-arrhythmia-database-1.0.0/mit-bih-arrhythmia-database-1.0.0'
ludb = data_dir / 'raw/lobachevsky-university-electrocardiography-database-1.0.1/lobachevsky-university-electrocardiography-database-1.0.1'
mit_raw_data_path = data_dir / 'raw/mit_bit_ecg.h5'
ludb_raw_data_path = data_dir / 'raw/ludb_ecg.h5'

# processed data paths
mit_processed_data_path = data_dir / 'processed/mit_bit_ecg.h5'
ludb_processed_data_path = data_dir / 'processed/ludb_ecg.h5'

# cleaned data paths
mit_cleand_data_path = data_dir / 'cleaned/mit_bit_ecg.h5'
ludb_cleand_data_path = data_dir / 'cleaned/ludb_ecg.h5'

# URL
mit_db_url = 'https://physionet.org/files/mitdb/1.0.0/'
ludb_url = 'https://physionet.org/files/ludb/1.0.1/'
