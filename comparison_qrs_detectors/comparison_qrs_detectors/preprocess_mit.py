# -*- coding: utf-8 -*-
"""
comparison_qrs_detectors.preprocess_mit
---------------
This module provides several data preprocess methods for the MIT-BIH Arrhythmia Database.

Credits:
    https://github.com/berndporr/py-ecg-detectors/blob/master/tester_MITDB.py by Bernd Porr

:copyright: (c) 2021 by Open Innovation Lab
:license: BSD 3-clause, see LICENSE for more details.
"""

# Imports
# built-in
import math
import time
import os

# 3rd party
import h5py
import pandas as pd
from biosppy import utils
from pandas import HDFStore, DataFrame, read_hdf  # create (or open) an hdf5 file and opens in append mode
import numpy as np
import wfdb
import progressbar


# local
from config import *

def get_data_anno(file, participant):
    """Utility function - read from database
    """
    # Get signal
    data = pd.DataFrame({"ECG": wfdb.rdsamp(file[:-4], channels=[0])[0][:, 0]}, dtype=np.float16)

    data["Participant"] = "MIT-Arrhythmia_%.2i" %(participant)
    data["Sample"] = range(len(data))
    data["Sampling_Rate"] = 360
    data["Database"] = "MIT-Arrhythmia-x" if "x_mitdb" in file else "MIT-Arrhythmia"
    #data = data.convert_dtypes(convert_string = False)
    data = data.astype({'ECG': 'float32', 'Sample': 'int32', 'Sampling_Rate': 'int16'})

    # getting annotations
    anno = wfdb.rdann(file[:-4], 'atr')
    #print(anno.symbol)
    # check if there are any other than the normal reference beat
    #if not (set(anno.symbol) == {'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r', 'F', 'e', 'j', 'n', 'E', '/', 'f', 'Q', '?'}):
    #    assert 1 == 0, f'Element {participant} (Patient {data["Participant"][0]}) has abnormal beats.'

    # check whether all beats have a start or and end
    last_checked_spl = 0
    for counter in range(len(anno.symbol)):
        if counter == 0:
            assert anno.symbol[0] == '+' or "~", 'There is no correct beginning.'
        if counter > 0:
            # Beat annotations and non-beat annotations:
            assert anno.symbol[counter] in ['N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r', 'F', 'e', 'j', 'n', 'E', '/', 'f', 'Q', '?'] or \
                   ['[', '!', ']', 'x', '(', ')', 'p', 't', 'u', '`', "'", '^', '|', '~', '+', 's', 't', '*', 'D', '=','"', '@'], \
                f'There is no correct (non-)beat type ({anno.symbol[counter]}).'

        if counter == len(anno.symbol)-1:
            assert anno.symbol[counter] in ['N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r', 'F', 'e', 'j', 'n', 'E', '/', 'f', 'Q', '?'] or \
                   ['[', '!', ']', 'x', '(', ')', 'p', 't', 'u', '`', "'", '^', '|', '~', '+', 's', 't', '*', 'D', '=', '"', '@'], 'There is no correct ending.'

        if anno.symbol[counter] in ['t' 'p' 'u']:
            print('In patient {0} there is ({1})-peak annotation.'.format(data["Participant"][0], anno.symbol[counter]))

        # check if we have any timing problems
        #if anno.sample[counter] > last_checked_spl:
        #    print('There are timing problems in {0} because of {1}. Exclude these record.'.format(data["Participant"][0], anno.sample[counter]))


    #anno = np.unique(anno.sample[np.in1d(anno.symbol, ['N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r', 'F', 'e', 'j', 'n', 'E', '/', 'f', 'Q', '?'])])
    t = np.unique(anno.sample[np.in1d(anno.symbol, ['t'])])
    N = np.unique(anno.sample[np.in1d(anno.symbol, ['N'])])
    p = np.unique(anno.sample[np.in1d(anno.symbol, ['p'])])
    if t.size == 0:
        t = np.full_like(N, np.NaN)
    if p.size == 0:
        p = np.full_like(N, np.NaN)
    # u = np.unique(anno.sample[np.in1d(anno.symbol, ['u'])])
    anno = pd.DataFrame({"Tpeaks": t, "Rpeaks": N, "Ppeaks": p})
    #anno = pd.DataFrame([np.transpose(t), np.transpose(N), np.transpose(p)], columns=["Tpeaks", "Rpeaks", "Ppeaks"])
    anno["Participant"] = "MIT-Arrhythmia_%.2i" %(participant)
    anno["Sampling_Rate"] = 360
    anno["Database"] = "MIT-Arrhythmia-x" if "x_mitdb" in file else "MIT-Arrhythmia"
    #anno = anno.convert_dtypes(convert_string = False)
    anno = anno.astype({'Tpeaks': 'int32', 'Rpeaks': 'int32', 'Sampling_Rate': 'int16'})

    return data, anno


def read_mit_bit_files():
    """
    Processes the data from the MIT-BIT database and save it into pandas dataframe.

    Intended to be used under this particular circumstance,
    with :func:`download_url <comparison_qrs_detectors.comparison_qrs_detectors.download_db.download_url>` called before it, and potentially {to be determined}
    called after it, but optional.

    :returns: A modified pandas dataframe.
    """

    # Get signal
    dfs_ecg = []
    dfs_rpeaks = []

    raw = data_dir / 'raw' / 'mit-bih-arrhythmia-database-1.0.0/mit-bih-arrhythmia-database-1.0.0'
    absolut = 'C:/Users/Andreas/PycharmProjects/Comparison-QRS-Detectors/data/raw/mit-bih-arrhythmia-database-1.0.0/mit-bih-arrhythmia-database-1.0.0/'
    data_files = [absolut + file for file in os.listdir(raw) if ".dat" in file]
    bar = progressbar.ProgressBar(maxval=len(data_files)).start()

    for participant, file in enumerate(data_files):

        #print("Participant: " + str(participant + 1) + "/" + str(len(data_files)))

        data, anno = get_data_anno(file, participant)

        # Store with the rest
        dfs_ecg.append(data)
        dfs_rpeaks.append(anno)

        # Store additional recording if available
        if "x_" + file.replace(raw.as_posix(), "") in os.listdir(raw / "x_mitdb/"):
            print("  - Additional recording detected.")
            data, anno = get_data_anno(raw.as_posix() + "/x_mitdb/" + "x_" + file.replace(raw.as_posix(), ""), participant)
            # Store with the rest
            dfs_ecg.append(data)
            dfs_rpeaks.append(anno)

        bar.update(participant+1)

    # Concatenate each data and annotations to one pandas dataframe
    df_ecg = pd.concat(dfs_ecg)
    dfs_rpeaks = pd.concat(dfs_rpeaks)

    # Annotate metadata
    metadata = {'User': 'Andreas Roth',
                'Date': time.time(),
                'OS': os.name,
                'Database': 'MIT-BIH Arrhythmia Database (MIT-BIT)',
                'Suspects': '47',
                'Record': '48',
                'Manual annotation': 'Yes',
                'Organization': 'Beth Israel Hospital Arrhythmia Laboratory',
                'Leads': 'i, ii, iii, avr, avl, avf, v1, v2, v3, v4, v5, v6',
                'Duration [min]': '30',
                'Sampling Frequency': '360',
                'Waves': '100000',
                'QRS': '21966',
                'Patients': 'inpatients and outpatients',
                'Age': '23-89',
                'Gender': '22 women, 25 men',
                'Licence': 'MIT License'}

    print("\n" + str(participant + 1) + "/" + str(len(data_files)) + " Participant successful loaded!")

    # Save to csv file
    #df_ecg.to_csv(data_dir / processed / "MIT_ECGs.csv", index=False)
    #dfs_rpeaks.to_csv(data_dir / processed / "MIT_Rpeaks.csv", index=False)

    return df_ecg, dfs_rpeaks, metadata

def save_mit_hdf5(df_ecg, dfs_rpeaks, name=mit_raw_data_path):
    # Save as hdf5 file
    # We can create a HDF5 file using the HDFStore class provided by Pandas
    # normalize path
    path = utils.normpath(name.as_posix())
    print("\nSaving data to hdf5 file...")
    with pd.HDFStore(path, mode='w') as hdf:
        #hdf = pd.HDFStore('ecg.h5')

        # Now we can store a dataset into the file we just created:
        # df = DataFrame(np.random.rand(5, 3), columns=('A', 'B', 'C'))  # put the dataset in the storage
        # Group for formatted ECG database
        # Sub-Group for each ECG data and annotations and dataframe for samples and R-Peak annotations
        hdf.put('ECG/Data', value=df_ecg, format='table', data_columns=True, complevel=9, complib='blosc')
        hdf.put('ECG/Rpeaks', value=dfs_rpeaks, format='table', data_columns=True, complevel=9, complib='blosc')
        #df_ecg.to_hdf(hdf, 'ECG/Data', mode="w", value=df_ecg, format='table', data_columns=True, complevel=9, complib='blosc')
        #dfs_rpeaks.to_hdf(hdf, 'ECG/Rpeaks', mode="a", value=dfs_rpeaks, format='table', data_columns=True, complevel=9, complib='blosc')

        # Annotate metadata to hdf5 file
        metadata = {'User': 'Andreas Roth',
                    'Date': time.time(),
                    'OS': os.name,
                    'Database': 'MIT-BIH Arrhythmia Database (MIT-BIT)',
                    'Suspects': '47',
                    'Record': '48',
                    'Manual annotation': 'Yes',
                    'Organization': 'Beth Israel Hospital Arrhythmia Laboratory',
                    'Leads': 'i, ii, iii, avr, avl, avf, v1, v2, v3, v4, v5, v6',
                    'Duration [min]': '30',
                    'Sampling Frequency': '360',
                    'Waves': '100000',
                    'QRS': '21966',
                    'Patients': 'inpatients and outpatients',
                    'Age': '23-89',
                    'Gender': '22 women, 25 men',
                    'Licence': 'MIT License'}
        hdf.get_storer('ECG/Data').attrs.metadata = metadata

        #hdf.close()
        #print('Saved to hdf5 file {0}.'.format(str(path)))

        # The structure used to represent the hdf file in Python is a dictionary and we can access to our data using
        # the name of the dataset as key:
        #print(hdf['ECG/Data'].shape)

        # show structure of hdf5 storage
        #print(hdf)

def open_mit_hdf5(name=mit_raw_data_path):
    # normalize path
    path = utils.normpath(name.as_posix())
    # Open hdf5 file
    print("\nLoad data from hdf5 file...")
    with pd.HDFStore(path, mode='r') as hdf:
        ecg_dset = hdf.get('ECG/Data')
        rpeaks_dset = hdf.get('ECG/Rpeaks')
        #ecg_dset = hdf['ECG/Data']  # Dataset pointed to ssd memory, gone when hf closed
        ecg = ecg_dset  # numpy/pandas array
        #rpeaks_dset = hdf['ECG/Rpeaks']  # Dataset pointed to ssd memory, gone when hf closed
        rpeaks = rpeaks_dset  # numpy/pandas array
        metadata_dset = hdf.get_storer('ECG/Data').attrs.metadata
        metadata = metadata_dset
    print('Open hdf5 file {0}.'.format(path))
    return ecg, rpeaks, metadata

def get_mit_hdf5(name=mit_raw_data_path):
    # this query selects the ecg dataframe
    data = read_hdf(name, 'ECG/Data')
    anno = read_hdf(name, 'ECG/Data')
    ecg = data
    rpeaks = anno
    data.close()  # closes the files
    anno.close()  # closes the files
    return ecg, rpeaks
