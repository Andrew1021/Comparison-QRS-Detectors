# -*- coding: utf-8 -*-
"""
comparison_qrs_detectors.preprocess_ludb
---------------
This module provides several data preprocess methods for the Lobachevsky University Electrocardiography Database.

Credits:
    https://github.com/neuropsychology/NeuroKit/blob/dev/data/ludb/download_ludb.py by Dominique Makowski

:copyright: (c) 2021 by Open Innovation Lab
:license: BSD 3-clause, see LICENSE for more details.

Script for formatting the Lobachevsky University Electrocardiography Database
The database consists of 200 10-second 12-lead ECG signal records representing different morphologies of the ECG signal. The ECGs were collected from healthy volunteers and patients, which had various cardiovascular diseases. The boundaries of P, T waves and QRS complexes were manually annotated by cardiologists for all 200 records.
Steps:
    1. In the command line, run 'pip install gsutil'
    2. Then, 'gsutil -m cp -r gs://ludb-1.0.0.physionet.org D:/YOURPATH/NeuroKit/data/ludb'
    This will download all the files in a folder named 'ludb-1.0.0.physionet.org' at the
    destination you entered.
    3. Run this script.
"""

# Imports
# built-in
import time
import os

# 3rd party
from collections import OrderedDict

import h5py
import pandas as pd
import numpy as np
import wfdb

# local
from biosppy import utils
from config import *
from tqdm import tqdm


def read_ludb_files(files=ludb):
    """
    Processes the data from the LUDB database and save it into pandas dataframe.

    Intended to be used under this particular circumstance,
    with :func:`download_url <comparison_qrs_detectors.comparison_qrs_detectors.download_db.download_url>` called before it, and potentially {to be determined}
    called after it, but optional.

    :parameter: Path to downloaded raw LUDB data

    :returns: A modified pandas dataframe.
    """
    # Get signal
    dfs_ecg = []
    dfs_rpeaks = []


    files = files / 'data'

    for participant in tqdm(range(200)):
        filename = str(participant + 1)
        #data, info = wfdb.rdsamp("./ludb-1.0.0.physionet.org/" + filename)
        data, info = wfdb.rdsamp(files.as_posix() + "/" + filename)

        # Get signal
        data = pd.DataFrame(data, columns=info["sig_name"])
        data = data[["i"]].rename(columns={"i": "ECG"})
        data["Participant"] = "LUDB_%.2i" %(participant + 1)
        data["Sample"] = range(len(data))
        data["Sampling_Rate"] = info['fs']
        data["Database"] = "LUDB"
        data = data.astype({'ECG': 'float32', 'Sample': 'int32', 'Sampling_Rate': 'int16'})

        # Get annotations
        anno = wfdb.rdann(files.as_posix() + "/" + filename, 'atr_i')

        # check if there are any other than the expected three types of beats
        if not (set(anno.symbol) == {'t', '(', 'p', ')', 'N'} or
                set(anno.symbol) == {'N', '(', 't', ')'}):
            assert 1 == 0, f'Element {participant} (Patient {data["Participant"]}) has abnormal beats.'
        # check whether all beats have a start or and end
        last_checked_spl = 0
        for counter in range(len(anno.symbol)):
            if counter % 3 == 0:
                assert anno.symbol[counter] == '(', 'There is no correct beginning.'
            elif counter % 3 == 1:
                assert anno.symbol[counter] in ['t', 'N', 'p'], \
                    f'There is no correct beat type ({anno.symbol[counter]}).'
            elif counter % 3 == 2:
                assert anno.symbol[counter] == ')', 'There is no correct ending.'
            elif anno.symbol[counter] in ['t' 'p' 'u']:
                print('In patient {0} there is ({1})-peak annotation.'.format(data["Participant"][0],
                                                                              anno.symbol[counter]))
            # check if we have any timing problems
            assert anno.sample[counter] > last_checked_spl, 'There are timing problems.'

        # check whether all annotations start and end with a beat
        assert anno.symbol[0] == '(' and anno.symbol[len(anno.symbol)-1] == ')', \
            'The beat at the beginning has no end or start.'

        # check whether all annotations can be divided by three
        assert len(anno.symbol) % 3 == 0, 'There is something off.'

        t = anno.sample[np.where(np.array(anno.symbol) == "t")[0]]  # Peak of T-wave
        N = anno.sample[np.where(np.array(anno.symbol) == "N")[0]] # Normal beat (Peak of R-wave)
        p = anno.sample[np.where(np.array(anno.symbol) == "p")[0]]    # Peak of P-wave
        # u = anno.sample[np.where(np.array(anno.symbol) == "u")[0]]    # Peak of U-wave
        if t.size == 0:
            t = np.full_like(N, np.NaN)
        if p.size == 0:
            p = np.full_like(N, np.NaN)

        #anno = pd.DataFrame(OrderedDict([("Tpeaks", t), ("Rpeaks", N), ("Ppeaks", p)]))
        anno_ordered = OrderedDict([("Tpeaks", t), ("Rpeaks", N), ("Ppeaks", p)])
        anno = pd.DataFrame({key: pd.Series(value) for key, value in anno_ordered.items()})
        #anno = pd.DataFrame([t, N, p], columns=["Tpeaks", "Rpeaks", "Ppeaks"])
        anno["Participant"] = "LUDB_%.2i" %(participant + 1)
        anno["Sampling_Rate"] = info['fs']
        anno["Database"] = "LUDB"
        anno = anno.fillna(0)
        anno = anno.astype({'Tpeaks': 'int32', 'Rpeaks': 'int32', 'Ppeaks': 'int32', 'Sampling_Rate': 'int16'})

        # Store with the rest
        dfs_ecg.append(data)
        dfs_rpeaks.append(anno)

    # Concatenate each data and annotations to one pandas dataframe
    df_ecg = pd.concat(dfs_ecg)
    dfs_rpeaks = pd.concat(dfs_rpeaks)

    # Annotate metadata
    metadata = {'User': 'Andreas Roth',
                'Date': time.time(),
                'OS': os.name,
                'Database': 'Lobachevsky University Database (LUDB)',
                'Suspects': '200',
                'Record': '200',
                'Manual annotation': 'Yes',
                'Measuring Device': 'Schiller Cardiovit AT-101 cardiograph',
                'Leads': 'i, ii, iii, avr, avl, avf, v1, v2, v3, v4, v5, v6',
                'Duration [s]': '10',
                'Sampling Frequency': '500',
                'Waves': '58429',
                'QRS': '21966',
                'T': '19666',
                'P': '16797',
                'Patients': 'healthy and cardiovascular diseases',
                'Age': '11-90, average 54',
                'Gender': '85 women, 115 men',
                'Licence': 'ODC Attribution License'}

    print(str(participant + 1) + "/200" + " Participant successful loaded!")

    # Save to csv file
    #df_ecg.to_csv(data_dir / processed / "LUDB_ECGs.csv", index=False)
    #dfs_rpeaks.to_csv(data_dir / processed / "LUDB_Rpeaks.csv", index=False)

    return df_ecg, dfs_rpeaks, metadata

def save_ludb_hdf5(df_ecg, dfs_rpeaks, name=ludb_raw_data_path):
    # Save as hdf5 file
    # We can create a HDF5 file using the HDFStore class provided by Pandas
    # normalize path
    path = utils.normpath(name.as_posix())
    with pd.HDFStore(path, mode='w') as hdf:
        # hdf = pd.HDFStore('ecg.h5')

        # Now we can store a dataset into the file we just created:
        # df = DataFrame(np.random.rand(5, 3), columns=('A', 'B', 'C'))  # put the dataset in the storage
        # Group for formatted ECG database
        # Sub-Group for each ECG data and annotations and dataframe for samples and R-Peak annotations
        hdf.put('ECG/Data', value=df_ecg, format='table', data_columns=True, complevel=9, complib='blosc')
        hdf.put('ECG/Rpeaks', value=dfs_rpeaks, format='table', data_columns=True, complevel=9, complib='blosc')

        # Annotate metadata to hdf5 file
        metadata = {'User': 'Andreas Roth',
                    'Date': time.time(),
                    'OS': os.name,
                    'Database': 'Lobachevsky University Database (LUDB)',
                    'Suspects': '200',
                    'Record': '200',
                    'Manual annotation': 'Yes',
                    'Measuring Device': 'Schiller Cardiovit AT-101 cardiograph',
                    'Leads': 'i, ii, iii, avr, avl, avf, v1, v2, v3, v4, v5, v6',
                    'Duration [s]': '10',
                    'Sampling Frequency': '500',
                    'Waves': '58429',
                    'QRS': '21966',
                    'T': '19666',
                    'P': '16797',
                    'Patients': 'healthy and cardiovascular diseases',
                    'Age': '11-90, average 54',
                    'Gender': '85 women, 115 men',
                    'Licence': 'ODC Attribution License'}
        hdf.get_storer('ECG/Data').attrs.metadata = metadata

def open_ludb_hdf5(name=ludb_raw_data_path):
    # normalize path
    path = utils.normpath(name.as_posix())
    # Open hdf5 file
    with pd.HDFStore(path, mode='r') as hdf:
        print(hdf.keys())
        ecg_dset = hdf.get('ECG/Data')
        rpeaks_dset = hdf.get('ECG/Rpeaks')
        # ecg_dset = hdf['ECG/Data']  # Dataset pointed to ssd memory, gone when hf closed
        ecg = ecg_dset  # numpy/pandas array
        # rpeaks_dset = hdf['ECG/Rpeaks']  # Dataset pointed to ssd memory, gone when hf closed
        rpeaks = rpeaks_dset  # numpy/pandas array
        metadata_dset = hdf.get_storer('ECG/Data').attrs.metadata
        metadata = metadata_dset
    print('Open hdf5 file {0}.'.format(path))
    return ecg, rpeaks, metadata

# return keyes of hdf5 file
def get_ludb_keyes(hf):
    keyes = []
    for key in hf.keys():
        # print(key)
        keyes.append(key)
    return keyes

def get_ludb_hdf5(name):
    # return object of hdf5 file if root group is 'ECG'
    def get_objects(name, obj):
        if 'ECG' in name:
            return obj

    # open hdf5 file
    with h5py.File(name, 'r') as f:
        group = f.visititems(get_objects)
        ecg = group['Data/Samples']  # Add [()] at the end to avoid error when dataset is closed
        rpeaks = group['Annotations/Rpeaks']  # This are datasets
        print('First ecg data sample: {0} and annotation: {0}'.format(ecg[0], rpeaks[0]))
