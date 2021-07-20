# -*- coding: utf-8 -*-
"""
comparison_qrs_detectors
-------
A toolbox for comparison of the Pan-Tompkins, Engelse-Zeelenberg and Christov QRS detector written in Python.
:copyright: (c) 2021 by Open Innovation Lab
:license: BSD 3-clause, see LICENSE for more details.
"""
import traceback
import time

#%% Testing ray
# Start Ray
import ray
import psutil
if not ray.is_initialized():
    GB = 1024 ** 3
    ray.init(num_cpus=psutil.cpu_count(logical=False), include_dashboard=False)#, _memory=4*GB, object_store_memory=2*GB)

import numpy as np
import pandas as pd
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 25)
np.set_printoptions(linewidth=400)
import pylab as pl
from biosppy.signals import ecg, tools
from biosppy.signals import tools as st
from config import mit_raw_data_path, ludb_raw_data_path
from preprocess_ludb import read_ludb_files, save_ludb_hdf5, open_ludb_hdf5
from preprocess_mit import read_mit_bit_files, save_mit_hdf5, open_mit_hdf5
from qrs_detectors import process_ecg, compare_segmentation, plot_raw_signal, plot_processed_signal
from tqdm import tqdm

@ray.remote(num_returns=3)
def load_dataset(database, hdf5=False):
    """Load dataset and annotations for each patient into pandas dataframes"""

    if database == "mit-bih":
        if hdf5==False:
            # load ECG data from mit-bit database into pandas dataframes and show them
            signal, anno, metadata = read_mit_bit_files()

        elif hdf5==True:
            # save ECG and Annotations from pandas dataframe to hdf5 file
            save_mit_hdf5(signal, anno, mit_raw_data_path)
            # load ECG and Annotations from hdf5 file into into pandas dataframes
            signal, anno, metadata = open_mit_hdf5(mit_raw_data_path)

        # split ECG and annotations dataframe into individual patients
        df_patients = signal[["ECG", "Participant"]]
        df_rpeaks = anno[["Rpeaks", "Tpeaks", "Ppeaks", "Participant"]]
        patients = []
        rpeaks = []

        for patient in tqdm(range(2)): #48
            patients.append(df_patients.loc[df_patients['Participant'] == 'MIT-Arrhythmia_%.2i' % (patient)])
            #patients.append(df_patients.query('Participant == '+'MIT-Arrhythmia_%.2i' % (patient)))
            rpeaks.append(df_rpeaks.loc[df_rpeaks['Participant'] == 'MIT-Arrhythmia_%.2i' % (patient)])

    elif database == "ludb":
        if hdf5 == False:
            # load ECG data from ludb database into pandas dataframes and show them
            signal, anno, metadata = read_ludb_files()
            # save ECG and Annotations from pandas dataframe to hdf5 file
            #save_ludb_hdf5(signal, anno, ludb_raw_data_path)
        elif hdf5 == True:
            # load ECG and Annotations from hdf5 file into into pandas dataframes
            signal, anno, metadata = open_ludb_hdf5(ludb_raw_data_path)

        # split ECG and annotations dataframe into individual patients
        df_patients = signal[["ECG", "Participant"]]
        df_rpeaks = anno[["Rpeaks", "Tpeaks", "Ppeaks", "Participant"]]
        patients = []
        rpeaks = []
        for patient in tqdm(range(200)):
            patients.append(df_patients.loc[df_patients['Participant'] == "LUDB_%.2i" % (patient + 1)])
            rpeaks.append(df_rpeaks.loc[df_rpeaks['Participant'] == 'LUDB_%.2i' % (patient + 1)])

    else:
        return print("No database (mit-bih or ludb) to load from specified!")

    print("ECG pandas dataframe: \n{0}\n\nAnnotations pandas dataframe: \n{1}\n\n{2}".format(signal.head(), anno.head(), metadata))

    return patients, rpeaks, metadata


def raw_signal_evaluation(signals, anno, metadata, patient=0):
    """Raw signal evaluation for patient"""
    plot_raw_signal(signals[patient], anno[patient], metadata)
    mean, median, maxAmp, sigma2, sigma, ad, kurt, skew = st.signal_stats(signal=signals[patient]['ECG'].to_numpy())

    raw = pd.DataFrame(data=np.array([[mean, median, maxAmp, sigma2, sigma, ad, kurt, skew]]),
                       columns=['mean', 'median', 'max', 'var', 'std_dev', 'abs_dev', 'kurtosis', 'skewness'])

    #raw_orderd = raw.as_dict()
    #raw = pd.DataFrame(data=raw_orderd, columns=raw_orderd[0].keys())
    print("Various metrics describing the raw input signal of participent {0}.\n{1}\n".format(
        signals[patient]['Participant'][0], raw))
    return raw


#@ray.remote
def qrs_detection(signals, metadata, patient=0):
    """Process raw signal and detect r-peaks with the three qrs-detectors"""
    # process it and plot (standard hamilton)
    pantompkins1 = process_ecg(signal=signals[patient]['ECG'].to_numpy(), sampling_rate=int(metadata['Sampling Frequency']),
                               show=True, detector="pantompkins1")

    engzee = process_ecg(signal=signals[patient]['ECG'].to_numpy(), sampling_rate=int(metadata['Sampling Frequency']),
                         show=True, detector="engzee")

    christov = process_ecg(signal=signals[patient]['ECG'].to_numpy(), sampling_rate=int(metadata['Sampling Frequency']),
                           show=True, detector="christov")

    return pantompkins1, engzee, christov


#@ray.remote
def evaluation_qrs_detectors(signals, detectors, rpeaks, metadata, patient=0):
    """QRS detector evaluation base on different metrics"""

    detecor_eval = []
    for detector in detectors:

        # compute different in detected to reference heart rate
        det_hr = detector['heart_rate']
        det_heart_rate_ts = detector['heart_rate_ts']
        ref_hr_idx, ref_hr = st.get_heart_rate(beats=rpeaks[patient]['Rpeaks'].to_numpy(),
                                               sampling_rate=int(metadata['Sampling Frequency']),
                                               smooth=True,
                                               size=3)
        # get time vectors
        length = len(signals[patient]['ECG'])
        T = (length - 1) / int(metadata['Sampling Frequency'])
        ts = np.linspace(0, T, length, endpoint=True)
        ts_hr = ts[ref_hr_idx]

        if len(ref_hr) == len(det_hr):
            diff_hr = []
            for i in range(len(ref_hr)):
                diff_hr.append(abs(ref_hr[i] - det_hr[i]))

            # plot heart rate
            pl.plot(ts_hr, ref_hr, linewidth=2, color='green', linestyle='-', label='Ref Heart Rate')
            pl.plot(det_heart_rate_ts, det_hr, linewidth=2, color='blue', linestyle='-', label='Det Heart Rate')
            pl.plot(det_heart_rate_ts, diff_hr, linewidth=2, color='red', linestyle='--', label='Diff Heart Rate')

            pl.title('Hard Rate')
            pl.xlabel('Time (s)')
            pl.ylabel('Heart Rate (bpm)')
            pl.legend()
            pl.grid()
            pl.show()
        else:
            print("Heart rate arrays didn't have same length!\n")

        # r-peak detection evaluation
        rpeaks_eval = compare_segmentation(signal=signals[patient], reference=rpeaks[patient]['Rpeaks'].to_numpy(), test=detector['rpeaks'],
                                           sampling_rate=int(metadata['Sampling Frequency']),
                                           offset=0, minRR=None, tol=0.05)
        rpeaks_eval_orderd = rpeaks_eval.as_dict()
        #print(rpeaks_eval_orderd)
        #rpeaks_eval = pd.DataFrame(rpeaks_eval_orderd, columns=rpeaks_eval_orderd.keys())  # , columns=rpeaks_eval_orderd.keys())
        rpeaks_eval = pd.DataFrame({key: pd.Series(value) for key, value in rpeaks_eval_orderd.items()})
        #print(rpeaks_eval.head())
        #TP, FP, TN, FN, perf, acc, err, sens, spec, detectionIdx, matchIdx, dev, mdev, sdev, rIBIm, rIBIs, tIBIm, tIBIs = \
        #    compare_segmentation(reference=rpeaks[patient]['Rpeaks'].to_numpy(), test=detector['rpeaks'],
        #                                   sampling_rate=int(metadata['Sampling Frequency']),
        #                                   offset=0, minRR=None, tol=0.05)
        #rpeaks_eval = pd.DataFrame(data=np.array([[TP, FP, TN, FN, perf, acc, err, sens, spec, detectionIdx, matchIdx,
        #                           dev, mdev, sdev, rIBIm, rIBIs, tIBIm, tIBIs]]), columns=['TP', 'FP', 'TN', 'FN',
        #                           'performance', 'acc', 'err', 'sens', 'spec', 'detect', 'match', 'deviation',
        #                           'mean_deviation', 'std_deviation', 'mean_ref_ibi', 'std_ref_ibi', 'mean_test_ibi',
        #                           'std_test_ibi'])

        rpeaks_eval.loc[0, 'Absolut matches'] = len(rpeaks_eval['match'].to_numpy()) / len(rpeaks[patient]['Rpeaks'].to_numpy())

        rpeaks_eval.rename(columns={'TP': 'True positive', 'FP': 'False positive',
                                    'TN': 'True negative', 'FN': 'False negative',
                                    'performance': 'Test performance', 'err': 'Error rate',
                                    'match': 'Match to R-peak', 'detect': 'TP(0)/FP(1)/TN(2)/FN(3)',
                                    'acc': 'Positive Predictive Value (PPV)', 'sens': 'Sensitivity',
                                    'spec': 'Specificity', 'mean_deviation': 'Mean Error (ME)',
                                    'deviation': 'Absolute error', 'std_deviation': 'Std. of error',
                                    'mean_ref_ibi': 'Mean ref interbeat interval',
                                    'std_ref_ibi': 'Std. ref interbeat interval',
                                    'mean_test_ibi': 'Mean test interbeat interval',
                                    'std_test_ibi': 'Std. test interbeat interval'}, inplace=True)

        plot_processed_signal(signals[patient], detector['rpeaks'], rpeaks[patient], metadata)
        print("Segmentation performance of R-peak detector for participent {0}.\n{1}\n".format(
            signals[patient]['Participant'][0], rpeaks_eval))

        #tools.find_intersection(x1=None, y1=None, x2=None, y2=None,
        #                                        alpha = 1.5, xtol = 1e-06, ytol = 1e-06)

        detecor_eval.append(rpeaks_eval)

    return detecor_eval


def test_mit_preprocess():
    """Test function - preprocess test mit
    """
    try:
        print("Preprocess MIT-BIH Test")
        # Quick test
        # load ECG data from mit-bih database into pandas dataframes and show them
        patients_ray, anno_ray, metadata_ray = load_dataset.remote(database="mit-bih", hdf5=False)
        patients = ray.get(patients_ray)
        anno = ray.get(anno_ray)
        metadata = ray.get(metadata_ray)
        del patients_ray, anno_ray, metadata_ray
        # load ECG data (mit-bih database) from hdf5 file into pandas dataframes and show them
        #patients, anno, metadata = load_dataset(database="mit-bih", hdf5=True)

        #print("Patient 1:")
        #print("ECG with {0} records: {1}".format(patients[0]['ECG'].to_numpy().size, patients[0]['ECG'].to_numpy()))
        #print("Rpeaks with {0} records: {1}\n".format(anno[0]['Rpeaks'].to_numpy().size, anno[0]['Rpeaks'].to_numpy()))

        # Plot and evaluation of raw ecg signal
        raw = raw_signal_evaluation(patients, anno, metadata, patient=1)

        # r-peaks detection with qrs-detectors
        pantompkins1, engzee, christov = qrs_detection(patients, metadata, patient=1)

        # qrs-detector evaluation
        eval = evaluation_qrs_detectors(patients, [pantompkins1, engzee, christov], anno, metadata, patient=1)

        return 0
    except (RuntimeError, TypeError, NameError, IOError, ValueError, EnvironmentError) as err:
        #print("Error:", err)
        print(traceback.format_exc())
        return 1

def test_ludb_preprocess():
    """Test function - preprocess ludb
    """
    try:
        print("\nPreprocess LUDB Test")
        # Quick test
        # load ECG data from ludb database into pandas dataframes and show them
        patients_ray, anno_ray, metadata_ray = load_dataset.remote(database="ludb", hdf5=False)
        patients = ray.get(patients_ray)
        anno = ray.get(anno_ray)
        metadata = ray.get(metadata_ray)
        del patients_ray, anno_ray, metadata_ray
        # load ECG data (mit-bih database) from hdf5 file into pandas dataframes and show them
        # patients, anno, metadata = load_dataset(database="mit-bih", hdf5=True)

        # print("Patient 1:")
        # print("ECG with {0} records: {1}".format(patients[0]['ECG'].to_numpy().size, patients[0]['ECG'].to_numpy()))
        # print("Rpeaks with {0} records: {1}\n".format(anno[0]['Rpeaks'].to_numpy().size, anno[0]['Rpeaks'].to_numpy()))

        # Plot and evaluation of raw ecg signal
        raw = raw_signal_evaluation(patients, anno, metadata, patient=0)

        # r-peaks detection with qrs-detectors
        pantompkins1, engzee, christov = qrs_detection(patients, metadata, patient=0)

        # qrs-detector evaluation
        eval = evaluation_qrs_detectors(patients, [pantompkins1, engzee, christov], anno, metadata, patient=0)

        return 0
    except (RuntimeError, TypeError, NameError, IOError, ValueError, EnvironmentError) as err:
        #print("Error:", err)
        print(traceback.format_exc())
        return 1

#if __name__ == "__Test_Preprocess__":

mit_bih_test = test_mit_preprocess()
ludb_test = 0#test_ludb_preprocess()

print("MIT-BIH preprocess test executed with {0}.\n"
      "LUDB preprocess test executed with {1}.".format(mit_bih_test, ludb_test))
if not (mit_bih_test and ludb_test):
    print("All tests executed well!")
elif not mit_bih_test:
    print("MIT-BIT preprocess test failed!")
elif not ludb_test:
    print("LUDB preprocess test failed!")
