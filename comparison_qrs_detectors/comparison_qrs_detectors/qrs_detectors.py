# Imports
# compat
from __future__ import absolute_import, division, print_function

# 3rd party
import math
import warnings

import numpy as np
import pandas
import scipy.signal as ss
import scipy.signal as signal
import seaborn as sns
# style settings
from ecgdetectors import Detectors
from matplotlib.pyplot import figure
from scipy.interpolate import interp1d
from sklearn import metrics

sns.set(style='whitegrid', rc={'axes.facecolor': '#EFF2F7'})

# local
from biosppy import tools as st
from biosppy import plotting, utils, signals
from biosppy.signals.ecg import hamilton_segmenter, christov_segmenter, engzee_segmenter, correct_rpeaks, extract_heartbeats


def process_ecg(signal=None, sampling_rate=1000., show=True, detector='hamilton'):
    """Process a raw ECG signal and extract relevant signal features using
    default parameters.
    Parameters
    ----------
    signal : array
        Raw ECG signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    show : bool, optional
        If True, show a summary plot.
    Returns
    -------
    ts : array
        Signal time axis reference (seconds).
    filtered : array
        Filtered ECG signal.
    rpeaks : array
        R-peak location indices.
    templates_ts : array
        Templates time axis reference (seconds).
    templates : array
        Extracted heartbeat templates.
    heart_rate_ts : array
        Heart rate time axis reference (seconds).
    heart_rate : array
        Instantaneous heart rate (bpm).
    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # ensure numpy
    signal = np.array(signal)

    sampling_rate = float(sampling_rate)

    # filter signal
    order = int(0.3 * sampling_rate)
    filtered, _, _ = st.filter_signal(signal=signal,
                                      ftype='FIR',
                                      band='bandpass',
                                      order=order,
                                      frequency=[3, 45],
                                      sampling_rate=sampling_rate)

    if detector=='hamilton':
        # segment hamilton
        rpeaks, = hamilton_segmenter(signal=filtered, sampling_rate=sampling_rate)
    elif detector=='pantompkins1':
        # Pan-Tompkins
        MWA_name = 'cumulative'
        diff = np.diff(filtered)

        squared = diff * diff

        N = int(0.12 * sampling_rate)
        mwa = MWA_from_name(MWA_name)(squared, N)
        mwa[:int(0.2 * sampling_rate)] = 0

        rpeaks = panPeakDetect(mwa, sampling_rate)

    elif detector=='pantompkins2':
        rpeaks = _ecg_findpeaks_pantompkins(filtered, sampling_rate=sampling_rate)

    elif detector=='engzee':
        # segment Engelse and Zeelenberg
        rpeaks, = engzee_segmenter(signal=filtered, sampling_rate=sampling_rate, threshold=0.48)
    elif detector=='christov':
        # segment christov
        rpeaks, = christov_segmenter(signal=filtered, sampling_rate=sampling_rate)

    # correct R-peak locations
    rpeaks, = correct_rpeaks(signal=filtered,
                             rpeaks=rpeaks,
                             sampling_rate=sampling_rate,
                             tol=0.05)

    # extract templates
    templates, rpeaks = extract_heartbeats(signal=filtered,
                                           rpeaks=rpeaks,
                                           sampling_rate=sampling_rate,
                                           before=0.2,
                                           after=0.4)

    # compute heart rate
    hr_idx, hr = st.get_heart_rate(beats=rpeaks,
                                   sampling_rate=sampling_rate,
                                   smooth=True,
                                   size=3)

    # get time vectors
    length = len(signal)
    T = (length - 1) / sampling_rate
    ts = np.linspace(0, T, length, endpoint=True)
    ts_hr = ts[hr_idx]
    ts_tmpl = np.linspace(-0.2, 0.4, templates.shape[1], endpoint=False)

    # plot
    if show:
        plotting.plot_ecg(ts=ts,
                          raw=signal,
                          filtered=filtered,
                          rpeaks=rpeaks,
                          templates_ts=ts_tmpl,
                          templates=templates,
                          heart_rate_ts=ts_hr,
                          heart_rate=hr,
                          path=None,
                          show=True)

    # output
    args = (ts, filtered, rpeaks, ts_tmpl, templates, ts_hr, hr)
    names = ('ts', 'filtered', 'rpeaks', 'templates_ts', 'templates',
             'heart_rate_ts', 'heart_rate')

    return utils.ReturnTuple(args, names)


def pan_tompkins_detector(unfiltered_ecg, fs, MWA_name='cumulative'):
    """
    Jiapu Pan and Willis J. Tompkins.
    A Real-Time QRS Detection Algorithm.
    In: IEEE Transactions on Biomedical Engineering
    BME-32.3 (1985), pp. 230–236.
    """

    f1 = 5 / fs
    f2 = 15 / fs

    b, a = signal.butter(1, [f1 * 2, f2 * 2], btype='bandpass')

    filtered_ecg = signal.lfilter(b, a, unfiltered_ecg)

    diff = np.diff(filtered_ecg)

    squared = diff * diff

    N = int(0.12 * fs)
    mwa = MWA_from_name(MWA_name)(squared, N)
    mwa[:int(0.2 * fs)] = 0

    mwa_peaks = panPeakDetect(mwa, fs)

    return mwa_peaks


def panPeakDetect(detection, fs):
    min_distance = int(0.25 * fs)

    signal_peaks = [0]
    noise_peaks = []

    SPKI = 0.0
    NPKI = 0.0

    threshold_I1 = 0.0
    threshold_I2 = 0.0

    RR_missed = 0
    index = 0
    indexes = []

    missed_peaks = []
    peaks = []

    for i in range(len(detection)):

        if i > 0 and i < len(detection) - 1:
            if detection[i - 1] < detection[i] and detection[i + 1] < detection[i]:
                peak = i
                peaks.append(i)

                if detection[peak] > threshold_I1 and (peak - signal_peaks[-1]) > 0.3 * fs:

                    signal_peaks.append(peak)
                    indexes.append(index)
                    SPKI = 0.125 * detection[signal_peaks[-1]] + 0.875 * SPKI
                    if RR_missed != 0:
                        if signal_peaks[-1] - signal_peaks[-2] > RR_missed:
                            missed_section_peaks = peaks[indexes[-2] + 1:indexes[-1]]
                            missed_section_peaks2 = []
                            for missed_peak in missed_section_peaks:
                                if missed_peak - signal_peaks[-2] > min_distance and signal_peaks[
                                    -1] - missed_peak > min_distance and detection[missed_peak] > threshold_I2:
                                    missed_section_peaks2.append(missed_peak)

                            if len(missed_section_peaks2) > 0:
                                missed_peak = missed_section_peaks2[np.argmax(detection[missed_section_peaks2])]
                                missed_peaks.append(missed_peak)
                                signal_peaks.append(signal_peaks[-1])
                                signal_peaks[-2] = missed_peak

                else:
                    noise_peaks.append(peak)
                    NPKI = 0.125 * detection[noise_peaks[-1]] + 0.875 * NPKI

                threshold_I1 = NPKI + 0.25 * (SPKI - NPKI)
                threshold_I2 = 0.5 * threshold_I1

                if len(signal_peaks) > 8:
                    RR = np.diff(signal_peaks[-9:])
                    RR_ave = int(np.mean(RR))
                    RR_missed = int(1.66 * RR_ave)

                index = index + 1

    signal_peaks.pop(0)

    return signal_peaks

def MWA_from_name(function_name):
    if function_name == "cumulative":
        return MWA_cumulative
    elif function_name == "convolve":
        return MWA_convolve
    elif function_name == "original":
        return MWA_original
    else:
        raise RuntimeError('invalid moving average function!')


# Fast implementation of moving window average with numpy's cumsum function
def MWA_cumulative(input_array, window_size):
    ret = np.cumsum(input_array, dtype=float)
    ret[window_size:] = ret[window_size:] - ret[:-window_size]

    for i in range(1, window_size):
        ret[i - 1] = ret[i - 1] / i
    ret[window_size - 1:] = ret[window_size - 1:] / window_size

    return ret


# Original Function
def MWA_original(input_array, window_size):
    mwa = np.zeros(len(input_array))
    mwa[0] = input_array[0]

    for i in range(2, len(input_array) + 1):
        if i < window_size:
            section = input_array[0:i]
        else:
            section = input_array[i - window_size:i]

        mwa[i - 1] = np.mean(section)

    return mwa


# Fast moving window average implemented with 1D convolution
def MWA_convolve(input_array, window_size):
    ret = np.pad(input_array, (window_size - 1, 0), 'constant', constant_values=(0, 0))
    ret = np.convolve(ret, np.ones(window_size), 'valid')

    for i in range(1, window_size):
        ret[i - 1] = ret[i - 1] / i
    ret[window_size - 1:] = ret[window_size - 1:] / window_size

    return ret

# =============================================================================
# Pan & Tompkins (1985)
# =============================================================================
def _ecg_findpeaks_pantompkins(signal, sampling_rate=1000):
    """From https://github.com/berndporr/py-ecg-detectors/
    - Jiapu Pan and Willis J. Tompkins. A Real-Time QRS Detection Algorithm.
    In: IEEE Transactions on Biomedical Engineering BME-32.3 (1985), pp. 230–236.
    """
    diff = np.diff(signal)

    squared = diff * diff

    N = int(0.12 * sampling_rate)
    mwa = _ecg_findpeaks_MWA(squared, N)
    mwa[: int(0.2 * sampling_rate)] = 0

    mwa_peaks = _ecg_findpeaks_peakdetect(mwa, sampling_rate)

    mwa_peaks = np.array(mwa_peaks, dtype="int")
    return mwa_peaks

def _ecg_findpeaks_MWA(signal, window_size):
    """From https://github.com/berndporr/py-ecg-detectors/"""

    mwa = np.zeros(len(signal))
    sums = np.cumsum(signal)

    def get_mean(begin, end):
        if begin == 0:
            return sums[end - 1] / end

        dif = sums[end - 1] - sums[begin - 1]
        return dif / (end - begin)

    for i in range(len(signal)):  # pylint: disable=C0200
        if i < window_size:
            section = signal[0:i]
        else:
            section = get_mean(i - window_size, i)

        if i != 0:
            mwa[i] = np.mean(section)
        else:
            mwa[i] = signal[i]

    return mwa

def _ecg_findpeaks_peakdetect(detection, sampling_rate=1000):
    """From https://github.com/berndporr/py-ecg-detectors/"""
    min_distance = int(0.25 * sampling_rate)

    signal_peaks = [0]
    noise_peaks = []

    SPKI = 0.0
    NPKI = 0.0

    threshold_I1 = 0.0
    threshold_I2 = 0.0

    RR_missed = 0
    index = 0
    indexes = []

    missed_peaks = []
    peaks = []

    for i in range(len(detection)):  # pylint: disable=R1702,C0200

        # pylint: disable=R1716
        if i > 0 and i < len(detection) - 1 and detection[i - 1] < detection[i] and detection[i + 1] < detection[i]:
            peak = i
            peaks.append(peak)  # pylint: disable=R1716
            if detection[peak] > threshold_I1 and (peak - signal_peaks[-1]) > 0.3 * sampling_rate:

                signal_peaks.append(peak)
                indexes.append(index)
                SPKI = 0.125 * detection[signal_peaks[-1]] + 0.875 * SPKI
                if RR_missed != 0 and signal_peaks[-1] - signal_peaks[-2] > RR_missed:
                    missed_section_peaks = peaks[indexes[-2] + 1 : indexes[-1]]
                    missed_section_peaks2 = []
                    for missed_peak in missed_section_peaks:
                        if missed_peak - signal_peaks[-2] > min_distance:
                            if signal_peaks[-1] - missed_peak > min_distance:
                                if detection[missed_peak] > threshold_I2:
                                    missed_section_peaks2.append(missed_peak)

                    if missed_section_peaks2:
                        missed_peak = missed_section_peaks2[np.argmax(detection[missed_section_peaks2])]
                        missed_peaks.append(missed_peak)
                        signal_peaks.append(signal_peaks[-1])
                        signal_peaks[-2] = missed_peak

            else:
                noise_peaks.append(peak)
                NPKI = 0.125 * detection[noise_peaks[-1]] + 0.875 * NPKI

            threshold_I1 = NPKI + 0.25 * (SPKI - NPKI)
            threshold_I2 = 0.5 * threshold_I1

            if len(signal_peaks) > 8:
                RR = np.diff(signal_peaks[-9:])
                RR_ave = int(np.mean(RR))
                RR_missed = int(1.66 * RR_ave)

            index += 1

    signal_peaks.pop(0)

    return signal_peaks


import pylab as pl

def plot_raw_signal(signal, peaks, metadata):
    """"Create a plot for raw ECG signal with annotations.

    Parameters
    ----------
    signal : array
           Raw ECG signal.
    peaks : array
           peaks location indices.
    metadata : dict
           Dictionary with signal metadata.
    """
    signal = signal['ECG'].to_numpy()
    rpeaks = peaks['Rpeaks'].to_numpy()
    tpeaks = peaks['Tpeaks'].to_numpy()
    ppeaks = peaks['Ppeaks'].to_numpy()

    Fs = metadata['Sampling Frequency']
    N = len(signal)  # number of samples
    T = (N - 1) / int(Fs)  # duration

    # Test for 6s
    # Fs = metadata['Sampling Frequency']
    # N = (6 * int(Fs)) + 1  # Number of samples for given time of 30s and frequency
    # T = (N - 1) / int(Fs)  # duration
    # signal = signal[:N]
    # rpeaks = rpeaks[:7]

    ts = np.linspace(0, T, N, endpoint=False)  # relative timestamps

    pl.title('Raw ECG signal with annotations')
    pl.xlabel('Time [s]')
    pl.ylabel('Amplitude [mV]')
    #pl.plot(ts, signal, color='b', linewidth=2, label='Raw ECG signal')
    #pl.scatter(ts[rpeaks], signal[rpeaks],
    #           color='r',
    #           linewidth=1,
    #           label='R-peaks')

    ymin = np.min(signal)
    ymax = np.max(signal)
    alpha = 0.1 * (ymax - ymin)
    ymax += alpha
    ymin -= alpha

    pl.plot(ts, signal, color='b', linewidth=2, label='Raw ECG signal')
    pl.vlines(ts[rpeaks], ymin, ymax,
               color='r',
               linewidth=1,
               label='R-peaks')
    # if not (np.isnan(tpeaks.any()) and np.isnan(ppeaks.any())) and not ((tpeaks.any() and ppeaks.any()) == -2147483648):
    #     pl.vlines(ts[tpeaks], ymin, ymax,
    #               color='green',
    #               linewidth=1,
    #               label='T-peaks')
    #     pl.vlines(ts[ppeaks], ymin, ymax,
    #               color='orange',
    #               linewidth=1,
    #               label='P-peaks')
    pl.legend()
    pl.grid()
    pl.show()

def plot_processed_signal(signal, rpeaks, anno, metadata):
    """"Create a plot for processed ECG signal with detected r-peaks.

    Parameters
    ----------
    signal : array
           Raw ECG signal.
    rpeaks : array
           r-peaks location indices.
    anno : array
           peaks location indices.
    metadata : dict
           Dictionary with signal metadata.
    """
    signal = signal['ECG'].to_numpy()
    #rpeaks = rpeaks[0]['Rpeaks'].to_numpy()
    ranno = anno['Rpeaks'].to_numpy()
    tanno = anno['Tpeaks'].to_numpy()
    panno = anno['Ppeaks'].to_numpy()

    Fs = metadata['Sampling Frequency']
    N = len(signal)  # number of samples
    T = (N - 1) / int(Fs)  # duration

    # Test for 6s
    #Fs = metadata['Sampling Frequency']
    #N = (6 * int(Fs)) + 1  # Number of samples for given time of 30s and frequency
    #T = (N - 1) / int(Fs)  # duration
    #signal = signal[:N]
    #rpeaks = rpeaks[:7]
    #ranno = ranno[:7]

    ts = np.linspace(0, T, N, endpoint=False)  # relative timestamps
    pl.title('Processed ECG signal with detected R-peaks and annotations')
    pl.xlabel('Time [s]')
    pl.ylabel('Amplitude [mV]')
    #pl.plot(ts, signal, color='b', linewidth=2, label='Raw ECG signal')
    #pl.scatter(ts[ranno], signal[rpeaks],
    #           color='r',
    #           linewidth=1,
    #           label='R-peaks')

    ymin = np.min(signal)
    ymax = np.max(signal)
    alpha = 0.1 * (ymax - ymin)
    ymax += alpha
    ymin -= alpha

    pl.plot(ts, signal, color='b', linewidth=2, label='Raw ECG signal')
    pl.vlines(ts[rpeaks], ymin, ymax,
              color='m',
              linewidth=1,
              label='Detected R-peaks')
    pl.vlines(ts[ranno], ymin, ymax,
               color='r',
               linewidth=1,
               label='R-peaks')

    # if not (np.isnan(tanno.any()) and np.isnan(panno.any())) and not ((tanno.any() and panno.any()) == -2147483648):
    #     pl.vlines(ts[tanno], ymin, ymax,
    #               color='green',
    #               linewidth=1,
    #               label='T-peaks')
    #     pl.vlines(ts[panno], ymin, ymax,
    #               color='orange',
    #               linewidth=1,
    #               label='P-peaks')
    pl.legend()
    pl.grid()
    pl.show()

## Metrics
# True Positive (TP) - QRS detection within range of 100 ms to left and right of QRS complex
# True Negative (TN) - interval between two QRS complexes minus their 100 ms thresholds not contain any prediction
# False Positive (FP) - prediction farther away than 100 ms from QRS complex
# False Negative (FN) - 200 ms interval around QRS complex not containing a prediction

# Positive Predictive Value (PPV) = TP / TP + FP
# Sensitivity (Sens) = TP / TP + FN
# Specificity (Spec) = TN / TN + FP (how well algorithm are able to recognize periods without heartbeats)
# Mean Error (ME or equivalent to Prediction Offset or Time Difference, respectively) =
# averaging time differences between prediction and closest actual QRS complex (within 100ms r/l)
# If No prediction is closest to a QRS complex, this is not reflected in the metric.
# on top
# Heart rate absolut
# rr interval diff

# binary_detections
# len(binary_detections) = ecg - shape[0]
# np.argwhere(binary_detections)
# idx = np.argwhere(binary_detections
# rr_intervalls = np.diff(idx)

def compare_segmentation(signal, reference=None, test=None, sampling_rate=500.,
                         offset=0, minRR=None, tol=0.05):
    """Compare the segmentation performance of a list of R-peak positions
    against a reference list.
    Parameters
    ----------
    reference : array
        Reference R-peak location indices.
    test : array
        Test R-peak location indices.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    offset : int, optional
        Constant a priori offset (number of samples) between reference and
        test R-peak locations.
    minRR : float, optional
        Minimum admissible RR interval (seconds).
    tol : float, optional
        Tolerance between corresponding reference and test R-peak
        locations (seconds).
    Returns
    -------
    TP : int
        Number of true positive R-peaks.
    FP : int
        Number of false positive R-peaks.
    performance : float
        Test performance; TP / len(reference).
    acc : float
        Accuracy rate; TP / (TP + FP).
    err : float
        Error rate; FP / (TP + FP).
    match : list
        Indices of the elements of 'test' that match to an R-peak
        from 'reference'.
    deviation : array
        Absolute errors of the matched R-peaks (seconds).
    mean_deviation : float
        Mean error (seconds).
    std_deviation : float
        Standard deviation of error (seconds).
    mean_ref_ibi : float
        Mean of the reference interbeat intervals (seconds).
    std_ref_ibi : float
        Standard deviation of the reference interbeat intervals (seconds).
    mean_test_ibi : float
        Mean of the test interbeat intervals (seconds).
    std_test_ibi : float
        Standard deviation of the test interbeat intervals (seconds).
    """

    # check inputs
    if reference is None:
        raise TypeError("Please specify an input reference list of R-peak \
                        locations.")

    if test is None:
        raise TypeError("Please specify an input test list of R-peak \
                        locations.")

    if minRR is None:
        minRR = np.inf

    sampling_rate = float(sampling_rate)

    raw = signal['ECG'].to_numpy()

    # ensure numpy
    reference = np.array(reference)
    test = np.array(test)

    # convert to samples
    minRR = minRR * sampling_rate
    tol = tol * sampling_rate

    # confusion matrix
    TP = 0  # True Positive (TP) - QRS detection within range of 100 ms to left and right of QRS complex (error < 100ms)
    FP = 0  # False Positive (FP) - prediction farther away than 100 ms from QRS complex (error > 100ms)
    TN = 0  # True Negative (TN) - interval between two QRS complexes minus their 100 ms thresholds not contain any prediction
            # ( error > (rr_interval - 100ms) )
    FN = 0  # False Negative (FN) - 200 ms interval around QRS complex not containing a prediction (error > 200ms)

    matchIdx = []
    detectionIdx = [] # TP: 0, FP: 1, TN: 2, FN: 3
    dev = []

    for i, r in enumerate(test):
        # deviation to closest R in reference
        ref = reference[np.argmin(np.abs(reference - (r + offset)))]
        error = np.abs(ref - (r + offset))

        if error < tol:
            TP += 1
            matchIdx.append(i)
            dev.append(error)
            detectionIdx.append(0)
        elif error > tol:
            if error > (2 * tol):
                FN += 1
                detectionIdx.append(3)
            if len(matchIdx) > 0:
                bdf = r - test[matchIdx[-1]]
                if tol < error < (bdf - tol):
                    FP += 1
                    detectionIdx.append(1)
                if error > (bdf - tol):
                    TN += 1
                    detectionIdx.append(2)

    print("Ref. peaks: {0}, Det. peaks: {1}".format(len(reference), len(test)))
    print("TP: {0}, FP: {1}, TN: {2}, FN: {3}".format(TP, FP, TN, FN))


    # reference  # annot exists
    # # borders array exists
    # borders = np.zeros((1, len(reference * 2)))
    # test  # detec exists
    # signal = signal['ECG'].to_numpy()  # signal exits

    binary_detect = np.zeros((len(raw),), dtype=int)
    #print(binary_detect)
    #print(binary_detect.shape)
    binary_detect[test] = 1
    binary_ref = np.zeros((len(raw),), dtype=int)
    binary_ref[reference] = 1

    # Threshold for 100ms
    Fs = sampling_rate
    N = int((tol * int(Fs)) + 1)  # Number of samples for given time of 100ms and frequency
    #signal = signal[:N]
    # rpeaksmatch = []
    # for i, value in enumerate(binary_ref):
    #     if value == 1:
    #         if N < i < (len(binary_ref) - N):
    #             if np.sum(binary_ref[i-N:i+N]) == 1 and np.sum(binary_detect[i-N:i+N]) == 1:
    #                 TP += 1
    #                 rpeaksmatch.append(i)
    #             elif np.sum([binary_ref[i-N:i+N], binary_detect[i-N:i+N]]) > 2:
    #                 TP += 1
    #                 FP += np.sum([binary_ref[i-N:i+N], binary_detect[i-N:i+N]]) - 1
    #                 print("Error! More than one correct R-Peak in interval detected (exactly {0}).".format(np.sum([binary_ref[i-N:i+N], binary_detect[i-N:i+N]]) - 1))
    #             elif np.sum([binary_ref[i-N:i+N], binary_detect[i-N:i+N]]) == 0:
    #                 bdf = ((i - binary_ref[rpeaksmatch[-1]]) / 2) - N
    #                 np.sum([binary_ref[(i - binary_ref[rpeaksmatch[-1]])/2:i-N],
    #                         binary_ref[i+N:(i - binary_ref[rpeaksmatch[-1]])/2]])
    #                 TN += 1
    #             elif np.sum(binary_ref[i-N:i+N]) == 1 and np.sum(binary_detect[i-N:i+N]) == 0:
    #                 FN += 1
    #             elif np.sum(binary_ref[i-N:i+N]) == 0 and np.sum(binary_detect[i-N:i+N]) == 1:
    #                 np.sum([i+binary_ref[(i - binary_ref[rpeaksmatch[-1]])/2:i+binary_ref[(i - binary_ref[rpeaksmatch[-1]]-N,
    #                         binary_ref[i + N:i - binary_ref[rpeaksmatch[-1]]]]])
    #                 FP += 1
    #         else:
    #             if np.sum([[i], binary_detect[i]]) == 1:
    #                 TP += 1
    #             elif np.sum([binary_ref[i], binary_detect[i]]) == 0:
    #                 TN += 1
    #             elif binary_ref[i] == 1 and binary_detect[i] == 0:
    #                 FN += 1
    #             elif binary_ref[i] == 0 and binary_detect[i] == 1:
    #                 FP += 1

    conf_mat = metrics.confusion_matrix(binary_ref, binary_detect, labels=[1])
    conf_mat = conf_mat.astype(float)
    TP0 = conf_mat[0, 0]
    FP0 = sum(conf_mat[:, 0]) - conf_mat[0, 0]
    TN0 = sum(sum(conf_mat)) - sum(conf_mat[0, :]) - sum(conf_mat[:, 0]) + conf_mat[0, 0]
    FN0 = sum(conf_mat[0, :]) - conf_mat[0, 0]

    print("0: TP: {0}, FP: {1}, TN: {2}, FN: {3}".format(TP0, FP0, TN0, FN0))


    detected_rpeaks = np.unique(test)
    reference_rpeaks = np.unique(reference)

    TP1 = 0
    TN1 = 0
    FP1 = 0
    FN1 = 0

    for i, anno_value in enumerate(detected_rpeaks):
        test_range = np.arange(anno_value - tol/2 - offset, anno_value + 1 + tol/2 - offset)
        in1d = np.in1d(test_range, reference_rpeaks)
        if np.any(in1d):
            TP1 = TP1 + 1

        if 0 < i < len(detected_rpeaks) - 1:
            tn_lower_range = np.arange(anno_value - ((anno_value - detected_rpeaks[i-1])/2), anno_value + 1 - tol/2)
            tn_upper_range = np.arange(anno_value + tol/2, anno_value + 1 + ((detected_rpeaks[i + 1] - anno_value)/2))
            tn_lower_in1d = np.in1d(tn_lower_range, reference_rpeaks)
            tn_upper_in1d = np.in1d(tn_upper_range, reference_rpeaks)
            if np.any(tn_lower_in1d) or np.any(tn_upper_in1d):
                TN1 = TN1 + 1


            fp_lower_range = np.arange(anno_value + ((detected_rpeaks[i + 1] - anno_value)/2), anno_value + 1 + (detected_rpeaks[i + 1] - anno_value) - tol/2)
            if i < len(detected_rpeaks) - 2:
                fp_upper_range = np.arange(anno_value + (detected_rpeaks[i + 1] - anno_value) + tol, anno_value + 1 + (detected_rpeaks[i + 1] - anno_value) + (detected_rpeaks[i + 2] - detected_rpeaks[i + 1])/2)
            elif i >= len(detected_rpeaks) - 2:
                fp_upper_range = np.arange(anno_value + (detected_rpeaks[i + 1] - anno_value) + tol,
                                           anno_value + 1 + (detected_rpeaks[i + 1] - anno_value) + 3*tol)
            fp_lower_in1d = np.in1d(fp_lower_range, reference_rpeaks)
            fp_upper_in1d = np.in1d(fp_upper_range, reference_rpeaks)
            if np.any(fp_lower_in1d) or np.any(fp_upper_in1d):
                FP1 = FP1 + 1


            fn_range = np.arange(anno_value + (detected_rpeaks[i + 1] - anno_value) - tol/2, anno_value + 1 + (detected_rpeaks[i + 1] - anno_value) + tol/2)
            fn_in1d = np.in1d(fn_range, reference_rpeaks)
            if np.any(fn_in1d):
                FN1 = FN1 + 1

        if i == 0 or i == len(detected_rpeaks):
            if not np.any(in1d):
                TN1 = TN1 + 1

    #FP1 = len(detected_rpeaks) - TP1
    #FN1 = abs(len(reference_rpeaks) - TP1)
    #TN1 = abs(len(detected_rpeaks) - FN1)
    print("1: TP: {0}, FP: {1}, TN {2}, FN: {3}".format(TP1, FP1, TN1, FN1))

    def calcMedianDelay(detected_peaks, unfiltered_ecg, search_samples):

        r_peaks = []
        window = int(search_samples)

        for i in detected_peaks:
            i = int(i)
            if i < window:
                section = unfiltered_ecg[:i]
                r_peaks.append(np.argmax(section))
            else:
                section = unfiltered_ecg[i - window:i]
                r_peaks.append(window - np.argmax(section))

        m = int(np.median(r_peaks))
        # print("Delay = ",m)
        return m

    def evaluate_detector(test, annotation, delay, tol=0):

        test = np.unique(test)
        reference = np.unique(annotation)

        TP = 0

        for anno_value in test:
            test_range = np.arange(anno_value - tol - delay, anno_value + 1 + tol - delay)
            in1d = np.in1d(test_range, reference)
            if np.any(in1d):
                TP = TP + 1

        FP = len(test) - TP
        FN = len(reference) - TP

        return TP, FP, FN

    #delay = calcMedianDelay(test, raw, max_delay_in_samples)

    #if delay > 1:
    TP2, FP2, FN2 = evaluate_detector(test, reference, delay=0, tol=tol/2)
    TN2 = len(reference) - (FN2)

    print("2: TP: {0}, FP: {1}, TN {2}, FN: {3}".format(TP2, FP2, TN2,  FN2))

    # Confussion matrix
    # for i, value in enumerate(binary_ref):
    #     if value == 1:
    #         if N < i < (len(binary_ref) - N):
    #             binary_ref[i - N:i + N] = 1
    #         elif i < N:
    #             binary_ref[0:i+N] = 1
    #         elif i > (len(binary_ref) - N):
    #             binary_ref[i - N:len(binary_ref)] = 1
    # conf_mat = metrics.confusion_matrix(binary_ref, binary_detect, labels=[0, 1])
    # conf_mat = conf_mat.astype(float)
    # TP = conf_mat[0, 0]
    # FP = sum(conf_mat[:, 0]) - conf_mat[0, 0]
    # TN = sum(sum(conf_mat)) - sum(conf_mat[0, :]) - sum(conf_mat[:, 0]) + conf_mat[0, 0]
    # FN = sum(conf_mat[0, :]) - conf_mat[0, 0]
    # print("TP: {0}, FP: {1}, TN: {2}, FN: {3}".format(TP, FP, TN, FN))

    # TP = conf_mat[0, 0]
    # FP = sum(conf_mat[:, 0]) - conf_mat[0, 0]
    # TN = sum(sum(conf_mat)) - sum(conf_mat[0, :]) - sum(conf_mat[:, 0]) + conf_mat[0, 0]
    # FN = sum(conf_mat[0, :]) - conf_mat[0, 0]

        #
        # for cnter in range(len(borders)):
        #     if cnter % 4 == 0:
        #         # TP or FN range would be borders[cnter] until borders[cnter+1]
        #         # This is the range where there should be one annotation
        #         if np.sum(binary_detect[borders[cnter]:borders[cnter + 1]]) == 1:
        #             TP += 1
        #         elif np.sum(binary_detect[borders[cnter]:borders[cnter + 1]]) > 1:
        #             raise ValueError('oops.')
        #             TP += 1
        #             FP += np.sum(binary_detect[borders[cnter]:borders[cnter + 1]]) - 1
        #         else:
        #             FN += 1
        #     if cnter % 4 == 1:
        #         # FP or TN would be borders[cnter] until borders[cnter+1]
        #         # This is the range where there should not be any annotation
        #         if np.sum(binary_detect[borders[cnter]:borders[cnter + 1]]):
        #             FP += np.sum(binary_detect[borders[cnter]:borders[cnter + 1]])
        #         else:
        #             TN += 1

        # if error < tol:
        #     TP += 1
        #     matchIdx.append(i)
        #     dev.append(error)
        #     detectionIdx.append(0)
        # elif error > tol:
        #     if error > (2 * tol):
        #         FN += 1
        #         detectionIdx.append(3)
        #     elif len(matchIdx) > 0:
        #         bdf = r - test[matchIdx[-1]]
        #         if bdf < minRR:
        #             # false positive, but removable with RR interval check
        #             pass
        #         elif bdf > minRR:
        #             FP += 1
        #             detectionIdx.append(1)
        #         elif error > (bdf - tol):
        #             TN += 1
        #             detectionIdx.append(2)
        #     else:
        #         FP += 1
        #         detectionIdx.append(1)

    # convert deviations (error by TP) to time
    dev = np.array(dev, dtype='float')
    dev /= sampling_rate
    nd = len(dev)
    if nd == 0:
        mdev = np.nan
        sdev = np.nan
    elif nd == 1:
        mdev = np.mean(dev)
        sdev = 0.
    else:
        mdev = np.mean(dev)
        sdev = np.std(dev, ddof=1)



    # interbeat interval thresholds
    th1 = 1.5  # 40 bpm
    th2 = 0.3  # 200 bpm

    # reference interbeat intervals
    rIBI = np.diff(reference)
    rIBI = np.array(rIBI, dtype='float')
    # Basic resampling to signal sampling rate to standardize the scale
    rIBI /= sampling_rate

    good = np.nonzero((rIBI < th1) & (rIBI > th2))[0]
    rIBI = rIBI[good]   # list of rr-intervall in s for annotated r-peaks

    nr = len(rIBI)
    if nr == 0:
        rIBIm = np.nan
        rIBIs = np.nan
    elif nr == 1:
        rIBIm = np.mean(rIBI)
        rIBIs = 0.
    else:
        rIBIm = np.mean(rIBI)
        rIBIs = np.std(rIBI, ddof=1)

    # detected interbeat intervals
    tIBI = np.diff(test[matchIdx])
    tIBI = np.array(tIBI, dtype='float')
    # Basic resampling to signal sampling rate to standardize the scale
    tIBI /= sampling_rate

    good = np.nonzero((tIBI < th1) & (tIBI > th2))[0]
    tIBI = tIBI[good]   # list of rr-intervall in s for detected r-peaks

    nt = len(tIBI)
    if nt == 0:
        tIBIm = np.nan
        tIBIs = np.nan
    elif nt == 1:
        tIBIm = np.mean(tIBI)
        tIBIs = 0.
    else:
        tIBIm = np.mean(tIBI)
        tIBIs = np.std(tIBI, ddof=1)

    # plot RR-intervals
    fig = pl.figure(figsize=(20, 15))
    fig.suptitle("RR-intervals")
    pl.subplot(211)
    pl.title("Reference RR-intervals")
    pl.xlabel("Time (s)")
    pl.ylabel("RR-interval (s)")
    pl.plot(np.cumsum(rIBI), rIBI, linewidth=2, label="ref. RR-interval", color="#51A6D8")
    #outlier_low = np.mean(rIBI) - 0.3 * np.mean(rIBI)
    #outlier_high = np.mean(rIBI) + 2 * - 0.3 * np.mean(rIBI)
    #pl.axhline(y=outlier_low)
    #pl.axhline(y=outlier_high, label="Threshold boundary")

    pl.subplot(212)
    pl.title("Detected RR-intervals")
    pl.xlabel("Time (s)")
    pl.ylabel("RR-interval (s)")
    pl.plot(np.cumsum(tIBI), tIBI, linewidth=2, label="det. RR-interval", color="#A651D8")
    #outlier_low = np.mean(tIBI) - 0.3 * np.mean(rIBI)
    #outlier_high = np.mean(tIBI) + 2 * - 0.3 * np.mean(rIBI)
    #pl.axhline(y=outlier_low)
    #pl.axhline(y=outlier_high, label="Threshold boundary (RR_mean +/- (30% of RR_mean)")
    pl.show()

    # plot distribution RR-intervals
    fig = pl.figure(figsize=(20, 15))
    fig.suptitle("Distribution of RR-intervals")

    pl.subplot(211)
    pl.title("Distribution of reference RR-intervals")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ignore FutureWarning
        #sns.kdeplot(rIBI, label="rr-intervals", color="#A651D8", shade=True)
        sns.distplot(rIBI, rug=True, hist=True, label="rr-intervals", color="#A651D8")

    outlier_low = np.mean(rIBI) - 2 * np.std(rIBI)
    outlier_high = np.mean(rIBI) + 2 * np.std(rIBI)

    pl.axvline(x=outlier_low)
    pl.axvline(x=outlier_high, label="outlier boundary")
    bottom, top = pl.ylim()
    pl.text(outlier_low - 0.1, (top-bottom)/2, "outliers low (< mean - 2 sigma)")
    pl.text(outlier_high + 0.020, (top-bottom)/2, "outliers high (> mean + 2 sigma)")
    pl.xlabel("RR-interval (s)")
    pl.ylabel("Density")
    pl.legend()

    pl.subplot(212)
    pl.title("Distribution of detected RR-intervals")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ignore FutureWarning
        #sns.kdeplot(tIBI, label="rr-intervals", color="#A651D8", shade=True)
        sns.distplot(tIBI, rug=True, hist=True, label="rr-intervals", color="#A651D8")

    outlier_low = np.mean(tIBI) - 2 * np.std(tIBI)
    outlier_high = np.mean(tIBI) + 2 * np.std(tIBI)

    pl.axvline(x=outlier_low)
    pl.axvline(x=outlier_high, label="outlier boundary")
    bottom, top = pl.ylim()
    pl.text(outlier_low - 0.1, (top-bottom)/2, "outliers low (< mean - 2 sigma)")
    pl.text(outlier_high + 0.02, (top-bottom)/2, "outliers high (> mean + 2 sigma)")
    pl.xlabel("RR-interval (s)")
    pl.ylabel("Density")
    pl.legend()

    pl.show()

    # Interpolation RR-Intervalls
    # create interpolation function based on the rr-samples.
    # reference rr-intervall interpolation
    x = np.cumsum(rIBI)# / sampling_rate
    if np.max(np.cumsum(rIBI)) >= np.max(np.cumsum(tIBI)):
        stop = np.cumsum(tIBI)
    else:
        stop = np.cumsum(rIBI)
    f = interp1d(x, rIBI, kind='cubic')
    # sample rate for interpolation
    fs = 1 / (np.min(rIBI)/2)
    #fs = 4.0
    steps = 1 / fs

    # now we can sample from interpolation function
    ref_xx = np.arange(1, np.max(stop), steps)
    ref_rr_interpolated = f(ref_xx)

    fig = pl.figure(figsize=(20, 15))
    pl.title("Interpolation of reference RR-Intervalls")
    pl.subplot(211)
    pl.title("RR intervals")
    pl.plot(x, rIBI, color="k", markerfacecolor="#A651D8", markeredgewidth=0, marker="o", markersize=8)
    pl.xlabel("Time (s)")
    pl.ylabel("RR-interval (s)")
    pl.title("Interpolated")

    pl.subplot(212)
    pl.title("RR-Intervals (cubic interpolation)")
    pl.plot(ref_xx, ref_rr_interpolated, color="k", markerfacecolor="#51A6D8", markeredgewidth=0, marker="o", markersize=8)
    pl.xlabel("Time (s)")
    pl.ylabel("RR-interval (s)")
    pl.show()

    # detected rr-interval interpolation
    x = np.cumsum(tIBI)  # / sampling_rate
    f = interp1d(x, tIBI, kind='cubic')
    # sample rate for interpolation
    #fs = 4.0
    steps = 1 / fs

    # now we can sample from interpolation function
    det_xx = np.arange(1, np.max(stop), steps)
    det_rr_interpolated = f(det_xx)

    fig = pl.figure(figsize=(20, 15))
    fig.suptitle('Detected interpolated signals')
    pl.title("Interpolation of detected RR-Intervalls")
    pl.subplot(211)
    pl.title("RR intervals")
    pl.plot(x, tIBI, color="k", markerfacecolor="#A651D8", markeredgewidth=0, marker="o", markersize=8)
    pl.xlabel("Time (s)")
    pl.ylabel("RR-interval (s)")
    pl.title("Interpolated")

    pl.subplot(212)
    pl.title("RR-Intervals (cubic interpolation)")
    pl.plot(det_xx, det_rr_interpolated, color="k", markerfacecolor="#51A6D8", markeredgewidth=0, marker="o", markersize=8)
    pl.xlabel("Time (s)")
    pl.ylabel("RR-interval (s)")
    pl.show()

    fig = pl.figure(figsize=(20, 15))
    fig.suptitle('Derivation interpolated signals')
    pl.subplot(211)
    dev_rr_interpol = np.abs(ref_rr_interpolated - det_rr_interpolated)
    pl.title("RR-Intervals deviation (cubic interpolation)")
    pl.plot(det_xx, dev_rr_interpol, color="k", markerfacecolor="#51A6D8", markeredgewidth=0, marker="o",
            markersize=8)
    pl.xlabel("Time (s)")
    pl.ylabel("RR-interval (s)")

    pl.subplot(212)
    pl.title("Distribution of deviated interpolated signals")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ignore FutureWarning
        # sns.kdeplot(rIBI, label="rr-intervals", color="#A651D8", shade=True)
        sns.distplot(dev_rr_interpol, rug=True, hist=True, label="rr-intervals", color="#A651D8")

    outlier_high = np.mean(dev_rr_interpol) + 2 * np.std(dev_rr_interpol)

    pl.axvline(x=outlier_high, label="outlier boundary")
    bottom, top = pl.ylim()
    pl.text(outlier_high + 0.015, (top-bottom)/2, "outliers high (> mean + 2 sigma)")
    pl.xlabel("RR-interval (s)")
    pl.ylabel("Density")
    pl.legend()

    pl.show()

    figure(figsize=(25, 25))
    import neurokit2 as nk
    hrv_indices = nk.hrv(test, sampling_rate=sampling_rate, show=True)

    # output
    perf = float(TP) / len(reference)
    acc = float(TP) / (TP + FP) # Positive Predictive Value (PPV) = TP / TP + FP
    err = float(FP) / (TP + FP)
    if TP and FN != 0:
        sens = float(TP) / (TP + FN)  # Sensitivity (Sens) = TP / TP + FN
    else:
        sens = 100
    if TN and FP != 0:
        spec = float(TN) / (TN + FP)  # Specificity (Spec) = TN / TN + FP
    else:
        spec = 100
    dev = dev.tolist()

    args = (TP, FP, TN, FN, perf, acc, err, sens, spec, detectionIdx, matchIdx, dev, mdev, sdev, rIBIm, rIBIs,
            tIBIm, tIBIs)
    names = ('TP', 'FP', 'TN', 'FN', 'performance', 'acc', 'err', 'sens', 'spec', 'detect', 'match', 'deviation',
             'mean_deviation', 'std_deviation', 'mean_ref_ibi', 'std_ref_ibi',
             'mean_test_ibi', 'std_test_ibi')

    return utils.ReturnTuple(args, names)
