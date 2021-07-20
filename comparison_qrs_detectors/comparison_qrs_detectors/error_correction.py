import numpy as np
import neurokit2 as nk
import pandas as pd
import matplotlib.pyplot as plt

seg_mask_explanation = [
    'normal',
    'around sleep stage change point',
    'NaN in sleep stage',
    'NaN in signal',
    'overly high/low amplitude',
    'flat signal',
    'NaN in feature',
    'NaN in spectrum',
    'muscle artifact',
    'spurious spectrum',
    'missing or additional R peak']

def addari_error_correction(signal, rpeaks, window_time, step_time, Fs):
    """ Artifact removal for the rr peaks using ADARRI
    Arguments:
        signal -- np.ndarray
        window_time -- in seconds
        Fz -- in Hz

        Keyword arguments:
        notch_freq
        bandpass_freq
        start_end_remove_window_num -- default 0, number of windows removed at the beginning and the end of the signal
        amplitude_thres -- default 1000, mark all segments with np.any(signal_seg>=amplitude_thres)=True
        to_remove_mean -- default False, whether to remove the mean of signal from each channel

        Outputs:
        signal segments -- a list of np.ndarray, each has size=(window_size, channel_num)
        labels --  a list of labels of each window, if none, returns None
        segment start ids -- a list of starting ids of each window in (sample_num,)
        segment masks --
    """
    # Define start, steps, und end of window
    window_time = int(round(len(signal) - 1) / int(Fs))  # duration
    window_size = int(round(window_time * Fs))
    step_size = len(signal)
    start_ids = np.arange(0, signal.shape - window_size + 1, step_size)
    seg_masks = [seg_mask_explanation[0]] * len(start_ids)

    # Artifact removal for the rr peaks using ADARRI
    adarri = np.log1p(np.abs(np.diff(np.diff(rpeaks))))
    artifact_pos = np.where(adarri >= 5)[0] + 2
    rpeak_q10, rpeak_q90 = np.percentile(signal[rpeaks], (10, 90))
    artifact_pos = artifact_pos.tolist() + \
                   np.where((signal[rpeaks] < rpeak_q10 - 300) | (signal[rpeaks] > rpeak_q90 + 300))[0].tolist()
    artifact_pos = np.sort(np.unique(artifact_pos))

    rr_peak_artifact1d = []
    for ap in artifact_pos:
        rr_peak_artifact1d.extend(np.where((start_ids <= rpeaks[ap]) & (rpeaks[ap] < start_ids + window_size))[0])
    rr_peak_artifact1d = np.sort(np.unique(rr_peak_artifact1d))
    for i in rr_peak_artifact1d:
        seg_masks[i] = seg_mask_explanation[10]

    # create binary signal
    signal = np.zeros_like(signal, dtype='double')
    signal[rpeaks] = 1.
    signal_segs = signal[list(map(lambda x: np.arange(x, x + window_size), start_ids))].transpose(1, 0, 2)
    # (#window, #ch, window_size+2padding)

    # mark binary signals with all 0's
    fewbeats2d = signal_segs.sum(axis=2) < 4.5 * 60 / 3  # can't have less than 20 beats/min
    fewbeats1d = np.where(np.any(fewbeats2d, axis=1))[0]
    for i in fewbeats1d:
        seg_masks[i] = seg_mask_explanation[10]

    return signal_segs, start_ids, seg_masks


def nk2_fix_rpeaks(signal, rpeaks, Fs, iterative=True, robust_method=False, method="Kubios", show_plot=False):
    """Correct erroneous peak placements.

        Identify and correct erroneous peak placements based on outliers in peak-to-peak differences (period).

        Parameters
        ----------
        signal : Union[list, np.array, pd.Series]
            The raw ECG channel.
        rpeaks : list or array or DataFrame or Series or dict
            The samples at which the peaks occur. If an array is passed in, it is assumed that it was obtained
            with `signal_findpeaks()`. If a DataFrame is passed in, it is assumed to be obtained with `ecg_findpeaks()`
            or `ppg_findpeaks()` and to be of the same length as the input signal.
        Fs : int
            The sampling frequency of the signal that contains the peaks (in Hz, i.e., samples/second).
        iterative : bool
            Whether or not to apply the artifact correction repeatedly (results in superior artifact correction).
        robust_method : bool
            Use a robust method of standardization (see `standardize()`) for the relative thresholds.
        method : str
            Either "Kubios" or "Neurokit". "Kubios" uses the artifact detection and correction described
            in Lipponen, J. A., & Tarvainen, M. P. (2019). Note that "Kubios" is only meant for peaks in
            ECG or PPG. "neurokit" can be used with peaks in ECG, PPG, or respiratory data.
        show_plot : bool
            Whether or not to visualize artifacts and artifact thresholds.

        Returns
        -------
        peaks_clean : array
            The corrected peak locations.
        artifacts : dict
            Only if method="Kubios". A dictionary containing the indices of artifacts, accessible with the
            keys "ectopic", "missed", "extra", and "longshort".
    """
    # Sanitize input
    signal = nk.signal_sanitize(signal)

    if isinstance(rpeaks, list):
        # ensure numpy
        rpeaks = np.array(rpeaks)

    if isinstance(rpeaks, pd.DataFrame):
        rpeaks = rpeaks.to_numpy()

    # Attempt to retrieve column.
    if isinstance(rpeaks, tuple):
        if isinstance(rpeaks[0], (dict, pd.DataFrame)):
            rpeaks = rpeaks[0]
        elif isinstance(rpeaks[1], dict):
            rpeaks = rpeaks[1]
        else:
            rpeaks = rpeaks[0]

    if isinstance(rpeaks, pd.DataFrame):
        col = [col for col in rpeaks.columns if np.key in col]
        if len(col) == 0:
            raise TypeError(
                "NeuroKit error: _signal_formatpeaks(): wrong type of input ",
                "provided. Please provide indices of peaks.",
            )
        peaks_signal = rpeaks[col[0]].values
        rpeaks = np.where(peaks_signal == 1)[0]

    if isinstance(rpeaks, dict):
        col = [col for col in list(rpeaks.keys()) if np.key in col]
        if len(col) == 0:
            raise TypeError(
                "NeuroKit error: _signal_formatpeaks(): wrong type of input ",
                "provided. Please provide indices of peaks.",
            )
        rpeaks = rpeaks[col[0]]

    # Retrieve length.
    try:  # Detect if single peak or no RR-Interval
        len(rpeaks)
    except TypeError:
        rpeaks = np.array([rpeaks])

    # Detect if two peaks and just one RR-Interval
    if len(rpeaks) == 1:
        raise TypeError("Just two R-Peaks or one RR-Interval detected!")

    # interbeat intervals
    rIBI = np.diff(rpeaks)
    rIBI = np.array(rIBI, dtype='float')

    # Basic resampling to signal sampling rate to standardize the scale
    rIBI /= Fs

    # interbeat interval within thresholds
    th1 = 1.5  # 40 bpm
    th2 = 0.3  # 200 bpm
    good = np.nonzero((rIBI < th1) & (rIBI > th2))[0]
    rIBI = rIBI[good]  # list of rr-intervall in s for annotated r-peaks

    nr = len(rIBI)
    if nr == 0:
        print("No good RR-Interval detected!")
        return
    elif nr == 1:
        print("Just one good RR-Interval detected!")
        return
    else:
        rIBImean = np.mean(rIBI)
        rIBIstd = np.std(rIBI, ddof=1)
        rIBImax = np.max(rIBI)
        rIBImin = np.min(rIBI)
        relIBImax = rIBImean + rIBIstd
        relIBImin = rIBImean - rIBIstd

    peaks_corrected = nk.signal_fixpeaks(rpeaks, sampling_rate=Fs, iterative=iterative, show=show_plot,
                      interval_min=rIBImin, interval_max=rIBImax, relative_interval_min=relIBImin, relative_interval_max=relIBImax,
                      robust=robust_method, method=method)
    plt.show()

    if show_plot == True:
        # Plot and shift original peaks to the right to see the difference.
        plt.rcParams['figure.figsize'] = [15, 9]  # Bigger images
        fig = nk.events_plot([rpeaks + 50, peaks_corrected], signal)
        plt.show()

    return peaks_corrected
