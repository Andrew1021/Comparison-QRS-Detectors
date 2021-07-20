# 3rd party
import neurokit2 as nk
from neurokit2 import signal_period
import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.rcParams['figure.figsize'] = [30, 25]  # Bigger images
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 25)
np.set_printoptions(linewidth=400)

# local
from preprocess_mit import read_mit_bit_files
from preprocess_ludb import read_ludb_files


def pantompkins1985(ecg, sampling_rate):
    signal, info = nk.ecg_peaks(ecg, sampling_rate=sampling_rate, method="pantompkins1985")
    return info["ECG_R_Peaks"]

def engzeemod2012(ecg, sampling_rate):
    signal, info = nk.ecg_peaks(ecg, sampling_rate=sampling_rate, method="engzeemod2012")
    return info["ECG_R_Peaks"]

def christov2004(ecg, sampling_rate):
    signal, info = nk.ecg_peaks(ecg, sampling_rate=sampling_rate, method="christov2004")
    return info["ECG_R_Peaks"]

def pantompkins1985_correction(ecg, sampling_rate):
    signal, info = nk.ecg_peaks(ecg, sampling_rate=sampling_rate, method="pantompkins1985", correct_artifacts=True)
    return info["ECG_R_Peaks"]

def engzeemod2012_correction(ecg, sampling_rate):
    signal, info = nk.ecg_peaks(ecg, sampling_rate=sampling_rate, method="engzeemod2012", correct_artifacts=True)
    return info["ECG_R_Peaks"]

def christov2004_correction(ecg, sampling_rate):
    signal, info = nk.ecg_peaks(ecg, sampling_rate=sampling_rate, method="christov2004", correct_artifacts=True)
    return info["ECG_R_Peaks"]


def benchmark_ecg_preprocessing(function, ecg, rpeaks=None, sampling_rate=1000, error_correction=False):
    """Benchmark ECG preprocessing pipelines.

    Parameters
    ----------
    function : function
        Must be a Python function which first argument is the ECG signal and which has a
        ``sampling_rate`` argument.
    ecg : pd.DataFrame or str
        The path to a folder where you have an `ECGs.csv` file or directly its loaded DataFrame.
        Such file can be obtained by running THIS SCRIPT (TO COMPLETE).
    rpeaks : pd.DataFrame or str
        The path to a folder where you have an `Rpeaks.csv` fils or directly its loaded DataFrame.
        Such file can be obtained by running THIS SCRIPT (TO COMPLETE).
    sampling_rate : int
        The sampling frequency of `ecg_signal` (in Hz, i.e., samples/second). Only used if ``ecgs``
        and ``rpeaks`` are single vectors.

    Returns
    --------
    pd.DataFrame
        A DataFrame containing the results of the benchmarking


    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> # Define a preprocessing routine
    >>> def function(ecg, sampling_rate):
    >>>     signal, info = nk.ecg_peaks(ecg, method='engzeemod2012', sampling_rate=sampling_rate)
    >>>     return info["ECG_R_Peaks"]
    >>>
    >>> # Synthetic example
    >>> ecg = nk.ecg_simulate(duration=20, sampling_rate=200)
    >>> true_rpeaks = nk.ecg_peaks(ecg, sampling_rate=200)[1]["ECG_R_Peaks"]
    >>>
    >>> nk.benchmark_ecg_preprocessing(function, ecg, true_rpeaks, sampling_rate=200)
    >>>
    >>> # Example using database (commented-out)
    >>> # nk.benchmark_ecg_preprocessing(function, r'path/to/GUDB_database')

    """
    # find data
    if rpeaks is None:
        rpeaks = ecg

    if isinstance(ecg, str):
        ecg = pd.read_csv(ecg + "/ECGs.csv")

    if isinstance(rpeaks, str):
        rpeaks = pd.read_csv(rpeaks + "/Rpeaks.csv")

    if isinstance(ecg, pd.DataFrame):
        results, evaluations = _benchmark_ecg_preprocessing_databases(function, ecg, rpeaks, error_correction=error_correction)
    else:
        results, evaluations = _benchmark_ecg_preprocessing(function, ecg, rpeaks, sampling_rate=sampling_rate, error_correction=error_correction)

    return results, evaluations

# =============================================================================
# Utils
# =============================================================================
def _benchmark_ecg_preprocessing_databases(function, ecgs, rpeaks, error_correction):
    """A wrapper over _benchmark_ecg_preprocessing when the input is a database."""
    # Run algorithms
    results = []
    evaluations = []
    for participant in ecgs["Participant"].unique():
        for database in ecgs[ecgs["Participant"] == participant]["Database"].unique():

            # Extract the right slice of data
            ecg_slice = ecgs[(ecgs["Participant"] == participant) & (ecgs["Database"] == database)]
            rpeaks_slice = rpeaks[(rpeaks["Participant"] == participant) & (rpeaks["Database"] == database)]
            sampling_rate = ecg_slice["Sampling_Rate"].unique()[0]

            # Extract values
            ecg = ecg_slice["ECG"].values
            rpeak = rpeaks_slice["Rpeaks"].values

            # Run benchmark
            result, evaluation = _benchmark_ecg_preprocessing(function, ecg, rpeak, sampling_rate, error_correction)

            # Add info
            result["Participant"] = participant
            result["Database"] = database

            evaluation["Participant"] = participant
            evaluation["Database"] = database

            results.append(result)
            evaluations.append(evaluation)

    return pd.concat(results), pd.concat(evaluations)

def _benchmark_ecg_preprocessing(function, ecg, rpeak, sampling_rate=1000, error_correction=False):
    # Apply function
    t0 = datetime.datetime.now()
    try:
        if not error_correction:
            found_rpeaks = function(ecg, sampling_rate=sampling_rate)
        else:
            found_rpeaks = function(ecg, sampling_rate=sampling_rate)
        duration = (datetime.datetime.now() - t0).total_seconds()
    # In case of failure
    except Exception as error:  # pylint: disable=broad-except
        return pd.DataFrame(
            {
                "Sampling_Rate": [sampling_rate],
                "Duration": [np.nan],
                "Score": [np.nan],
                "Recording_Length": [len(ecg) / sampling_rate / 60],
                "Error": str(error),
            }
        )

    # Compare R peaks
    score, error = benchmark_ecg_compareRpeaks(rpeak, found_rpeaks, sampling_rate=sampling_rate)

    evaluation = evaluate_qrs_detectors(ecg, rpeak, found_rpeaks, sampling_rate, tolerance=0.05, show_plot=False)

    return pd.DataFrame(
        {
            "Sampling_Rate": [sampling_rate],
            "Duration": [duration],
            "Score": [score],
            "Recording_Length": [len(ecg) / sampling_rate / 60],
            "Error": error,
        },
    ), evaluation

# =============================================================================
# Comparison methods
# =============================================================================
def benchmark_ecg_compareRpeaks(true_rpeaks, found_rpeaks, sampling_rate=250):
    # Failure to find sufficient R-peaks
    if len(found_rpeaks) <= 3:
        return np.nan, "R-peaks detected <= 3"

    length = np.max(np.concatenate([true_rpeaks, found_rpeaks]))

    true_interpolated = signal_period(
        true_rpeaks, sampling_rate=sampling_rate, desired_length=length, interpolation_method="linear"
    )
    found_interpolated = signal_period(
        found_rpeaks, sampling_rate=sampling_rate, desired_length=length, interpolation_method="linear"
    )

    return np.mean(np.abs(found_interpolated - true_interpolated)), "None"


def evaluate_qrs_detectors(signal, reference_rpeaks, detected_rpeaks, sampling_frequency, tolerance, show_plot=False):

    # Threshold for 100ms
    # tol = 0.1  # 100 ms
    # Fs_mit = mit_metadata["Sampling Frequency"]
    # Fs_ludb = ludb_metadata["Sampling Frequency"]
    # N_mit = int((tol * int(Fs_mit)) + 1)  # Number of samples for given time of 100ms and frequency
    # N_ludb = int((tol * int(Fs_ludb)) + 1)  # Number of samples for given time of 100ms and frequency

    def compute_confusion_matrix_and_delays(frames_detections, frames_annotations, tolerance_frames):
        """
        compute the confusion matrix of the evaluation. For each annotation, consider a interval of tolerance around it, and
        check if there is a detection. If Yes : correct detection (TP) and the delays between the corresponding annotation
        and it is measured, if No : missed complex (FN). Every detections which were not in a tolerance interval around an
        annotation is a false detection (FP).
        :param frames_detections: list of QRS detections (localisations) of the chosen algorithm
        :type frames_detections: list(int)
        :param frames_annotations: list of beat annotations (localisations)
        :type frames_annotations: list(int)
        :param tolerance_frames: number of frames corresponding to the value of the tolerance in milliseconds
        :type tolerance_frames: int
        :return: list of calculated criteria and the list of delays between annotations and their corresponding correct
        detections
        :rtype: tuple(list(int),list(int))
        """
        true_pos = 0
        false_neg = 0
        delays = []
        for fr in frames_annotations:
            interval = range(fr - tolerance_frames, fr + tolerance_frames + 1)
            corresponding_detections_frame = list(set(interval).intersection(frames_detections))
            if len(corresponding_detections_frame) > 0:
                true_pos += 1
                delays.append(corresponding_detections_frame[0] - fr)
            else:
                false_neg += 1
        false_pos = len(frames_detections) - true_pos
        return [true_pos, false_pos, false_neg], delays

    tolerance = int(tolerance * sampling_frequency) + 1  # Number of samples for given time of 0.05s (50ms) and frequency
    [TP, FP, FN], delays = compute_confusion_matrix_and_delays(reference_rpeaks, detected_rpeaks, tolerance)
    TN = len(signal) - (TP + FP + FN)
    # output metrics
    performance = float(TP) / len(reference_rpeaks) # Test performance = TP / len(reference r-peaks)
    accuracy = float(TP) / (TP + FP)         # Positive Predictive Value (PPV) = TP / TP + FP
    error = float(FP) / (TP + FP)            # Error rate = FP / (TP + FP)
    if TP and FN != 0:
        sensitivity = float(TP) / (TP + FN)  # Sensitivity (Sens) = TP / TP + FN
    else:
        sensitivity = 1.
    if TN and FP != 0:
        specificity = float(TN) / (TN + FP)  # Specificity (Spec) = TN / TN + FP
    else:
        specificity = 1.

    # calculate mean and std. delay from detected to reference peaks
    delays = np.array(delays, dtype='float')
    if len(delays) == 0:
        mdev = np.nan
        sdev = np.nan
    elif len(delays) == 1:
        mdev = np.mean(delays)
        sdev = 0.
    else:
        mdev = np.mean(delays)
        sdev = np.std(delays, ddof=1)

    evaluation = pd.DataFrame({"TP": TP, "FN": FN, "FP": FP, "TN": TN, "Positive Prediction Value": [accuracy],
                               "Sensitivity": [sensitivity], "Specificity": [specificity], "Mean Error (ms)": [mdev],
                               "Std. Mean Error (ms)": [sdev]})

    if show_plot:
        # Confusion matrix
        cf_matrix = np.array([[TP, FN], [FP, TN]])
        group_names = ['True Pos', 'False Neg', 'False Pos', 'True Neg']
        group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
        group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
        labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
        labels = np.asarray(labels).reshape(2, 2)
        sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
        plt.show()
        # PPV, Sens, Sec, mean error boxplot
        #sns.barplot(x="Algorithms", y="", data=evaluation[["Positive Prediction Value", "Sensitivity", "Specificity"]])
        #sns.barplot(x="Algorithms", y="", data=evaluation[["Mean Error (s)", "Std. Mean Error"]])
        #plt.show()

    return evaluation

def distort_signal(signal):
    # Distort the signal (add noise, linear trend, artifacts etc.)
    return nk.signal_distort(signal=signal.to_numpy(),
                                  noise_amplitude=0.1,
                                  noise_frequency=[5, 10, 20],
                                  powerline_amplitude=0.05,
                                  artifacts_amplitude=0.3,
                                  artifacts_number=3,
                                  linear_drift=0.5)


### Run the Benchmarking
## Note: This takes a long time (several hours).

# load ECG data and annotations from mit-bih database into pandas dataframes
mit_data, mit_anno, mit_metadata = read_mit_bit_files()

# load ECG data and annotationsfrom ludb database into pandas dataframes
ludb_data, ludb_anno, ludb_metadata = read_ludb_files()

# Concatenate them together
ecgs = [mit_data, ludb_data]
rpeaks = [mit_anno, ludb_anno]
metadatas = [mit_metadata, ludb_metadata]

functions = [pantompkins1985, engzeemod2012, christov2004]
correction_functions = [pantompkins1985_correction, engzeemod2012_correction, christov2004_correction]

results = []
evaluations = []
for method in correction_functions:
    for i in range(1,len(ecgs)):
        if False:
            ecgs[i].loc[:, "ECG"] = distort_signal(ecgs[i]["ECG"])
        ecgs[i].loc[:, "ECG"] = nk.ecg_clean(ecgs[i]["ECG"], sampling_rate=int(metadatas[i]["Sampling Frequency"]), method=method.__name__.replace("_correction", ""))
        result, evaluation = benchmark_ecg_preprocessing(method, ecgs[i], rpeaks[i])
        result["Method"] = method.__name__
        evaluation["Method"] = method.__name__
        results.append(result)
        evaluations.append(evaluation)
results = pd.concat(results).reset_index(drop=True)
evaluations = pd.concat(evaluations).reset_index(drop=True)
print(results)
print(evaluations)
print(results.describe())
print(evaluations.describe())

plt.rcParams['figure.figsize'] = [15, 9]  # Bigger images
# Errors and bugs
# Plot a bar chart
#sns.set_style("dark")
plot = sns.barplot(data=results, x="Method", y="Score", hue="Database", ci=None)  # result[["Duration", "Score", "Error"]]
plt.show()
plot = sns.barplot(data=results, x="Method", y="Duration", hue="Database", ci=None)  # result[["Duration", "Score", "Error"]]
plt.show()

# Confusion matrix cf_matrix
cf_matrix = np.array([[evaluations["TP"].sum(), evaluations["FN"].sum()], [evaluations["FP"].sum(), evaluations["TN"].sum()]])
group_names = ['True Pos', 'False Neg', 'False Pos', 'True Neg']
group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
labels = np.asarray(labels).reshape(2, 2)
sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
plt.show()

plot = sns.barplot(data=evaluations, x="Method", y="Positive Prediction Value", hue="Database", ci=None)  # result[["Duration", "Score", "Error"]]
plt.show()

plot = sns.barplot(data=evaluations, x="Method", y="Sensitivity", hue="Database", ci=None)  # result[["Duration", "Score", "Error"]]
plt.show()

plot = sns.barplot(data=evaluations, x="Method", y="Specificity", hue="Database", ci=None)  # result[["Duration", "Score", "Error"]]
plt.show()

plot = sns.barplot(data=evaluations, x="Method", y="Mean Error (ms)", hue="Database", ci=None)  # result[["Duration", "Score", "Error"]]
plt.show()

plot = sns.barplot(data=evaluations, x="Method", y="Std. Mean Error (ms)", hue="Database", ci=None)  # result[["Duration", "Score", "Error"]]
plt.show()

# Results
methods = ["pantompkins1985", "engzeemod2012", "christov2004"]

color = {"pantompkins1985": "#E91E63", "engzeemod2012": "#4CAF50", "christov2004": "#2196F3"}

# Errors and bugs
#sns.barplot(x=results["Method"], y="Evaluation", hue="Detector",
#            data=[results["Duration"][:199], results["Score"][:199], results["Error"][:199]])




