import neurokit2 as nk
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from preprocess_ludb import read_ludb_files
import numpy as np
import pandas as pd
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 25)
np.set_printoptions(linewidth=400)
import seaborn as sns
plt.rcParams['figure.figsize'] = [30, 25]  # Bigger images
plt.rcParams['font.size']= 14


# load ECG data from mit-bih database into pandas dataframes
#data, anno, metadata = read_mit_bit_files()

# load ECG data from ludb database into pandas dataframes
data, anno, metadata = read_ludb_files()
#print(data["ECG"][:30000])

# Preprocess the data (filter, find peaks, etc.)
# pantompkins1985', 'engzeemod2012', 'christov2004'
processed_data, info = nk.ecg_process(ecg_signal=data["ECG"][:30000], sampling_rate=int(metadata["Sampling Frequency"]), method="pantompkins1985")
print(processed_data.head)

# Visualise the processing
plt.rcParams['figure.figsize'] = [15, 9]  # Bigger images
nk.ecg_plot(processed_data, sampling_rate=int(metadata["Sampling Frequency"]), show_type="default")
plt.show()
plt.rcParams['figure.figsize'] = [40, 40]  # Bigger images
nk.ecg_plot(processed_data, sampling_rate=int(metadata["Sampling Frequency"]), show_type="artifacts")
plt.show()

# Compute relevant features
results = nk.ecg_analyze(data=processed_data, sampling_rate=int(metadata["Sampling Frequency"]))
print(results)

# Compute HRV indices
plt.rcParams['figure.figsize'] = [25, 20]  # Bigger images
hrv_indices = nk.hrv(peaks=processed_data["ECG_R_Peaks"], sampling_rate=int(metadata["Sampling Frequency"]), show=True)
print(hrv_indices)
plt.show()

# Delineate
plt.rcParams['figure.figsize'] = [25, 20]  # Bigger images processed_data["ECG_Clean"]
signal, waves = nk.ecg_delineate(ecg_cleaned=processed_data["ECG_Clean"].to_numpy(), rpeaks=info["ECG_R_Peaks"], sampling_rate=int(metadata["Sampling Frequency"]), method="dwt", show=True, show_type='all')
plt.show()

# Distort the signal (add noise, linear trend, artifacts etc.)
distorted = nk.signal_distort(signal=data["ECG"][:30000].to_numpy(),
                              noise_amplitude=0.1,
                              noise_frequency=[5, 10, 20],
                              powerline_amplitude=0.05,
                              artifacts_amplitude=0.3,
                              artifacts_number=3,
                              linear_drift=0.5)

# Clean (filter and detrend)
cleaned = nk.signal_detrend(distorted)
cleaned = nk.signal_filter(cleaned, lowcut=0.5, highcut=1.5)

# Compare the 3 signals
plt.rcParams['figure.figsize'] = [15, 10]  # Bigger images
plot = nk.signal_plot([data["ECG"][:30000].to_numpy(), distorted, cleaned], sampling_rate=int(metadata["Sampling Frequency"]), subplots=True, standardize=True, labels=["Raw Signal", "Distorted Signal", "Cleaned Signal"])
plt.show()

# Find optimal time delay, embedding dimension and r
#plt.rcParams['figure.figsize'] = [25, 20]  # Bigger images
#parameters = nk.complexity_optimize(signal=data["ECG"].to_numpy(), show=True) #[:150000].to_numpy()
#plt.show()

# Detrended Fluctuation Analysis
sample_entropy = nk.entropy_sample(signal=data["ECG"][:30000].to_numpy())
approximate_entropy = nk.entropy_approximate(signal=data["ECG"][:30000].to_numpy())
print(sample_entropy, approximate_entropy)

# Decompose signal using Empirical Mode Decomposition (EMD)
#plt.rcParams['figure.figsize'] = [25, 20]  # Bigger images
#components = nk.signal_decompose(signal=data["ECG"][:30000].to_numpy(), method='emd')
# Visualize components
#nk.signal_plot(components=components)
#plt.show()

# Recompose merging correlated components
#plt.rcParams['figure.figsize'] = [25, 20]  # Bigger images
#recomposed = nk.signal_recompose(components=components, threshold=0.99)
# Visualize components
#nk.signal_plot(recomposed=recomposed)
#plt.show()

# Get the Signal Power Spectrum Density (PSD) using different methods
plt.rcParams['figure.figsize'] = [25, 20]  # Bigger images
welch = nk.signal_psd(signal=data["ECG"][:30000].to_numpy(), method="welch", min_frequency=1, max_frequency=20, show=True)
plt.show()
multitaper = nk.signal_psd(signal=data["ECG"][:30000].to_numpy(), method="multitapers", max_frequency=20, show=True)
lomb = nk.signal_psd(signal=data["ECG"][:30000].to_numpy(), method="lomb", min_frequency=1, max_frequency=20, show=True)
burg = nk.signal_psd(signal=data["ECG"][:30000].to_numpy(), method="burg", min_frequency=1, max_frequency=20, order=10, show=True)
plt.show()

# Highest Density Interval (HDI)
plt.rcParams['figure.figsize'] = [25, 20]  # Bigger images
ci_min, ci_max = nk.hdi(x=processed_data["ECG_Clean"], ci=0.95, show=True)
plt.show()

# Find events
events = nk.events_find(event_channel=data["ECG"][:30000].to_numpy())

# Plot the location of event with the signals
plot = nk.events_plot(events, data["ECG"][:30000].to_numpy())
plt.show()

# Build and plot epochs
epochs = nk.epochs_create(processed_data, events, sampling_rate=int(metadata["Sampling Frequency"]), epochs_start=-1, epochs_end=6)

for i, epoch in enumerate (epochs):
    epoch = epochs[epoch]  # iterate epochs",

    epoch = epoch[['ECG_Clean', 'ECG_Rate']]  # Select relevant columns",

    nk.standardize(epoch).plot(legend=True)  # Plot scaled signals"
    plt.show()

# Extract Event Related Features
# With these segments, we are able to compare how the physiological signals vary across
# the different events. We do this by:
# 1. Iterating through our object epochs
# 2. Storing the mean value of :math:`X` feature of each condition in a new dictionary
# 3. Saving the results in a readable format
# We can call them epochs-dictionary, the mean-dictionary and our results-dataframe
df = {}  # Initialize an empty dict,

for epoch_index in epochs:
    df[epoch_index] = {}  # then Initialize an empty dict inside of it with the iterative

    # Save a temp var with dictionary called <epoch_index> in epochs-dictionary
    epoch = epochs[epoch_index]

    # We want its features:
    # Feature 1 ECG
    ecg_baseline = epoch["ECG_Rate"].loc[-100:0].mean()  # Baseline
    ecg_mean = epoch["ECG_Rate"].loc[0:30000].mean()  # Mean heart rate in the 0-4 seconds
    # Store ECG in df
    df[epoch_index]["ECG_Rate"] = ecg_mean - ecg_baseline  # Correct for baseline


df = pd.DataFrame.from_dict(df, orient="index")  # Convert to a dataframe
print(df)  # Print DataFrame

# You can now plot and compare how these features differ according to the event of interest.
sns.boxplot(y="ECG_Rate", data=df)
plt.show()

# Half the data
#epochs = nk.epochs_create(data["ECG"][:30000], events=[0, 15000], sampling_rate=int(metadata["Sampling Frequency"]), epochs_start=0, epochs_end=150)
#print(epochs)
# Analyze
#ECG_features = nk.ecg_intervalrelated(epochs)
#print(ECG_features)

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(anno["Rpeaks"][:59], info["ECG_R_Peaks"])
TN = confusion_matrix[0][0]
FN = confusion_matrix[1][0]
TP = confusion_matrix[1][1]
FP = confusion_matrix[0][1]

# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP)
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)

# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)
