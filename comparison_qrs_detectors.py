# Main function file
from .mit_bih_download import read_mit_bit_files
import numpy as np
from biosppy.signals import ecg

def main():
    """Main function - print quick test example
    """
    print("Example Quick Test")
    # Quick test
    # download ECG data from mit-bit database
    read_mit_bit_files()
    # load raw ECG signal
    signal = np.loadcsv('.data/ECGs.csv')

    # process it and plot
    out = ecg.ecg(signal=signal, sampling_rate=1000., show=True)

if __name__ == "__main__":
    main()
