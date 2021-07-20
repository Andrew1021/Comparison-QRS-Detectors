
def save_mit_hdf5(df_ecg, dfs_rpeaks, name=mit_processed_data_path):
    # Save as hdf5 file
    with h5py.File(name, 'w') as hf:
        # Group for formatted ECG database
        ecg = hf.create_group('ECG')
        # Sub-Group for each ECG data and annotations
        data = ecg.create_group('Data')
        anno = ecg.create_group('Annotations')
        # Add datasets for samples and R-Peak annotations
        ecg_hdf5 = data.create_dataset("Samples", data=df_ecg, compression="gzip", compression_opts=4, chunks=True)
        ecg_hdf5 = anno.create_dataset("Rpeaks", data=dfs_rpeaks, compression="gzip", compression_opts=4,
                                      chunks=True)
        # Annotate metadata to hdf5 file
        metadata = {'User': 'Andreas Roth',
                    'Date': time.time(),
                    'OS': os.name,
                    'Database': 'MIT-BIH Arrhythmia Database (MIT-BIT)',
                    'Suspects': '47',
                    'Record': '4000',
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
        hf.attrs.update(metadata)
    print('Saved to hdf5 file {0} with status {1}'.format(name, hf.status_code))

def open_mit_hdf5(name=mit_processed_data_path):
    # Open hdf5 file
    with h5py.File(name, 'r') as hf:
        print(get_mit_keyes(hf))
        print(hf['ECG/Samples'])
        ecg_dset = hf['ECG/Data/Samples']  # Dataset pointed to ssd memory, gone when hf closed
        ecg = ecg_dset  # numpy/pandas array
        rpeaks_dset = hf['ECG/Annotations/Rpeaks']  # Dataset pointed to ssd memory, gone when hf closed
        rpeaks = rpeaks_dset  # numpy/pandas array
    print('Open hdf5 file {0} with status {1}'.format(name, hf.status_code))
    return ecg, rpeaks

# return keyes of hdf5 file
def get_mit_keyes(hf):
    keyes = []
    for key in hf.keys():
        # print(key)
        keyes.append(key)
    return keyes

def get_mit_hdf5(name=mit_processed_data_path):

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
