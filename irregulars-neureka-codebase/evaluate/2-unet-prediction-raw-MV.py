"""Generate U-net predictions

Requires pre-processed EDF files similar to the training step
Requires the trained U-net model

Produces a results HDF5 file with predictions for every file
"""


# Libraries
import h5py
import numpy as np
import tensorflow as tf

# Import some utilities from the training folder
import sys
sys.path.insert(0, './irregulars-neureka-codebase/training/3-DNN/')
from utils import build_windowfree_unet, setup_tf
sys.path.insert(0, './irregulars-neureka-codebase/')
from library import nedc

# All relevant files
val_path = '/esat/biomeddata/kkontras/TUH/tuh_eeg/tuh_eeg_seizure/v2.0.3/edf/train/aaaaapks/s012_2014/01_tcp_ar/aaaaapks_s012_t001.h5' # Pre-processed data file
saved_predictions = './irregulars-neureka-codebase/evaluate/evaluation/prediction_test_raw.h5' # File to store the prediction
network_path = './irregulars-neureka-codebase/evaluate/attention_unet_raw.h5' # Path to trained weights

# Data settings
fs = 200
n_channels = 18
n_filters = 8

# Tensorflow function to detect GPU properly
setup_tf()

# Loading the signals
with h5py.File(val_path, 'r') as f:
    file_names_test = ['The Dataset']
    signals_test = []

    duration = f['duration'][()]
    assert fs == f['fs'][()]
    events = f['events'][()]
    ch_names = [x.decode() for x in f['channel_names'][()]]
    sig = []
    for ch in ch_names:
        sig.append(f[ch][()])

    # Put signals in a montage
    (signals_ds, _) = nedc.rereferenceMV(sig, [x.upper() for x in ch_names])
    data = np.array(signals_ds).T

    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    
    signals_test.append((data-mean)/(std+1e-8))
        
# Building a windowfree U-net and load trained weights
unet = build_windowfree_unet(n_channels=n_channels, n_filters=n_filters)
unet.load_weights(network_path)

# Predictions using the windowfree U-net on CPU, our GPU ran out of memory
y_probas = []
reduction = 4096//4
with tf.device('gpu:0'):
    for signal in signals_test:
        signal = signal[:len(signal)//reduction*reduction, :]
        print(signal.shape)
        prediction = unet.predict(signal[np.newaxis, :, :, np.newaxis])[0, :, 0, 0]
        y_probas.append(prediction)

# Saving predictions
dt_fl = h5py.vlen_dtype(np.dtype('float32'))
dt_str = h5py.special_dtype(vlen=str)
with h5py.File(saved_predictions, 'w') as f:
    dset_signals = f.create_dataset('signals', (len(file_names_test),), dtype=dt_fl)
    dset_file_names = f.create_dataset('filenames', (len(file_names_test),), dtype=dt_str)
    
    for i in range(len(file_names_test)):
        dset_signals[i] = y_probas[i]
        dset_file_names[i] = file_names_test[i]
