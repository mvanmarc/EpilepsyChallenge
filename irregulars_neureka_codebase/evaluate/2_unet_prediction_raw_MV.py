"""Generate U-net predictions

Requires pre-processed EDF files similar to the training step
Requires the trained U-net model

Produces a results HDF5 file with predictions for every file
"""


# Libraries
import h5py
import numpy as np
import tensorflow as tf
from irregulars_neureka_codebase.Dataloader.tuh_dataloader import TUH_Dataloader
import sys
from irregulars_neureka_codebase.training.DNN.utils import build_windowfree_unet, setup_tf
from tqdm import tqdm
from easydict import EasyDict


setup_tf()

config = EasyDict()
config.dataset = EasyDict()
config.dataset.window_size = 4096
config.dataset.stride = 4096
config.dataset.data_path = "/esat/biomeddata/kkontras/TUH/tuh_eeg/tuh_eeg_seizure/v2.0.3/TUH.h5"
config.training_params = EasyDict()
config.training_params.len_sample = 4096 * 30
config.training_params.fs = 200
config.training_params.batch_size = 32
config.training_params.test_batch_size = 32
config.training_params.pin_memory = False
config.training_params.num_workers = 6
config.training_params.seed = 0
config.model = EasyDict()
config.model.n_channels = 18
config.model.n_filters = 8
config.model.pre_dir = '/users/sista/kkontras/Documents/Epilepsy_Challenge/irregulars_neureka_codebase/evaluate/attention_unet_raw.h5'
config.model.save_preds = '/users/sista/kkontras/Documents/Epilepsy_Challenge/irregulars_neureka_codebase/evaluate/prediction_test_raw.h5'

dl = TUH_Dataloader(config)
unet = build_windowfree_unet(n_channels=config.model.n_channels, n_filters=config.model.n_filters)
unet.load_weights(config.model.pre_dir)

preds, labels = [], []
with tf.device('gpu:0'):
    for i, batch in tqdm(enumerate(dl.valid_loader), total=len(dl.valid_loader)):
        data = batch['data']['raw']
        label = batch['label']
        prediction = unet.predict(data.unsqueeze(dim=0).unsqueeze(dim=-1).numpy())
        #        prediction = unet.predict(signal[np.newaxis, :, :, np.newaxis])[0, :, 0, 0]

        preds.append(prediction)
        labels.append(label)
        break

# Saving predictions
dt_fl = h5py.vlen_dtype(np.dtype('float32'))
dt_str = h5py.special_dtype(vlen=str)
with h5py.File(config.model.save_preds, 'w') as f:
    dset_signals = f.create_dataset('signals', (len(file_names_test),), dtype=dt_fl)
    dset_file_names = f.create_dataset('filenames', (len(file_names_test),), dtype=dt_str)
    
    for i in range(len(file_names_test)):
        dset_signals[i] = preds[i]
        dset_file_names[i] = file_names_test[i]
