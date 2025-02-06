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
import einops
import torch
from keras.models import load_model

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
config.model.pre_dir_raw = '/users/sista/kkontras/Documents/Epilepsy_Challenge/irregulars_neureka_codebase/evaluate/attention_unet_raw.h5'
config.model.pre_dir_wiener = '/users/sista/kkontras/Documents/Epilepsy_Challenge/irregulars_neureka_codebase/evaluate/attention_unet_wiener.h5'
config.model.pre_dir_iclabel = '/users/sista/kkontras/Documents/Epilepsy_Challenge/irregulars_neureka_codebase/evaluate/attention_unet_iclabel.h5'
config.model.pre_dir_lstm = '/users/sista/kkontras/Documents/Epilepsy_Challenge/irregulars_neureka_codebase/evaluate/model-dnn-dnnw-dnnicalbl-lstm-4.h5'
config.model.save_preds = '/users/sista/kkontras/Documents/Epilepsy_Challenge/irregulars_neureka_codebase/evaluate/prediction_test_raw.h5'

dl = TUH_Dataloader(config)

unet_raw = build_windowfree_unet(n_channels=config.model.n_channels, n_filters=config.model.n_filters)
unet_raw.load_weights(config.model.pre_dir_raw)

unet_wiener = build_windowfree_unet(n_channels=config.model.n_channels, n_filters=config.model.n_filters)
unet_wiener.load_weights(config.model.pre_dir_wiener)

unet_iclabel = build_windowfree_unet(n_channels=config.model.n_channels, n_filters=config.model.n_filters)
unet_iclabel.load_weights(config.model.pre_dir_iclabel)

lstm_fusion_model = load_model(config.model.pre_dir_lstm)
# print("Parameters loaded.")


# test(model, modeltype, classifiers, filenames)
#
# f_nick = dict()
# for classifier in classifiers:
#     if classifier['format'] == 'nick':
#         f_nick[classifier['name']] = h5py.File(classifier['file'], 'r')
#
# # Predict probabilities
# results = list()
# for i, filename in enumerate(filenames):
#     x = prepare_file(i, filename, classifiers, f_nick, modeltype)
#     print(f_nick)
#     u = model.predict(x, batch_size=1)
#     model.reset_states()
#     results.append(u)
#
# with open('./irregulars-neureka-codebase/evaluate/evaluation/lstm-results.pkl', 'wb') as filehandler:
#     pickle.dump(results, filehandler)
#
# # Close Nick data
# for key in f_nick:
#     f_nick[key].close()

agg_features, labels = [], []
with tf.device('gpu:0'):
    for i, batch in tqdm(enumerate(dl.valid_loader), total=len(dl.valid_loader)):
        data = batch['data']['raw']
        label = batch['label']
        data = einops.rearrange(data, "b c t -> b t c").unsqueeze(dim=-1)
        features_raw = torch.from_numpy(unet_raw.predict(data.numpy())).squeeze().unsqueeze(dim=0)
        # break
        features_wiener = torch.from_numpy(unet_wiener.predict(data.numpy())).squeeze().unsqueeze(dim=0)
        features_iclabel = torch.from_numpy(unet_iclabel.predict(data.numpy())).squeeze().unsqueeze(dim=0)
        features = torch.concatenate([features_raw, features_wiener, features_iclabel], dim=0).unsqueeze(dim=0)
        agg_features.append(features)
        labels.append(label)
        u = lstm_fusion_model.predict(features.numpy(), batch_size=1)
        lstm_fusion_model.reset_states()
        # print(u.shape)

        # if i==2:
        break

# preds = torch.concatenate(preds)
# labels = torch.concatenate(labels)
#
# print(preds.shape)
# print(labels.shape)

# # Saving predictions
# dt_fl = h5py.vlen_dtype(np.dtype('float32'))
# dt_str = h5py.special_dtype(vlen=str)
# with h5py.File(config.model.save_preds, 'w') as f:
#     dset_signals = f.create_dataset('signals', (len(file_names_test),), dtype=dt_fl)
#     dset_file_names = f.create_dataset('filenames', (len(file_names_test),), dtype=dt_str)
#
#     for i in range(len(file_names_test)):
#         dset_signals[i] = preds[i]
#         dset_file_names[i] = file_names_test[i]
