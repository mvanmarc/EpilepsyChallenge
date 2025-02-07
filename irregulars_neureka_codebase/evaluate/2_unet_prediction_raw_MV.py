"""Generate U-net predictions

Requires pre-processed EDF files similar to the training step
Requires the trained U-net model

Produces a results HDF5 file with predictions for every file
"""


# Libraries
import h5py
import numpy as np
# import tensorflow as tf
from irregulars_neureka_codebase.Dataloader.tuh_dataloader import TUH_Dataloader
import sys
# from irregulars_neureka_codebase.training.DNN.utils import build_windowfree_unet, setup_tf, build_unet
from tqdm import tqdm
from easydict import EasyDict
import einops
import torch
# from keras.models import load_model
from irregulars_neureka_codebase.pytorch_models.neureka_models import NeurekaNet
# setup_tf()

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
config.model.window_size = 4096
config.model.n_channels = 18
config.model.n_filters = 8
config.model.tf_pre_dir_raw = '/users/sista/kkontras/Documents/Epilepsy_Challenge/irregulars_neureka_codebase/evaluate/attention_unet_raw.h5'
config.model.tf_pre_dir_wiener = '/users/sista/kkontras/Documents/Epilepsy_Challenge/irregulars_neureka_codebase/evaluate/attention_unet_wiener.h5'
config.model.tf_pre_dir_iclabel = '/users/sista/kkontras/Documents/Epilepsy_Challenge/irregulars_neureka_codebase/evaluate/attention_unet_iclabel.h5'
config.model.tf_pre_dir_lstm = '/users/sista/kkontras/Documents/Epilepsy_Challenge/irregulars_neureka_codebase/evaluate/model-dnn-dnnw-dnnicalbl-lstm-4.h5'

config.model.pre_dir_raw = '/users/sista/kkontras/Documents/Epilepsy_Challenge/irregulars_neureka_codebase/pytorch_models/neureka_pytorch_raw.pth'
config.model.pre_dir_wiener = '/users/sista/kkontras/Documents/Epilepsy_Challenge/irregulars_neureka_codebase/pytorch_models/neureka_pytorch_wiener.pth'
config.model.pre_dir_iclabel = '/users/sista/kkontras/Documents/Epilepsy_Challenge/irregulars_neureka_codebase/pytorch_models/neureka_pytorch_iclabel.pth'
config.model.pre_dir_lstm = '/users/sista/kkontras/Documents/Epilepsy_Challenge/irregulars_neureka_codebase/pytorch_models/neureka_pytorch_lstm.pth'

config.model.save_preds = '/users/sista/kkontras/Documents/Epilepsy_Challenge/irregulars_neureka_codebase/evaluate/prediction_test_raw.h5'

dl = TUH_Dataloader(config)

# unet_raw = build_unet(n_channels=config.model.n_channels, n_filters=config.model.n_filters)[0]
# unet_raw.load_weights(config.model.pre_dir_raw)
#
# unet_wiener = build_windowfree_unet(n_channels=config.model.n_channels, n_filters=config.model.n_filters)
# unet_wiener.load_weights(config.model.pre_dir_wiener)
#
# unet_iclabel = build_windowfree_unet(n_channels=config.model.n_channels, n_filters=config.model.n_filters)
# unet_iclabel.load_weights(config.model.pre_dir_iclabel)

# lstm_fusion_model = load_model(config.model.pre_dir_lstm)
# lstm_fusion_model.summary()

pytorch_net = NeurekaNet(config)
pytorch_net.cuda()
pytorch_net.eval()


agg_features, labels = [], []

for i, batch in tqdm(enumerate(dl.valid_loader), total=len(dl.valid_loader)):

    data = batch['data']['raw']
    label = batch['label']
    data = einops.rearrange(data, "b c t -> b t c").unsqueeze(dim=1).cuda().float()

    pred = pytorch_net(data)

    print(pred.shape)

    break

#
# with tf.device('gpu:0'):
#     for i, batch in tqdm(enumerate(dl.valid_loader), total=len(dl.valid_loader)):
#         data = batch['data']['raw']
#         label = batch['label']
#         # data = einops.rearrange(data, "b c t -> b t c").unsqueeze(dim=-1)
#         # features_raw = torch.from_numpy(unet_raw.predict(data.numpy())).squeeze().unsqueeze(dim=0)
#         # break
#         features_raw = pytorch_unet_raw(data.unsqueeze(dim=-1))
#         features_wiener = pytorch_unet_wiener(data.unsqueeze(dim=-1))
#         features_iclabel = pytorch_unet_iclabel(data.unsqueeze(dim=-1))
#         # print(features_raw- output)
#         # print(features_raw- output)
#         break
#         features_wiener = torch.from_numpy(unet_wiener.predict(data.numpy())).squeeze().unsqueeze(dim=0)
#         features_iclabel = torch.from_numpy(unet_iclabel.predict(data.numpy())).squeeze().unsqueeze(dim=0)
#         features = torch.concatenate([features_raw, features_wiener, features_iclabel], dim=0).unsqueeze(dim=0)
#         agg_features.append(features)
#         labels.append(label)
#         u = lstm_fusion_model.predict(features.numpy(), batch_size=1)
#         lstm_fusion_model.reset_states()
#         # print(u.shape)
#
#         # if i==2:
#         break

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
