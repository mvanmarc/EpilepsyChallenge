import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
import h5py
import numpy as np
import tensorflow as tf
from neureka_models import UNet1D
# Import some utilities from the training folder
import sys
sys.path.insert(0, './irregulars-neureka-codebase/training/3-DNN/')
from irregulars_neureka_codebase.training.DNN.utils import build_windowfree_unet, setup_tf
sys.path.insert(0, './irregulars-neureka-codebase/')
from irregulars_neureka_codebase.library import nedc


wpath = '/users/sista/kkontras/Documents/Epilepsy_Challenge/irregulars_neureka_codebase/evaluate/attention_unet_raw.h5'
# wpath = '/users/sista/kkontras/Documents/Epilepsy_Challenge/irregulars_neureka_codebase/evaluate/attention_unet_wiener.h5'
# wpath = '/users/sista/kkontras/Documents/Epilepsy_Challenge/irregulars_neureka_codebase/evaluate/attention_unet_iclabel.h5'
# wpath = ""
# Load TensorFlow Model
# tf_model = tf.keras.models.load_model(wpath)

unet_raw = build_windowfree_unet(n_channels=18, n_filters=8)
unet_raw.load_weights(wpath)
tf_weights = unet_raw.get_weights()


# Print layer names to match with PyTorch
for layer in unet_raw.layers:
    print(layer.name, [w.shape for w in layer.get_weights()])

import torch

torch_model = UNet1D()

with h5py.File(wpath, "r") as f:
    tf_layers = list(f.keys())

    def set_weights(torch_layer, tf_weight_name):
        """Helper function to assign weights"""
        weight = np.array(f[tf_weight_name][tf_layers[0]])  # Extract weight
        bias = np.array(f[tf_weight_name][tf_layers[1]])  # Extract bias

        if len(weight.shape) == 3:  # Conv1D (TensorFlow: [filters, kernel, channels])
            weight = np.transpose(weight, (2, 1, 0))  # PyTorch: [channels, filters, kernel]

        torch_layer.weight.data = torch.tensor(weight, dtype=torch.float32)
        torch_layer.bias.data = torch.tensor(bias, dtype=torch.float32)

    # Assign Weights
    set_weights(torch_model.conv1, "conv2d")
    set_weights(torch_model.conv2, "conv2d_1")
    set_weights(torch_model.conv3, "conv2d_2")
    set_weights(torch_model.conv4, "conv2d_3")
    set_weights(torch_model.conv5, "conv2d_4")
    set_weights(torch_model.conv6, "conv2d_5")
    set_weights(torch_model.conv7, "conv2d_6")

    set_weights(torch_model.deconv1, "conv2d_7")
    set_weights(torch_model.deconv2, "conv2d_8")
    set_weights(torch_model.deconv3, "conv2d_9")
    set_weights(torch_model.deconv4, "conv2d_10")

    set_weights(torch_model.final_conv, "conv2d_11")

    set_weights(torch_model.final_conv, "final_conv")

# Save PyTorch Model
torch.save(torch_model.state_dict(), "converted_model.pth")