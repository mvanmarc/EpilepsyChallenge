import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
import h5py
import numpy as np
import tensorflow as tf
from neureka_models import UNet1D, LSTM_Neureka
from keras.models import load_model

# Import some utilities from the training folder
import sys
sys.path.insert(0, './irregulars-neureka-codebase/training/3-DNN/')
from irregulars_neureka_codebase.training.DNN.utils import build_windowfree_unet, setup_tf, build_unet
sys.path.insert(0, './irregulars-neureka-codebase/')
from irregulars_neureka_codebase.library import nedc


# wpath = '/users/sista/kkontras/Documents/Epilepsy_Challenge/irregulars_neureka_codebase/evaluate/attention_unet_raw.h5'
# wpath = '/users/sista/kkontras/Documents/Epilepsy_Challenge/irregulars_neureka_codebase/evaluate/attention_unet_wiener.h5'
# wpath = '/users/sista/kkontras/Documents/Epilepsy_Challenge/irregulars_neureka_codebase/evaluate/attention_unet_iclabel.h5'
wpath = '/users/sista/kkontras/Documents/Epilepsy_Challenge/irregulars_neureka_codebase/evaluate/model-dnn-dnnw-dnnicalbl-lstm-4.h5'
# wpath = ""
# Load TensorFlow Model
# tf_model = tf.keras.models.load_model(wpath)

# unet_raw = build_unet(n_channels=18, n_filters=8)[0]
unet_raw = load_model(wpath)
# unet_raw.load_weights(wpath)
tf_weights = unet_raw.get_weights()
import einops
import torch
from collections import defaultdict

# torch_model = UNet1D(4096)
torch_model = LSTM_Neureka()

# model_parts = defaultdict(lambda: 0)
# for name, p in torch_model.named_parameters():
#     model_parts[name.split(".")[0]] += p.numel()
# for k, v in dict(model_parts).items():
#     print("{}: {}".format(k,v))


correspondence_dict = {
"conv_out0" : "conv2d_34",
"bn13" : "batch_normalization_13",
"conv13" : "conv2d_33",
"bn12" : "batch_normalization_12",
"conv12" : "conv2d_32",
"att0.att" : "conv2d_31",
"att0.gate" : "conv2d_33",
"att0.att_q" : "conv2d_29",
"att0.att_k" : "conv2d_30",
"bn11" : "batch_normalization_11",
"conv11" : "conv2d_28",
"att1.att_q" : "conv2d_24",
"att1.att_k" : "conv2d_25",
"att1.gate" : "biased_conv_3",
"att1.att" : "conv2d_26",
"bn10" : "batch_normalization_10",
"conv10" : "conv2d_23",
"att2.att_q" : "conv2d_19",
"att2.att_k" : "conv2d_20",
"att2.gate" : "biased_conv_2",
"att2.att" : "conv2d_21",
"bn9" : "batch_normalization_9",
"conv9" : "conv2d_18",
"att3.att_q":"conv2d_14",
"att3.att_k":"conv2d_15",
"att3.gate" : "biased_conv_1",
"att3.att" : "conv2d_16",
"bn8" : "batch_normalization_8",
"conv8" : "conv2d_13",
"att4.att_q" : "conv2d_9",
"att4.att_k" : "conv2d_10",
"att4.gate" : "biased_conv",
"att4.att" : "conv2d_11",
"bn7" : "batch_normalization_7",
"conv7" : "conv2d_7",
"bn6" : "batch_normalization_6",
"conv6" : "conv2d_6",
"bn5" : "batch_normalization_5",
"conv5" : "conv2d_5",
"bn4" : "batch_normalization_4",
"conv4" : "conv2d_4",
"bn3" : "batch_normalization_3",
"conv3" : "conv2d_3",
"bn2" : "batch_normalization_2",
"conv2" : "conv2d_2",
"bn1" : "batch_normalization_1",
"conv1" : "conv2d_1",
"bn" : "batch_normalization",
"conv" : "conv2d"
}
correspondence_dict = {
"lstm" : "bidirectional_1",
"dense" : "dense_1"
}


def set_weights(torch_layer, tf_weight_name, f, tf_layers):
    """Helper function to assign weights from TensorFlow to PyTorch"""


    if "model_weights" in f:
        f = f["model_weights"]
    print(f[tf_weight_name][tf_weight_name].keys())
    print(tf_weight_name)

    if "bias:0" in f[tf_weight_name][tf_weight_name].keys():
        bias = np.array(f[tf_weight_name][tf_weight_name]["bias:0"])
        torch_layer.bias.data = torch.tensor(bias, dtype=torch.float32)

    if "kernel:0" in f[tf_weight_name][tf_weight_name].keys():
        weight = np.array(f[tf_weight_name][tf_weight_name]["kernel:0"])
        # Ensure we adjust for dimension mismatch between TF and PyTorch (transpose)
        if len(weight.shape) == 4:
            weight = einops.rearrange(weight, "a b c d -> d c a b")
        if len(weight.shape) == 2 and tf_weight_name == "dense_1":
            weight = einops.rearrange(weight, "a b -> b a")
        if torch_layer.weight.data.shape != weight.shape:
            raise ValueError(f"Dimension mismatch: {torch_layer.weight.data.shape} != {weight.shape}")
        torch_layer.weight.data = torch.tensor(weight, dtype=torch.float32)

    if "gamma:0" in f[tf_weight_name][tf_weight_name].keys():
        gamma = np.array(f[tf_weight_name][tf_weight_name]["gamma:0"])
        torch_layer.weight.data = torch.tensor(gamma, dtype=torch.float32)
    if "beta:0" in f[tf_weight_name][tf_weight_name].keys():
        beta = np.array(f[tf_weight_name][tf_weight_name]["beta:0"])
        torch_layer.bias.data = torch.tensor(beta, dtype=torch.float32)
    if "moving_mean:0" in f[tf_weight_name][tf_weight_name].keys():
        moving_mean = np.array(f[tf_weight_name][tf_weight_name]["moving_mean:0"])
        torch_layer.running_mean = torch.tensor(moving_mean, dtype=torch.float32)
    if "moving_variance:0" in f[tf_weight_name][tf_weight_name].keys():
        moving_variance = np.array(f[tf_weight_name][tf_weight_name]["moving_variance:0"])
        torch_layer.running_var = torch.tensor(moving_variance, dtype=torch.float32)

    if "backward_lstm_1" in f[tf_weight_name][tf_weight_name].keys():
        backward_lstm_kernel = np.array(f[tf_weight_name][tf_weight_name]["backward_lstm_1"]["kernel:0"], dtype=np.float32)
        backward_lstm_recurrent_kernel = np.array(f[tf_weight_name][tf_weight_name]["backward_lstm_1"]["recurrent_kernel:0"], dtype=np.float32)
        backward_lstm_bias = np.array(f[tf_weight_name][tf_weight_name]["backward_lstm_1"]["bias:0"], dtype=np.float32)
        torch_layer.weight_ih_l0 = torch.nn.Parameter(torch.tensor(einops.rearrange(backward_lstm_kernel, "a b -> b a"), dtype=torch.float32))
        torch_layer.weight_hh_l0 = torch.nn.Parameter(torch.tensor(einops.rearrange(backward_lstm_recurrent_kernel, "a b -> b a"), dtype=torch.float32))
        torch_layer.bias_ih_l0 = torch.nn.Parameter(torch.tensor(backward_lstm_bias, dtype=torch.float32))
        torch_layer.bias_hh_l0 = torch.nn.Parameter(torch.tensor(backward_lstm_bias, dtype=torch.float32))

    if "forward_lstm_1" in f[tf_weight_name][tf_weight_name].keys():
        forward_lstm_kernel = np.array(f[tf_weight_name][tf_weight_name]["forward_lstm_1"]["kernel:0"], dtype=np.float32)
        forward_lstm_recurrent_kernel = np.array(f[tf_weight_name][tf_weight_name]["forward_lstm_1"]["recurrent_kernel:0"], dtype=np.float32)
        forward_lstm_bias = np.array(f[tf_weight_name][tf_weight_name]["forward_lstm_1"]["bias:0"], dtype=np.float32)
        torch_layer.weight_ih_l0 = torch.nn.Parameter(torch.tensor(einops.rearrange(forward_lstm_kernel, "a b -> b a"), dtype=torch.float32))
        torch_layer.weight_hh_l0 = torch.nn.Parameter(torch.tensor(einops.rearrange(forward_lstm_recurrent_kernel, "a b -> b a"), dtype=torch.float32))
        torch_layer.bias_ih_l0 = torch.nn.Parameter(torch.tensor(forward_lstm_bias, dtype=torch.float32))
        torch_layer.bias_hh_l0 = torch.nn.Parameter(torch.tensor(forward_lstm_bias, dtype=torch.float32))

def get_nested_attr(model, attr_path):
    """Recursively retrieve nested attributes (layers) in PyTorch model."""
    attrs = attr_path.split(".")  # Split "att0.att" into ["att0", "att"]
    if len(attrs) > 1:
        print(attrs)
    for attr in attrs:
        if hasattr(model, attr):
            model = getattr(model, attr)  # Move deeper in the hierarchy
        else:
            return None  # Layer does not exist
    return model  # Return the final layer


# Open the TensorFlow model file
with h5py.File(wpath, "r") as f:
    tf_layers = list(f.keys())  # List of layers in the TensorFlow model

    # To keep track of layers we've processed
    processed_layers = set()

    # Iterate over the correspondence dictionary and assign weights
    for pytorch_name, tensorflow_name in correspondence_dict.items():
        try:
            torch_layer = get_nested_attr(torch_model, pytorch_name)
            set_weights(torch_layer, tensorflow_name, f, tf_layers)
            processed_layers.add(pytorch_name)  # Mark this PyTorch layer as processed
            print(f"Layer {pytorch_name} initialized successfully.")
        except Exception as e:
            print(f"Warning: Layer {pytorch_name} not found in PyTorch model.")

    # Check if all TensorFlow layers are accounted for in the PyTorch model
    missing_layers = set(correspondence_dict.keys()) - processed_layers
    if missing_layers:
        print(f"Warning: The following layers were not initialized in the PyTorch model: {missing_layers}")
    else:
        print("All layers have been initialized successfully.")

    uninitialized_params = []
    for name, param in torch_model.named_parameters():
        if param.requires_grad and param.data.abs().sum() == 0:
            uninitialized_params.append(name)
    # Report unnitialized parameters
    if uninitialized_params:
        print(f"Warning: The following parameters were not initialized:\n{uninitialized_params}")
    else:
        print("All PyTorch model parameters have been initialized successfully!")

# Save the converted PyTorch model
# torch.save(torch_model.state_dict(), "neureka_pytorch_raw.pth")
# torch.save(torch_model.state_dict(), "neureka_pytorch_wiener.pth")
# torch.save(torch_model.state_dict(), "neureka_pytorch_iclabel.pth")
torch.save(torch_model.state_dict(), "neureka_pytorch_lstm.pth")
