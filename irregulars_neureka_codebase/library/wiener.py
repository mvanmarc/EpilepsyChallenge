import numpy as np
import pickle
import torch
import torch.nn.functional as F

filters = None
with open('./filters.pickle', 'rb') as handle:
    filters = pickle.load(handle)

def wiener_filter(data, v):
    """Apply Wiener filter.

    Args:
        data: data contained in an array (row = channels, column = samples)
        v: Wiener filter
    Return:
        out: filtered data
    """
    lag = int(v.shape[0]/data.shape[0])
    filtered_np = list()
    for j in range(v.shape[1]):
        v_shaped = np.reshape(v[:,j], (data.shape[0], lag))
        out = np.convolve(v_shaped[0, :], data[0, :], 'full')
        for i in range(1, v_shaped.shape[0]):
            out += np.convolve(v_shaped[i, :], data[i, :], 'full')
        filtered_np.append(out)
    t = np.arange(0, v.shape[0], step=lag, dtype=int)
    filtered_np = np.array(filtered_np)
    final_filtered_np = np.dot(v[t,:], filtered_np)

    return np.array(final_filtered_np[:,:data.shape[1]])

#can you turn this in pytorch

def wiener_filter_pytorch(data, v):

    batch, channels, samples = data.shape
    lag = v.shape[0] // channels  # Compute lag size

    # Reshape v into (num_filters, channels, lag) for convolution
    v_shaped = v.view(channels, lag, -1).permute(2, 0, 1)  # Shape (num_filters, channels, lag)
    conv_out = F.conv1d(data, v_shaped, padding=int((lag - 1)/2))
    t = torch.arange(0, v.shape[0], step=lag, dtype=torch.long, device=data.device)
    filtered = torch.matmul(v[t, :], conv_out)  # Apply Wiener transformation

    return filtered[:, :, :samples]

    # batch, channels, samples = data.shape
    # lag = v.shape[0] // channels  # Compute lag size
    # v_shaped = v.view(channels, lag, -1).permute(2, 0, 1)  # Shape (num_filters, channels, lag)
    # conv_out = F.conv1d(data, v_shaped, padding=int((lag - 1)/2))  # Shape (batch, num_filters, convolved_samples)
    # t = torch.arange(0, v.shape[0], step=lag, dtype=torch.long)
    # filtered = torch.matmul(v[t, :], conv_out)  # Matmul across selected time indices
    # return filtered[:, :, :samples]  # Ensure the output matches input shape


if __name__ == '__main__':
    data_np = np.random.randn(18, 4098)  # NumPy data
    data_torch = torch.tensor(data_np, dtype=torch.float32).unsqueeze(dim=0)  # PyTorch data (same values)
    output_np = wiener_filter(data_np, np.array(filters[0]))
    output_torch = wiener_filter_pytorch(data_torch, torch.tensor(np.array(filters[0]), dtype=torch.float32))
