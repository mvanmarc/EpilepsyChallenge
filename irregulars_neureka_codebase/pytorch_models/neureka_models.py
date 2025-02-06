import torch
import torch.nn as nn
import torch.nn.functional as F


class BiasedConv(nn.Module):
    def __init__(self, filters, kernel_size, strides=(1, 1), padding=0, activation=None, use_bias=True,
                 kernel_initializer='glorot_uniform', bias_initializer='zeros'):
        super(BiasedConv, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.use_bias = use_bias
        self.bias_initializer = bias_initializer
        self.kernel_initializer = kernel_initializer

        # Initialize the convolution layer
        self.conv = nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=kernel_size,
                              stride=strides, padding=padding, bias=False)  # no bias in conv

        # Add custom bias if required
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(filters))  # Default to zeros for bias

            if self.bias_initializer == 'ones':
                self.bias.data.fill_(1)  # Bias initializer as ones

    def forward(self, x):
        # Apply convolution operation
        x = self.conv(x)

        # Manually add the bias
        if self.use_bias:
            x += self.bias.view(1, self.filters, 1, 1)  # Expand bias to match the shape of output (N, filters, H, W)

        # Apply activation function if specified
        if self.activation:
            x = self.activation(x)

        return x

class AttentionPooling(nn.Module):
    def __init__(self, input_filters, filters, channels=18):
        super(AttentionPooling, self).__init__()
        self.filters = filters
        self.channels = channels

        # Convolutional layers for the attention mechanism
        self.att_q = nn.Conv2d(in_channels=input_filters[0], out_channels=filters,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.att_k = nn.Conv2d(in_channels=input_filters[1], out_channels=filters,
                               kernel_size=1, stride=1, padding=0, bias=False)

        self.gate = BiasedConv(filters=self.filters, kernel_size=(1, 1), strides=(1, 1),
                          padding=0, activation=F.sigmoid,
                          kernel_initializer='zeros', bias_initializer='ones')

        # Final attention convolution
        self.att = nn.Conv2d(in_channels=filters, out_channels=1, kernel_size=(1, 1), stride=1, padding=0, bias=False)
        nn.init.ones_(self.att.weight)  # Initialize the weights to ones

        # Average Pooling layer
        self.pool = nn.AvgPool2d(kernel_size=(1, channels), padding=0)

    def forward(self, inputs):
        query, value = inputs

        # Apply convolutions to get the query and key
        att_q = self.att_q(query)
        att_k = self.att_k(value)

        # Compute the gate by adding the query and key
        gated_output = F.sigmoid(self.gate(att_q + att_k))  # Apply Sigmoid to the gate

        # Apply the attention weights
        att = F.sigmoid(self.att(gated_output))  # Attention weights

        # Apply average pooling
        output = self.pool(att * value)

        return output


class UNet1D(nn.Module):
    def __init__(self, window_size, n_channels=18, n_filters=8):
        super(UNet1D, self).__init__()

        self.window_size = window_size

        # Encoding Path
        self.conv0 = nn.Conv2d(1, n_filters, (15, 1), padding=(7, 0))
        self.bn0 = nn.BatchNorm2d(n_filters)
        self.maxpool0 = nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1), padding=(1, 0))

        self.conv1 = nn.Conv2d(n_filters, 2*n_filters, (15, 1), padding=(7, 0))
        self.bn1 = nn.BatchNorm2d(2*n_filters)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1), padding=(1, 0))

        self.conv2 = nn.Conv2d(2*n_filters, 4*n_filters, (15, 1), padding=(7, 0))
        self.bn2 = nn.BatchNorm2d(4*n_filters)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1), padding=(1, 0))

        self.conv3 = nn.Conv2d(4*n_filters, 4*n_filters, (15, 1), padding=(7, 0))
        self.bn3 = nn.BatchNorm2d(4*n_filters)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1), padding=(1, 0))

        self.conv4 = nn.Conv2d(4*n_filters, 8*n_filters, (7, 1), padding=(3, 0))
        self.bn4 = nn.BatchNorm2d(8*n_filters)
        self.maxpool4 = nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1), padding=(1, 0))

        self.conv5 = nn.Conv2d(8*n_filters, 8*n_filters, (3, 1), padding=(1, 0))
        self.bn5 = nn.BatchNorm2d(8*n_filters)
        self.maxpool5 = nn.MaxPool2d(kernel_size=(1, n_channels), stride=(1, n_channels), padding=0)

        self.conv6 = nn.Conv2d(8*n_filters, 4*n_filters, (3, 1), padding=(1, 0))
        self.bn6 = nn.BatchNorm2d(4*n_filters)
        self.dropout = nn.Dropout(0.5)

        self.conv7 = nn.Conv2d(4*n_filters, 4*n_filters, (3, 1), padding=(1, 0))
        self.bn7 = nn.BatchNorm2d(4*n_filters)
        self.dropout1 = nn.Dropout(0.5)

        self.conv_out5 = nn.Conv2d(4*n_filters,1, (3, 1), stride=(1, 1), bias=False)
        self.conv_out4 = nn.Conv2d(8*n_filters,1, (3, 1), stride=(1, 1), bias=False)
        self.conv_out3 = nn.Conv2d(4*n_filters,1, (7, 1), stride=(1, 1), bias=False)
        self.conv_out2 = nn.Conv2d(4*n_filters,1, (15, 1), stride=(1, 1), bias=False)
        self.conv_out1 = nn.Conv2d(4*n_filters,1, (15, 1), stride=(1, 1), bias=False)


        # Dropout Layers
        self.dropout2 = nn.Dropout(0.5)

        self.upsample1 = nn.Upsample(scale_factor=(4,1), mode='nearest')
        self.upsample2 = nn.Upsample(scale_factor=(4,1), mode='nearest')  # up_sampling2d_2
        self.upsample3 = nn.Upsample(scale_factor=(4,1), mode='nearest')  # up_sampling2d_2
        self.upsample4 = nn.Upsample(scale_factor=(4,1), mode='nearest')  # up_sampling2d_2

        self.att0 = AttentionPooling(input_filters=[4*n_filters,8*n_filters], filters=4*n_filters, channels=n_channels)
        self.att1 = AttentionPooling(input_filters=[4*n_filters,8*n_filters], filters=4*n_filters, channels=n_channels)
        self.att2 = AttentionPooling(input_filters=[4*n_filters,8*n_filters], filters=4*n_filters, channels=n_channels)
        self.att3 = AttentionPooling(input_filters=[4*n_filters,8*n_filters], filters=4*n_filters, channels=n_channels)
        self.att4 = AttentionPooling(input_filters=[4*n_filters,8*n_filters], filters=4*n_filters, channels=n_channels)

        self.conv8 = nn.Conv2d(64, 32, (1, 1), bias=False)  # conv2d_9
        self.conv9 = nn.Conv2d(64, 32, (1, 1), bias=False)  # conv2d_9
        self.bias_conv = nn.Conv2d(32, 1, (1, 1), bias=False)  # biased_conv

        self.conv10 = nn.Conv2d(32, 1, (1, 1))  # conv2d_10
        self.avg_pool = nn.AvgPool2d((2, 1))  # equivalent to average_pooling2d

        self.conv11 = nn.Conv2d(96, 32, (3, 1))  # conv2d_11
        self.bn8 = nn.BatchNorm2d(32)
        self.conv12 = nn.Conv2d(32, 32, (1, 1), bias=False)  # conv2d_12
        self.conv13 = nn.Conv2d(32, 32, (1, 1), bias=False)  # conv2d_13
        self.bias_conv1 = nn.Conv2d(32, 1, (1, 1), bias=False)  # biased_conv_1



    # up_sampling2d []
    # conv2d_8 [(1, 1, 32, 32)]
    # conv2d_9 [(1, 1, 64, 32)]
    # add []
    # biased_conv [(32,)]
    # conv2d_10 [(1, 1, 32, 1), (1,)]
    # multiply []
    # average_pooling2d []
    # concatenate []
    # conv2d_11 [(3, 1, 96, 32), (32,)]
    # batch_normalization_8 [(32,), (32,), (32,), (32,)]
    # elu_8 []
    # up_sampling2d_1 []
    # conv2d_12 [(1, 1, 32, 32)]
    # conv2d_13 [(1, 1, 32, 32)]
    # add_1 []
    # biased_conv_1 [(32,)]
    # conv2d_14 [(1, 1, 32, 1), (1,)]
    # multiply_1 []
    # average_pooling2d_1 []
    # concatenate_1 []
    # conv2d_15 [(7, 1, 64, 32), (32,)]
    # batch_normalization_9 [(32,), (32,), (32,), (32,)]
    # elu_9 []
    # up_sampling2d_2 []
    # conv2d_16 [(1, 1, 32, 32)]
    # conv2d_17 [(1, 1, 32, 32)]
    # add_2 []
    # biased_conv_2 [(32,)]
    # conv2d_18 [(1, 1, 32, 1), (1,)]
    # multiply_2 []
    # average_pooling2d_2 []
    # concatenate_2 []
    # conv2d_19 [(15, 1, 64, 32), (32,)]
    # batch_normalization_10 [(32,), (32,), (32,), (32,)]
    # elu_10 []
    # up_sampling2d_3 []
    # conv2d_20 [(1, 1, 32, 32)]
    # conv2d_21 [(1, 1, 16, 32)]
    # add_3 []
    # biased_conv_3 [(32,)]
    # conv2d_22 [(1, 1, 32, 1), (1,)]
    # multiply_3 []
    # average_pooling2d_3 []
    # concatenate_3 []
    # conv2d_23 [(15, 1, 48, 32), (32,)]
    # batch_normalization_11 [(32,), (32,), (32,), (32,)]
    # elu_11 []
    # up_sampling2d_4 []
    # conv2d_24 [(1, 1, 32, 32)]
    # conv2d_25 [(1, 1, 8, 32)]
    # add_4 []
    # biased_conv_4 [(32,)]
    # conv2d_26 [(1, 1, 32, 1), (1,)]
    # multiply_4 []
    # average_pooling2d_4 []
    # concatenate_4 []
    # conv2d_27 [(15, 1, 40, 32), (32,)]
    # batch_normalization_12 [(32,), (32,), (32,), (32,)]
    # elu_12 []
    # conv2d_28 [(15, 1, 32, 32), (32,)]
    # batch_normalization_13 [(32,), (32,), (32,), (32,)]
    # elu_13 []
    # conv2d_29 [(15, 1, 32, 1), (1,)]
    def forward(self, x):

        lvl0 = F.elu(self.bn0(self.conv0(x)))
        x1_pool = self.maxpool0(lvl0)
        lvl1 = F.elu(self.bn1(self.conv1(x1_pool)))
        x2_pool = self.maxpool1(lvl1)
        lvl2 = F.elu(self.bn2(self.conv2(x2_pool)))
        x3_pool = self.maxpool2(lvl2)
        lvl3 = F.elu(self.bn3(self.conv3(x3_pool)))
        x4_pool = self.maxpool3(lvl3)
        lvl4 = F.elu(self.bn4(self.conv4(x4_pool)))
        x5_pool = self.maxpool4(lvl4)
        lvl5 = F.elu(self.bn5(self.conv5(x5_pool)))
        x5 = self.maxpool5(lvl5)
        x6 = self.dropout(F.elu(self.bn6(self.conv6(x5))))
        x7 = self.dropout(F.elu(self.bn7(self.conv7(x6))))

        out5 = F.sigmoid(self.conv_out5(x7))
        # out5 = out5.view(out5.shape[0], self.window_size//1024)  # Flatten to the required shape (batch_size, window_size//1024)

        up4 = self.upsample4(x7)
        att4 = self.att4([up4, lvl4])  # concatenate

        out4 = F.sigmoid(self.conv_out4(att4))
        # out4 = out4.view(out4.shape[0], self.window_size//256)  # Flatten to the required shape (batch_size, window_size//1024)

        x8 = F.elu(self.bn8(self.conv8(torch.concatenate([up4, att4], dim=1))))

        up3 = self.upsample3(x8)
        att3 = self.att3([up3, lvl3])  # concatenate

        out3 = F.sigmoid(self.conv_out3(att3))
        # out3 = out3.view(out3.shape[0], self.window_size//64)  # Flatten to the required shape (batch_size, window_size//1024)

        x9 = F.elu(self.bn9(self.conv9(torch.concatenate([up3, att3], dim=1))))

        up2 = self.upsample2(x9)
        att2 = self.att2([up2, lvl2])  # concatenate

        out2 = F.sigmoid(self.conv_out2(att2))
        # out2 = out2.view(out2.shape[0], self.window_size//16)  # Flatten to the required shape (batch_size, window_size//1024)

        x10 = F.elu(self.bn10(self.conv10(torch.concatenate([up2, att2], dim=1))))

        up1 = self.upsample1(x10)
        att1 = self.att1([up2, lvl2])  # concatenate

        out1 = F.sigmoid(self.conv_out1(att1))
        # out1 = out1.view(out1.shape[0], self.window_size//4)  # Flatten to the required shape (batch_size, window_size//1024)

        x11 = F.elu(self.bn(self.conv11([up1, att1])))

        up0 = self.upsample0(x11)
        att0 = self.att0([up0, lvl0])
        x12 = F.elu(self.bn(self.conv12([up0, att0])))
        x13 = F.elu(self.bn(self.conv13(x12)))

        out0 = F.sigmoid(self.conv_out0(x13))
        out0 = out1.view(out0.shape[0], self.window_size)  # Flatten to the required shape (batch_size, window_size//1024)


        return [out0, out1, out2, out3, out4, out5 ]


if __name__ == "__main__":
    # Example Usage
    model = UNet1D(window_size=4096)
    x = torch.randn(32, 1, 4096, 18)  # Batch of 8, 1 channel, 4096 features
    output = model(x)
    print(output.shape)  # Expected Output Shape: (8, 1, 4096)
