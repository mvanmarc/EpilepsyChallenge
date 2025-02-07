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
                              stride=strides, padding=padding, bias=True)  # no bias in conv

        nn.init.ones_(self.conv.weight)  # Initialize the weights to ones
        nn.init.zeros_(self.conv.bias)  # Initialize the weights to ones

    def forward(self, x):
        # Apply convolution operation
        x = self.conv(x)

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

        self.gate = nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=(1, 1),
                              stride=(1, 1), padding=0, bias=True)  # no bias in conv

        nn.init.ones_(self.gate.weight)  # Initialize the weights to ones
        self.gate.weight.requires_grad = False
        nn.init.zeros_(self.gate.bias)  # Initialize the weights to ones

        # Final attention convolution
        self.att = nn.Conv2d(in_channels=filters, out_channels=1, kernel_size=(1, 1), stride=1, padding=0, bias=True)

        nn.init.ones_(self.att.weight)  # Initialize the weights to ones
        nn.init.zeros_(self.att.bias)  # Initialize the weights to ones

        # Average Pooling layer
        self.pool = nn.AvgPool2d(kernel_size=(1, channels), padding=0)

    def forward(self, query, value):

        # Apply convolutions to get the query and key*
        att_q = self.att_q(query)
        att_k = self.att_k(value)

        # Compute the gate by adding the query and key
        gated_output = torch.sigmoid(self.gate(att_q + att_k))  # Apply Sigmoid to the gate

        # Apply the attention weights
        att = torch.sigmoid(self.att(gated_output))  # Attention weights

        # Apply average pooling
        output = self.pool(att * value)

        return output


class UNet1D(nn.Module):
    def __init__(self, window_size, n_channels=18, n_filters=8):
        super(UNet1D, self).__init__()

        self.window_size = window_size

        # Encoding Path
        self.conv = nn.Conv2d(1, n_filters, (15, 1), padding=(7, 0))
        self.bn = nn.BatchNorm2d(n_filters)
        self.maxpool0 = nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1), padding=(1, 0))

        self.conv1 = nn.Conv2d(n_filters, 2*n_filters, (15, 1), padding=(7, 0))
        self.bn1 = nn.BatchNorm2d(2*n_filters)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1), padding=(1, 0))

        self.conv2 = nn.Conv2d(2*n_filters, 4*n_filters, (15, 1), padding=(7, 0))
        self.bn2 = nn.BatchNorm2d(4*n_filters)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1), padding=(1, 0))

        self.conv3 = nn.Conv2d(4*n_filters, 4*n_filters, (7, 1), padding=(3, 0))
        self.bn3 = nn.BatchNorm2d(4*n_filters)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1), padding=(1, 0))

        self.conv4 = nn.Conv2d(4*n_filters, 8*n_filters, (3, 1), padding=(1, 0))
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

        # self.conv_out5 = nn.Conv2d(4*n_filters,1, (3, 1), stride=(1, 1), padding=(1,0), bias=True)
        self.upsample4 = nn.Upsample(scale_factor=(4,1), mode='nearest')  # up_sampling2d_2
        self.att4 = AttentionPooling(input_filters=[4*n_filters,8*n_filters], filters=4*n_filters, channels=n_channels)
        self.conv8 = nn.Conv2d(4*n_filters + 8*n_filters, 4*n_filters, (3, 1), padding=(1,0))  # conv2d_9
        self.bn8 = nn.BatchNorm2d(4*n_filters)


        # self.conv_out4 = nn.Conv2d(8*n_filters,1, (3, 1), stride=(1, 1), padding=(1,0), bias=True)
        self.upsample3 = nn.Upsample(scale_factor=(4,1), mode='nearest')  # up_sampling2d_2
        self.att3 = AttentionPooling(input_filters=[4*n_filters,4*n_filters], filters=4*n_filters, channels=n_channels)
        self.conv9 = nn.Conv2d(8*n_filters, 4*n_filters, (7, 1), padding=(3,0))  # conv2d_9
        self.bn9 = nn.BatchNorm2d(4*n_filters)


        # self.conv_out3 = nn.Conv2d(4*n_filters,1, (7, 1), stride=(1, 1), padding=(3,0), bias=True)
        self.upsample2 = nn.Upsample(scale_factor=(4,1), mode='nearest')  # up_sampling2d_2
        self.att2 = AttentionPooling(input_filters=[4*n_filters,4*n_filters], filters=4*n_filters, channels=n_channels)
        self.conv10 = nn.Conv2d(8*n_filters, 4*n_filters, (15, 1), padding=(7,0))  # conv2d_9
        self.bn10 = nn.BatchNorm2d(4*n_filters)


        # self.conv_out2 = nn.Conv2d(4*n_filters,1, (15, 1), stride=(1, 1), padding=(7,0), bias=True)
        self.upsample1 = nn.Upsample(scale_factor=(4,1), mode='nearest')
        self.att1 = AttentionPooling(input_filters=[4*n_filters,2*n_filters], filters=4*n_filters, channels=n_channels)
        self.conv11 = nn.Conv2d(6*n_filters, 4*n_filters, (15, 1), padding=(7,0))  # conv2d_9
        self.bn11 = nn.BatchNorm2d(4*n_filters)

        # self.conv_out1 = nn.Conv2d(2*n_filters,1, (15, 1), stride=(1, 1), padding=(7,0), bias=True)
        self.upsample0 = nn.Upsample(scale_factor=(4,1), mode='nearest')
        self.att0 = AttentionPooling(input_filters=[4*n_filters,n_filters], filters=4*n_filters, channels=n_channels)
        self.conv12 = nn.Conv2d(5 * n_filters, 4 * n_filters, (15, 1), padding=(7, 0))  # conv2d_9
        self.bn12 = nn.BatchNorm2d(4 * n_filters)
        self.conv13 = nn.Conv2d(4 * n_filters, 4 * n_filters, (15, 1), padding=(7, 0))  # conv2d_9
        self.bn13 = nn.BatchNorm2d(4 * n_filters)


        self.conv_out0 = nn.Conv2d(4*n_filters,1, (15, 1), stride=(1, 1), padding=(7,0), bias=True)


        # Dropout Layers
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):

        lvl0 = F.elu(self.bn(self.conv(x)))
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

        # out5 = F.sigmoid(self.conv_out5(x7))
        # out5 = out5.view(out5.shape[0], self.window_size//1024)  # Flatten to the required shape (batch_size, window_size//1024)

        up4 = self.upsample4(x7)
        att4 = self.att4(up4, lvl4)  # concatenate

        # out4 = F.sigmoid(self.conv_out4(att4))
        # out4 = out4.view(out4.shape[0], self.window_size//256)  # Flatten to the required shape (batch_size, window_size//1024)

        #concat should give [96 1]
        x8 = F.elu(self.bn8(self.conv8(torch.cat([up4, att4], dim=1))))

        up3 = self.upsample3(x8)
        att3 = self.att3(up3, lvl3)  #concatenate

        # out3 = F.sigmoid(self.conv_out3(att3))
        # out3 = out3.view(out3.shape[0], self.window_size//64)  # Flatten to the required shape (batch_size, window_size//1024)
        #concat should give [64 1]
        x9 = F.elu(self.bn9(self.conv9(torch.cat([up3, att3], dim=1))))

        up2 = self.upsample2(x9)
        att2 = self.att2(up2, lvl2)  # concatenate

        # out2 = F.sigmoid(self.conv_out2(att2))
        # out2 = out2.view(out2.shape[0], self.window_size//16)  # Flatten to the required shape (batch_size, window_size//1024)

        #concat should give [64 1]
        x10 = F.elu(self.bn10(self.conv10(torch.cat([up2, att2], dim=1))))

        up1 = self.upsample1(x10)
        att1 = self.att1(up1, lvl1)  # concatenate
        # out1 = F.sigmoid(self.conv_out1(att1))
        # out1 = out1.view(out1.shape[0], self.window_size//4)  # Flatten to the required shape (batch_size, window_size//1024)
        #concat should give [48 1]
        x11 = F.elu(self.bn11(self.conv11(torch.cat([up1, att1], dim=1))))

        up0 = self.upsample0(x11)
        att0 = self.att0(up0, lvl0)
        #concat should give [40 1]
        x12 = F.elu(self.bn12(self.conv12(torch.cat([up0, att0], dim=1))))
        x13 = F.elu(self.bn13(self.conv13(x12)))

        out0 = F.sigmoid(self.conv_out0(x13))
        out0 = out0.view(out0.shape[0], self.window_size)  # Flatten to the required shape (batch_size, window_size//1024)

        # return [out0, out1, out2, out3, out4, out5 ]
        return out0

if __name__ == "__main__":
    # Example Usage
    model = UNet1D(window_size=4096)
    x = torch.randn(32, 1, 4096, 18)  # Batch of 8, 1 channel, 4096 features
    output = model(x)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)
    # for out_i in output:
    #     print(out_i.shape)
    from torchsummary import summary
    summary(model.cuda(), (1, 4096, 18))

    total_tf = [  128, 32,   1936, 64,   7712, 128,   7200, 128,   6208, 256,
             12352, 256,   6176, 128,   3104, 128,    1024, 2048,  32, 33,
             9248, 128,   1024, 1024,  32, 33,    14368, 128,   1024,
             1024,  32, 33,    30752, 128,   1024, 512,  32, 33,
             23072, 128,   1024, 256,  32, 33,    19232, 128,  15392,
             128,  481, 241, 481, 225, 193, 97]

    # print([  128, 32,   1936, 64,   7712, 128,   7200, 128,   6208, 256,
    #          12352, 256,   6176, 128,   3104, 128,    1024, 2048,  32, 33,
    #          9248, 128,   1024, 1024,  32, 33,    14368, 128,   1024,
    #          1024,  32, 33,    30752, 128,   1024, 512,  32, 33,
    #          23072, 128,   1024, 256,  32, 33,    19232, 128,  15392,
    #          128,  481, 241, 481, 225, 193, 97])

    print(total_tf)
    import numpy as np
    print(np.array(total_tf).sum())
    print([p.numel() for p in model.parameters()])

    print(np.array([p.numel() for p in model.parameters()]).sum())
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(name, p.shape)
    # print(output.shape)  # Expected Output Shape: (8, 1, 4096)
