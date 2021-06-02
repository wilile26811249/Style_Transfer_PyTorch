import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU


class ConvLayer(nn.Module):
    """
    Input shape:  (B, C, H, W)
    Output shape: (B, C, H, W)
    """
    def __init__(self, in_dim, out_dim, kernel, stride, norm = "instance"):
        super(ConvLayer, self).__init__()
        # Padding Layers
        padding_size = kernel // 2
        self.reflection_pad = nn.ReflectionPad2d(padding_size)

        # Convolution Layer
        self.conv_layer = nn.Conv2d(in_dim, out_dim, kernel, stride)

        # InstanceNorm is for single image, BatchNorm is for batch images
        self.norm_type = norm
        if self.norm_type == "instance":
            self.norm_layer = nn.InstanceNorm2d(out_dim, affine = True)
        elif self.norm_type == "batch":
            self.norm_layer = nn.BatchNorm2d(out_dim, affine = True)

    def forward(self, x):
        x = self.reflection_pad(x)
        x = self.conv_layer(x)
        if self.norm_type == "None":
            out = x
        else:
            out = self.norm_layer(x)
        return out


class DeconvLayer(nn.Module):
    def __init__(self, in_dim, out_dim, kernel, stride, out_pad, norm = "instance"):
        super().__init__()
        # Padding Layers
        padding_size = kernel // 2

        # Transposed Convolution layer
        self.deconv = nn.ConvTranspose2d(in_dim, out_dim, kernel, stride, padding_size, out_pad)

        # InstanceNorm is for single image, BatchNorm is for batch images
        self.norm_type = norm
        if self.norm_type == "instance":
            self.norm_layer = nn.InstanceNorm2d(out_dim, affine = True)
        elif self.norm_type == "batch":
            self.norm_layer = nn.BatchNorm2d(out_dim, affine = True)

    def forward(self, x):
        x = self.deconv(x)
        if self.norm_type == "None":
            out = x
        else:
            out = self.norm_layer(x)
        return out


class ResidualLayer(nn.Module):
    def __init__(self, channels = 128, kernel = 3):
        super().__init__()
        self.first_conv = ConvLayer(channels, channels, kernel, stride = 1)
        self.relu = nn.ReLU()
        self.second_conv = ConvLayer(channels, channels, kernel, stride = 1)

    def forward(self, x):
        identity = x
        out = self.first_conv(x)
        out = self.relu(out)
        out = self.second_conv(out)

        # Residual operation
        out = out + identity
        return out


class TransformNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block = nn.Sequential(
            ConvLayer(3, 32, 9, 1),
            nn.ReLU(),
            ConvLayer(32, 64, 3, 2),
            nn.ReLU(),
            ConvLayer(64, 128, 3, 2),
            nn.ReLU()
        )
        self.residual_block = nn.Sequential(
            ResidualLayer(128, 3),
            ResidualLayer(128, 3),
            ResidualLayer(128, 3),
            ResidualLayer(128, 3),
            ResidualLayer(128, 3)
        )
        self.deconv_block = nn.Sequential(
            DeconvLayer(128, 64, 3, 2, 1),
            nn.ReLU(),
            DeconvLayer(64, 32, 3, 2, 1),
            nn.ReLU(),
            ConvLayer(32, 3, 9, 1)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.residual_block(x)
        x = self.deconv_block(x)
        return x

def test():
    model = TransformNet()
    img = torch.randn(1, 3, 256, 256)
    out = model(img)

    if img.shape == out.shape:
        print("Success")
    else:
        print("Failed")
