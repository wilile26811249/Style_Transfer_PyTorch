import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_dim, out_dim, stride = 1, padding = 1):
    return nn.Conv2d(in_dim, out_dim, 3, stride, padding)

class vgg16(nn.Module):
    """
    Pretrained weight download (Provide by Justin Johnson):
        https://web.eecs.umich.edu/~justincj/models/vgg16-00b39a1b.pth
    Github page:
        https://github.com/jcjohnson/pytorch-vgg
    """
    def __init__(self):
        super().__init__()
        self.conv1_1= conv3x3(3, 64)
        self.conv1_2 = conv3x3(64, 64)

        self.conv2_1 = conv3x3(64, 128)
        self.conv2_2 = conv3x3(128, 128)

        self.conv3_1 = conv3x3(128, 256)
        self.conv3_2 = conv3x3(256, 256)
        self.conv3_3 = conv3x3(256, 256)

        self.conv4_1 = conv3x3(256, 512)
        self.conv4_2 = conv3x3(512, 512)
        self.conv4_3 = conv3x3(512, 512)

        self.conv5_1 = conv3x3(512, 512)
        self.conv5_2 = conv3x3(512, 512)
        self.conv5_3 = conv3x3(512, 512)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        relu1_2 = x
        x = F.max_pool2d(x, kernel_size =2, stride = 2)

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        relu2_2 = x
        x = F.max_pool2d(x, kernel_size =2, stride = 2)

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        relu3_3 = x
        x = F.max_pool2d(x, kernel_size =2, stride = 2)

        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        relu4_3 = x

        features = {}
        features["relu1_2"] = relu1_2
        features["relu2_2"] = relu2_2
        features["relu3_3"] = relu3_3
        features["relu4_3"] = relu4_3
        return features


def vgg16_pretrained(weight_path = './weights/vgg16.pth'):
    model = vgg16()
    model.load_state_dict(torch.load(weight_path), strict=False)

    # Fixed the pretrained loss network in order to define our loss functions
    for param in model.parameters():
        param.requires_grad = False
    return model

def test():
    model = vgg16_pretrained()
    img = torch.randn(3, 3, 256, 256)
    style = model(img)

    for key, value in style.items():
        print(f"{key}: {value.shape}")
