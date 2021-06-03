import torch
import torch.nn as nn
from torchvision import models, transforms

class vgg16(nn.Module):
    """
    Pretrained weight download (Provide by Justin Johnson):
        https://web.eecs.umich.edu/~justincj/models/vgg16-00b39a1b.pth
    Github page:
        https://github.com/jcjohnson/pytorch-vgg
    """
    def __init__(self, weight_path = './weights/vgg16.pth'):
        super(vgg16, self).__init__()
        loss_network = models.vgg16(pretrained = False)
        loss_network.load_state_dict(torch.load(weight_path), strict = False)
        self.features = loss_network.features

    def forward(self, x):
        feature_extract = {
            '3' : 'relu1_2',
            '8' : 'relu2_2',
            '15' : 'relu3_3',
            '22' : 'relu4_3'
        }
        return_features = {}
        for name, layer in self.features._modules.items():
            x = layer(x)
            if name in feature_extract:
                return_features[feature_extract[name]] = x
                if name == '22':
                    break
        return return_features


def vgg16_pretrained(weight_path = './weights/vgg16.pth'):
    model = vgg16(weight_path)

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
