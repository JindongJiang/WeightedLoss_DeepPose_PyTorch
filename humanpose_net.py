import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import models


class HumanPoseNet(nn.Module):
    def __init__(self, out_c, vgg_pretrained=True):
        super(HumanPoseNet, self).__init__()

        self.model = models.vgg16_bn(pretrained=vgg_pretrained)
        self.model.classifier._modules['0'] = nn.Linear(25088, 4096)
        self.model.classifier._modules['3'] = nn.Linear(4096, 4096)
        self.model.classifier._modules['6'] = nn.Linear(4096, out_c)

    def forward(self, input):
        assert tuple(input.data.shape[-2:]) == (224, 224)
        x = self.model(input)

        return x


def mse_loss(pred, target, weight, weighted_loss=False, size_average=True):
    mask = (weight != 0).float()
    if weighted_loss:
        loss = torch.sum(weight * (pred - target) ** 2)
    else:
        loss = torch.sum(mask * (pred - target) ** 2)
    if size_average:
        loss /= torch.sum(mask)
    return loss
