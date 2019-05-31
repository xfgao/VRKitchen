from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import math
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models

from .network_utils import *


class StateEncoderShallowCNN(torch.nn.Module):
    """
    state encoder (shallow CNN)
    """
    def __init__(self, output_dim, drop):
        super(StateEncoderShallowCNN, self).__init__()
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(3, 32, 8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc = nn.Linear(64 * 7 * 7, output_dim)
        self.drop = drop
        self.apply(weights_init)

    def forward(self, state):
        x = F.relu(self.bn1(self.conv1(state)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(F.dropout(self.fc(x), p=self.drop))
        return x

class StateEncoderResNet(torch.nn.Module):
    def __init__(self, output_dim, drop):
        super(StateEncoderResNet, self).__init__()
        self.output_dim = output_dim
        self.backbone = models.resnet50(pretrained=True)
        for param in self.backbone.parameters():
            param.requires_grad = True
        # self.conv1 = nn.Conv2d(3, 32, 8, stride=4)
        # self.bn1 = nn.BatchNorm2d(32)
        # self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        # self.bn2 = nn.BatchNorm2d(64)
        # self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        # self.bn3 = nn.BatchNorm2d(64)
        self.fc = nn.Linear(1000, output_dim)
        self.drop = drop
        self.apply(weights_init)
    

    def forward(self, x):
        x = self.backbone(x)
        # x = F.relu(self.bn1(self.conv1(x)))
        # x = F.relu(self.bn2(self.conv2(x)))
        # x = F.relu(self.bn3(self.conv3(x)))
        # x = x.view(-1, 64 * 7 * 7)
        x = F.dropout(F.relu(self.fc(x)), p=self.drop)
        return x
