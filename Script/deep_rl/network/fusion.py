from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import torch
from torch import nn
import torch.nn.functional as F

from .network_utils import *


class FusionCat(torch.nn.Module):
    """
    Fusion by concat: [x, y] -> z
    """
    def __init__(self, input_dim1, input_dim2):
        super(FusionCat, self).__init__()
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2

        self.fc = layer_init(nn.Linear(input_dim1 + input_dim2, input_dim1), 1e-3)

    
    def forward(self, x, y):
        x_y_fusion = F.relu(self.fc(torch.cat([x, y], -1)))
        return x_y_fusion


class FusionAtt(torch.nn.Module):
    """
    Fusion by attention: x * sigmoid(y)
    """
    def __init__(self, input_dim1, input_dim2, drop=0.5):
        super(FusionAtt, self).__init__()
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.output_dim = input_dim1
        
        self.att_linear = nn.Linear(input_dim2, input_dim1)
    
        self.apply(weights_init)
    
    
    def forward(self, x, y):
        att = torch.sigmoid(self.att_linear(y))
        x_y_fusion = x * att
        return x_y_fusion
