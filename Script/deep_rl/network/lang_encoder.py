from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import torch
from torch import nn

from .network_utils import *


class LangEncoderBoW(torch.nn.Module):
    """
    language encoder (bag of words)
    """
    def __init__(self, dict_size, embedding_dim=128):
        super(LangEncoderBoW, self).__init__()
        self.dict_size = dict_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.EmbeddingBag(dict_size, embedding_dim)    

    def forward(self, lang):
        return self.embedding(lang)

