import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms 

import math
import numpy as np
import os
from os import path
import operator
import pandas as pd
import glob
import random
from torch import Tensor
import time
from numpy import sqrt
from numpy import argmax

#https://github.com/pytorch/pytorch/issues/24826
class PositionalEncoder(torch.nn.Module):
    def __init__(self, d_model, max_seq_len=72):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model) #36, 256
        for pos in range(max_seq_len): #in range 36 (each patient feature vector)
            for i in range(0, d_model, 2): #in range [256] within feature vector
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        with torch.no_grad():
            x = x * math.sqrt(self.d_model)
            seq_len = x.size(1)
            pe = self.pe[:, :seq_len]
            x = x + pe
            return x
        

class TransformerModel(nn.Module):
    def __init__(self,in_size,use_position_enc,
                 n_heads=2,
                 n_encoders=2,
                 num_outputs=2,
                ):
        """Custom Transformer Model class with encoder.
        Keyword arguments:
        in_size -- number of features in one frame's feature vector
        use_position_enc -- whether to use positional encoding
        n_heads -- number of (self-attention + feed forward) heads in an encoder
        n_encoders -- number of encoders
        num_outputs -- number of output classes (binary 0 or 1 means 2 output classes)"""
        
        super(TransformerModel, self).__init__()

        self.in_size=in_size
        self.use_position_enc = use_position_enc #positional encoding or not?
        
        self.encoder_layer =nn.TransformerEncoderLayer(in_size, 
                                                       n_heads,)
        
        self.encoder=nn.TransformerEncoder(self.encoder_layer, n_encoders)
#        self.decoder=nn.TransformerDecoder(self.decoder_layer, 2)

        print("transformer in size", self.in_size)

        self.classifier = nn.Linear(in_size, num_outputs)
        self.pos_embd=PositionalEncoder(in_size)
        
    def forward(self, x):
        """Forward function for Transformer Model.
        Reshapes input to encoder input (S, N, E) : S = sequence or # frame feature vectors fed in from patient (36), N = batch size (1), E = # of features (in_size, 256)"""
        x = x.permute(1, 0, 2) #now [S, 1, 256]
        
        if(self.use_position_enc): #positional encoding choice
            x = self.pos_embd(x)
        
        x = self.encoder(x)
        out = self.classifier(x)
        return out
    
    
