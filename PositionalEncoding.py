####### Importing Libraries
import os
import gc
import math
import torch
import numpy as np

####### Positional Encoding
def positionalencoding(d_model, T):

    """
    Function to generate positional embeddings

    INPUTS:-
    1) d_model: Input embedding size
    2) T: Total length
    
    OUTPUTS:-
    1) pe: Sinusoidal position encodings of shape [T,d_model]
    """

    ##### Initialization
    device = torch.device("cuda:0")
    pe = torch.zeros(T,d_model).to(device)
    position = torch.arange(0, T).unsqueeze(1).to(device)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float))**(-math.log(10000.0) / d_model)).to(device)
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe.to(device)

###### Testing
#device = torch.device("cuda:0")
#pe = positionalencoding(64,512).to(device)
#pe = pe.detach()
#a = torch.randn(10,512,64).to(device)
#b = a + pe
#print(b.shape)
#print(pe.get_device())
#print(pe.shape)