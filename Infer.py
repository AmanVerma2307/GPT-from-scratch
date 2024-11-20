####### Importing Libraries
import os
import gc
import time
import random
import argparse
import tqdm
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from Preprocess import preprocess
from VocabularyGen import vocab_gen
from DataLoader import DataSet
from GloVe_Layer import Glove_Gen
from PositionalEncoding import positionalencoding
from Attention import MHSA
from AttentionMaps import attn_map
from Generate import generate

####### Input arguments
parser = argparse.ArgumentParser()

parser.add_argument('--context_window',
                    type=int,
                    help='Size of the context window')
parser.add_argument('--num_heads',
                    type=int,
                    help="Number of attention heads")
parser.add_argument('--embedding_dim',
                    type=int,
                    help="Embedding size of the model")
parser.add_argument('--batch_size',
                    type=int,
                    help="Batch size for training/testing")
parser.add_argument('--model_name',
                    type=str,
                    help="Name of the model to be saved")
parser.add_argument('--mode',
                    type=str,
                    help='attn_map, generate, generate_beam')

args = parser.parse_args()
print('Importing and parsing done')

####### Data Processing

###### Vocabulary generation
X = preprocess('./Reviews_q2_main.csv',args.context_window)
word2idx, idx2word = vocab_gen(X,"<sos>","<eos>")
print('Vocabulary generated')

###### Data split
X_train, X_val = train_test_split(X, test_size=0.4, random_state=100) # Fixing the train-test split
X_val, X_test = train_test_split(X_val, test_size=0.5, random_state=100) # Fixing validation-test split

###### Dataloader
##### Dataset
TrainDataset = DataSet(X_train,word2idx,idx2word,
                       args.context_window,"<sos>","<eos>")
del(X_train)
gc.collect()


ValDataset = DataSet(X_val,word2idx,idx2word,
                    args.context_window,"<sos>","<eos>")
del(X_val)
gc.collect()

TestDataset = DataSet(X_test,word2idx,idx2word,
                    args.context_window,"<sos>","<eos>")
del(X_test)
gc.collect()

##### Dataloader
TrainLoader = torch.utils.data.DataLoader(TrainDataset,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   drop_last=False)
ValLoader = torch.utils.data.DataLoader(ValDataset,
                                   batch_size=args.batch_size,
                                   shuffle=False,
                                   drop_last=False)
TestLoader = torch.utils.data.DataLoader(TestDataset,
                                   batch_size=args.batch_size,
                                   shuffle=False,
                                   drop_last=False)

print('Dataloader set')

####### Model training

###### Defining essentials
input_dim = 100
d_model = args.embedding_dim
sos_token = '<sos>'
eos_token = '<eos>'
vocab_size = len(word2idx)
train_total = TrainDataset.__len__()
val_total = ValDataset.__len__()
device = torch.device("cuda:0")
train_loss = []
val_loss = []

###### Model
class model(torch.nn.Module):

    """
    Transformer model
    """

    def __init__(self,input_dim,embedding_dim,num_heads,vocab_size,vocab):
        
        #### Defining essentials
        super().__init__()
        self.input_dim = input_dim # Input embedding dimensions
        self.embedding_dim = embedding_dim # Embedding dimensions
        self.num_heads = num_heads # Number of attention heads
        self.vocab_size = vocab_size # Vocabulary size
        self.vocab = vocab # Input vocabulary

        #### Defining layers
        self.GloVe = Glove_Gen(self.vocab,sos_token,eos_token)
        self.MHSA = MHSA(self.num_heads,self.input_dim,self.embedding_dim)
        self.layer_norm = torch.nn.LayerNorm(self.embedding_dim)
        self.op_linear = torch.nn.Linear(self.embedding_dim,self.vocab_size)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self,x):

        """
        Transformer model

        INPUTS:-
        1) x: Input of size: [N,T]

        OUTPUTS:-
        1) out: Ouptut of size [N,vocab_size]
        2) Attn: Attention weights of size [N,H,T,T]
        """

        #### Defining essentials
        T = x.size(1)
        pe_embedding = positionalencoding(self.embedding_dim,T)
        pe_embedding = pe_embedding.detach()

        #### Forward pass
        x_glove = self.GloVe(x) # Glove layer: [N,T,D']
        x, attn = self.MHSA(x_glove) # MHSA: [N,T,D]
        attn = attn.detach().cpu() # Detaching attention weights
        x = self.layer_norm(x+x_glove) # Layer Norm: [N,T,D]
        out = self.op_linear(x[:,-1,:]) # Final FC layer
        out = self.softmax(out) # Softmax layer

        return out, attn

tx_model = model(input_dim,d_model,args.num_heads,vocab_size,word2idx)
tx_model.load_state_dict(torch.load('./Models/'+args.model_name+'.pth'))
tx_model = tx_model.to(device)
print('Model Formulated')

###### Attention map generation

if(args.mode == 'attn_map'):
    inp_1 = TestDataset.__getitem__(80)['data']
    inp_1 = inp_1.unsqueeze(0)
    inp_1 = inp_1.to(device)
    out, attn = tx_model.forward(inp_1)
    attn_map(attn,4,inp_1,idx2word)

if(args.mode == 'gen'):
    inp_1 = TestDataset.__getitem__(30)
    op = generate(inp_1,tx_model,args.context_window,100,idx2word,False,5)
    
    input_sentence = []
    for item in (inp_1['data'].cpu().numpy()):
        input_sentence.append(idx2word[int(item)])
    print('=============================================')
    print('INPUT')
    print(" ".join(input_sentence))
    print(" ")
    print('OUTPUT')
    print(" ".join(op))

