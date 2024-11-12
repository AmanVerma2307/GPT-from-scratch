####### Importing Libraries
import os
import torch
import numpy as np
import pandas as pd
import torch.utils
import torch.utils.data
from Preprocess import preprocess
from VocabularyGen import vocab_gen

####### DataLoader
class DataSet(torch.utils.data.Dataset):

    def __init__(self,X,word2idx,idx2word,cnt_wnd_len,sos_token,eos_token):
        
        self.word2idx = word2idx # Vocabulary
        self.idx2word = idx2word # Inverse vocabulary
        self.X_ind = [] # List to store indices of the input
        self.cnt_wnd_len = cnt_wnd_len # Context window length
        self.sos_token = sos_token
        self.eos_token = eos_token

        for x_sent in X:
            #x_sent.append(eos_token) # Appending "<eos>" token
            #x_sent[-1], x_sent[-2] = x_sent[-2], x_sent[-1] # Swapping "<eos>" to penultimate position
            self.X_ind.append(self.token2indices(x_sent)) # Converting to indicces
        
    def token2indices(self,tokens):
        token_idx = []
        for token in tokens:
            token_idx.append(self.word2idx[token])
        return (torch.Tensor(token_idx).long())
    
    def __len__(self):
        return len(self.X_ind)
    
    def __getitem__(self,idx):
        sample = {'data':self.X_ind[idx]}
        return sample
    
####### Testing
#X = preprocess('./Reviews_q2_main.csv',16)
#word2idx, idx2word = vocab_gen(X,"<sos>","<eos>")
#Dataset = DataSet(X,word2idx,idx2word,16,"<sos>","<eos>")
#print(Dataset.__getitem__(10))
#print(Dataset.__getitem__(10)['data'].shape)

#Dataloader = torch.utils.data.DataLoader(Dataset,
#                                         batch_size=32,
#                                         shuffle=True,
#                                         drop_last=False)

#for item in iter(Dataloader):
#    print(item['data'].shape)
