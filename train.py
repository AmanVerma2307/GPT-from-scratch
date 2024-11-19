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
from sklearn.model_selection import train_test_split
from Preprocess import preprocess
from VocabularyGen import vocab_gen
from DataLoader import DataSet
from GloVe_Layer import Glove_Gen
from PositionalEncoding import positionalencoding
from Attention import MHSA
from Perplexity import test 

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
parser.add_argument('--train_mode',
                    type=str,
                    help='full or full_width')

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

tx_model = model(input_dim,d_model,args.num_heads,vocab_size,word2idx).to(device)
print('Model Formulated')
#input = torch.randint(low=0,high=1000,size=(32,args.context_window)).long().to(device)
#out, attn = tx_model.forward(input)
#print(out.shape, attn.shape)

###### Training

##### Training Step
def train_epoch(dataloader, model, optimizer, criterion, train_mode):

    loss_total = 0.0
    perplexity_total = 0.0

    for item in tqdm.tqdm(iter(dataloader),colour='blue'):

        data = item['data']

        if(train_mode == 'full'):

            x = data[:,:-1] # Context window
            y = data[:,-1] # Label

            x = x.to(device)
            y = y.to(device)

            with torch.set_grad_enabled(True):

                out, _ = model.forward(x) # Forward pass
                loss_curr = criterion(out,y) 

                loss_curr.backward()
                optimizer.step()

            loss_total += loss_curr.item()*x.size(0)

        if(train_mode == 'full_width'): 

            for i in range(args.context_window):

                x = data[:,0:(i+1)]   
                y = data[:,(i+1)]

                x = x.to(device)
                y = y.to(device)

                with torch.set_grad_enabled(True):

                    out, _ = model.forward(x) # Forward pass
                    loss_curr = criterion(out,y) 

                    loss_curr.backward()
                    optimizer.step()

                loss_total += loss_curr.item()*x.size(0)

    if(train_mode == 'full'):
        loss_total = loss_total/train_total

    if(train_mode == 'full_width'):
        loss_total = loss_total/(train_total*(args.context_window))

    return loss_total

##### Validation step
def val_epoch(dataloader, model, criterion):

    loss_total = 0.0

    for item in tqdm.tqdm(iter(dataloader),colour='green'):

        data = item['data']

        x = data[:,:-1] # Context window
        y = data[:,-1] # Label

        x = x.to(device)
        y = y.to(device)
        
        with torch.set_grad_enabled(False):
            out, _ = model.forward(x)
            loss = criterion(out,y)

            loss_total += loss.item()*x.size(0)

    loss_total = loss_total/val_total
    return loss_total

###### Training Validation Loop
def train_val(train_loader, val_loader, test_loader, model, optimizer, criterion, criterion_test, num_epochs):

    model_path = './Models/'+args.model_name+'.pth'
    loss_best = 1e+6
    ppl_best = 1e+6

    for epoch in tqdm.tqdm(range(num_epochs),colour='yellow'):

        time_start = time.time()
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        #### Training
        train_loss_epoch = train_epoch(train_loader,model,optimizer,criterion,args.train_mode)
        train_loss.append(train_loss_epoch)

        #### Validation
        val_loss_epoch = val_epoch(val_loader,model,criterion)
        val_loss.append(val_loss_epoch)

        #### Testing
        test_ppl = test(test_loader,model,criterion_test,args.context_window)

        #### Saving
        if(val_loss_epoch < loss_best):
            loss_best = val_loss_epoch
            torch.save(model.state_dict(),model_path)
            ppl_best = test_ppl

        #### Outputs
        print('Total time:'+str(time.time() - time_start))
        print('Loss: '+str(train_loss_epoch))
        print('Validation Loss: '+str(val_loss_epoch))
        print('PPL: '+str(test_ppl))

    return train_loss, val_loss, ppl_best

###### Training and Validation
criterion = torch.nn.CrossEntropyLoss()
criterion_test = torch.nn.CrossEntropyLoss(reduction='none')
optimizer = torch.optim.Adam(tx_model.parameters(),lr=1e-3)
train_loss, val_loss, ppl_best = train_val(TrainLoader,ValLoader,TestLoader,tx_model,optimizer,criterion,criterion_test,20)

print('PPL Model: '+str(ppl_best))
np.savez_compressed('./Loss/'+args.model_name+'_trainloss.npz',np.array(train_loss))
np.savez_compressed('./Loss/'+args.model_name+'_valloss.npz',np.array(val_loss))
