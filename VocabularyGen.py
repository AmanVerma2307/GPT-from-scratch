###### Importing Libraries
import os
import re
import nltk
import torch
import torchtext
import numpy as np
import pandas as pd
from Preprocess import preprocess

###### Vocabulary Generation

##### Generator function
def vocab_gen(txt_list,sos_token,eos_token):

    """
    Function to form a dictionary word2idx and idx2word

    INPUTS:-
    1) txt_list: List storing all the sentences
    2) sos_token: The SOS token
    3) eos_token: The EOS token

    OUPTUS:-
    1) word2idx: Dictionary to store word:idx mapping
    2) idx2word: Dictionary to store idx:word mapping
    """

    #### Defining essentials
    word2idx = {sos_token:0, eos_token:1}
    counter = 1

    #### Iteration
    for sentence in txt_list: # Iteration over sentences
        for word in sentence.split(): # Iteration over words
            if(word not in word2idx):
                counter = counter + 1
                word2idx[word] = counter # Appending the word

    idx2word = {idx:word for word,idx in word2idx.items()}
    return word2idx, idx2word

def tokenization(text):
        text = re.split('\W+', text)
        return text

###### Testing
#X = preprocess('./Reviews_q2_main.csv')
#wordd2idx, idx2word = vocab_gen(X,"<sos>","<eos>")
#print(len(wordd2idx))