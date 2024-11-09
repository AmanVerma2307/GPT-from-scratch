###### Importing Libraries
import os
import gc
import torch
import numpy as np
import pandas as pd
from Preprocess import preprocess
from VocabularyGen import vocab_gen

###### Glove Embedding Generator
def Glove_Gen(tgt_vocab,sos_token,eos_token):

    """
    Function to generate weights for Glove embedding layer

    INPUTS:-
    1) tgt_vocab: Target vocab dictionary (word2idx)
    2) sos_token: The SOS token
    3) eos_token: The EOS token

    OUTPUTS:-
    1) tgt_embedding: Target Glove embedding weights (N,d) 
    """

    ##### Defining essentials
    glove_embeddings = []

    ##### Glove embeddings
    vocab, embeddings = [], []
    with open('./glove.twitter.27B/glove.twitter.27B.100d.txt','rt',encoding="utf8") as fi:
        full_content = fi.read().strip().split('\n')
    for i in range(len(full_content)):
        i_word = full_content[i].split(' ')[0]
        i_embeddings = [float(val) for val in full_content[i].split(' ')[1:]]
        vocab.append(i_word)
        embeddings.append(i_embeddings)

    #### Vocabulary
    vocab = [sos_token] + [eos_token] + vocab
    
    #### Embeddings
    embeddings = np.array(embeddings)
    sos_emb = np.zeros((1,embeddings.shape[1]))  # embedding for SOS token.
    eos_emb = np.mean(embeddings,axis=0,keepdims=True) # embedding for EOS token.
    embeddings = np.vstack((sos_emb,eos_emb,embeddings))

    vocab_word2idx = {term:idx for idx,term in enumerate(vocab)}
    vocab_idx2word = {idx:word for word,idx in vocab_word2idx.items()}

    ##### Iteration
    for word, idx in tgt_vocab.items():
        
        if(word in vocab_word2idx):
            glove_embeddings.append(embeddings[vocab_word2idx[word]])

            #for j, word_vocab in enumerate(vocab):
            #    if(word_vocab == word):
            #        glove_embeddings.append(embeddings[j])

        else:
            glove_embeddings.append(embeddings[0])
    
    del(embeddings)
    gc.collect()
    glove_embeddings = np.array(glove_embeddings)
    print(glove_embeddings.shape)
    #np.savez_compressed('./glove_embeddings.npz',glove_embeddings)

    ##### Glove layer generation
    embedding_layer = torch.nn.Embedding.from_pretrained(torch.from_numpy(glove_embeddings).float())
    return embedding_layer