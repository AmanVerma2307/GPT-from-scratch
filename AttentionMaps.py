####### Importing Library
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

###### Attention Map
def attn_map(attn,H,tokens,idx2word):

    """
    Function to generate attention maps

    INPUTS:-
    1) attn: Attention maps of shape [N,H,T,T]
    2) H: Number of attention heads to be visualized
    3) tokens: Input tokens
    4) idx2word: Dictionary to convert indices to words
    """

    #H, T = attn.size(1), attn.size(2)
    attn = attn.detach().cpu().numpy()
    tokens = (tokens.detach().cpu().numpy())[0,:] # Selecting the tokens
    word_tokens = []
    for item in tokens:
        word_tokens.append(idx2word[item])

    if(H == 4):
        
        fig, ((ax1,ax2),(ax3, ax4)) = plt.subplots(nrows=2, ncols=int(H/2), figsize=(8,6))

        for idx, ax in enumerate([ax1,ax2,ax3,ax4]):

            if(ax == ax1):
               
                ax.imshow(attn[0,idx,:,:])
                ax.set_title('H=1', fontsize=10)
                #ax.set_xticks(np.arange(len(word_tokens)), labels=word_tokens)
                #ax.set_yticks(np.arange(len(word_tokens)), labels=word_tokens)
                #plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

                #for i in range(len(word_tokens)):
                #    for j in range(len(word_tokens)):
                #        text = ax.text(j,i,np.round(attn[0,idx,i,j],decimals=2),
                #                       ha="center",va="center",
                #                      color="w")

            if(ax == ax2):

                ax.imshow(attn[0,idx,:,:])
                ax.set_title('H=2', fontsize=10)
                #ax.set_xticks(np.arange(len(word_tokens)), labels=word_tokens)
                #ax.set_yticks(np.arange(len(word_tokens)), labels=word_tokens)
                #plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

                #for i in range(len(word_tokens)):
                #    for j in range(len(word_tokens)):
                #        text = ax.text(j,i,np.round(attn[0,idx,i,j],decimals=2),
                #                       ha="center",va="center",
                #                       color="w")

            if(ax == ax3):
                
                ax.imshow(attn[0,idx,:,:])
                ax.set_title('H=3', fontsize=10)
                #ax.set_xticks(np.arange(len(word_tokens)), labels=word_tokens)
                #ax.set_yticks(np.arange(len(word_tokens)), labels=word_tokens)
                #plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

                #for i in range(len(word_tokens)):
                #    for j in range(len(word_tokens)):
                #        text = ax.text(j,i,np.round(attn[0,idx,i,j],decimals=2),
                #                       ha="center",va="center",
                #                      color="w")

            if(ax == ax4):
                
                ax.imshow(attn[0,idx,:,:])
                ax.set_title('H=4', fontsize=10)
                #ax.set_xticks(np.arange(len(word_tokens)), labels=word_tokens)
                #ax.set_yticks(np.arange(len(word_tokens)), labels=word_tokens)
                #plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

                #for i in range(len(word_tokens)):
                #    for j in range(len(word_tokens)):
                #        text = ax.text(j,i,np.round(attn[0,idx,i,j],decimals=2),
                #                       ha="center",va="center",
                 #                      color="w")

    plt.show()