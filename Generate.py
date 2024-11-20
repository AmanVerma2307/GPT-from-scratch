####### Importing Libraries
import os
import torch
import numpy as np
from BeamDecode import beam_search

####### Generate
device = torch.device("cuda:0")

def generate(input,model,context_window,max_words,idx2word,beam_decode,beam_width):

    """
    Function to generate outputs

    INPUTS:-
    1) input: Input tokens to the model
    2) model: The model
    3) context_window: The length of context window
    4) max_words: Maximum number of words to be generated
    5) idx2word: Inverse vocabulary
    6) beam_decode: Boolean to tell if beam decode is to be used

    OUPUTS:-
    1) output_greedy: The output list of words as per greedy algorithm
    2) output_beam: The output list as per beam search algorithm
    """

    ##### Mounting
    input = input['data'].to(device) # [1,(T+1)]
    input = input.unsqueeze(0)
    input = input[:,1:] # Slicing till context window length: [1,T]
    model = model.to(device)

    ##### Defining essentials
    output_prob = []
    output_list = []
    output_greedy = []
    #input_curr = torch.zeros_like(input).to(device) # Initial input_curr
    input_curr = input

    ##### Generation
    for idx in range(max_words):

        out, _  = model.forward(input_curr)
        
        #### Greedy decoding
        ### Output extraction
        out_max = torch.argmax(out.detach())        
        output_list.append(out_max)

        ### Next input manipulation
        input_new = torch.zeros_like(input).to(device) # Initial input_curr
        input_new[:,:-1] = input_curr[0,1:]
        input_new[0,-1] = torch.tensor(out_max).to(device)
        input_curr = input_new
        input_curr = input_curr.to(device)

    ##### Tokens to words
    for token in output_list:
        output_greedy.append(idx2word[int((token.cpu().numpy()))])

    return output_greedy