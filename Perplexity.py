####### Importing Libraries
import os
import gc
import time
import random
import argparse
import tqdm
import torch
import numpy as np

####### Test loop
device = torch.device("cuda:0")
def test(dataloader, model, criterion, context_window):

    ppl_total = 0.0

    for item in tqdm.tqdm(iter(dataloader),colour='magenta'):

        data = item['data']

        for i in range(context_window):

            x = data[:,0:(i+1)].to(device)
            y = data[:,(i+1)].to(device)

            with torch.set_grad_enabled(False):

                out, _ = model.forward(x)
                ppl_curr = torch.mean(torch.exp(criterion(out,y)))

                ppl_total += ppl_curr.item()*x.size(0)

    ppl_total = ppl_total/len(dataloader)
    return ppl_total












    