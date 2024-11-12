####### Importing Libraries
import os
import gc
import torch
import numpy as np
import matplotlib.pyplot as plt
from Preprocess import preprocess
from VocabularyGen import vocab_gen

####### Attention Head
class HardAttentionHead(torch.nn.Module):

    """
    Hard Attention Head 
    """

    def __init__(self, input_embedding_size, embedding_size):
        
        #### Defining essentials
        super().__init__()
        self.input_embedding_size = input_embedding_size # Size of the input embedding
        self.embedding_size = embedding_size # Embedding Size

        #### Defining layers
        self.query = torch.nn.Linear(self.input_embedding_size,
                                     self.embedding_size,
                                     bias=False)
        self.key = torch.nn.Linear(self.input_embedding_size,
                                   self.embedding_size,
                                   bias=False)
        self.value = torch.nn.Linear(self.input_embedding_size,
                                    self.embedding_size,
                                    bias=False)
        
    def forward(self,x):

        """
        Hard Attention Head

        INPUTS:-
        1) x: Input of shape [N,T,D']

        OUPUTS:-
        1) out: Output of shape [N,T,D]
        2) attn: Attention weights of shape [N,T,T]
        """

        #### Finding temporal steps in the input
        T = x.size(1) 
        
        #### Attention
        q = self.query(x) # Query: [N,T,D'] -> [N,T,D]
        k = self.key(x) # Key: [N,T,D'] -> [N,T,D]
        Att = torch.bmm(q,k.permute((0,2,1))) # Attention Map: QK^T, [N,T,T]
        Att = torch.masked_fill(torch.tril(Att),Att==0,float('-inf')) # (QK^T)o(M) -> [N,T,T]
        Att = torch.nn.functional.softmax(Att,dim=-1) # Softmax, [N,T,T]
        v = self.value(x) # Value: [N,T,D'] -> [N,T,D]
        out = torch.bmm(Att,v) # Attention aggregation: Att*V, [N,T,D]
        return out, Att

####### Multi-head Attention
class MHSA(torch.nn.Module):

    """
    Hard Multi-head Self Attention (MHSA)
    """

    def __init__(self,num_heads,input_embedding_dim,embedding_dim):

        #### Defining essentials
        super().__init__()
        self.num_heads = num_heads # Total number of heads
        self.input_embedding_dim = input_embedding_dim # Size of the input embedding
        self.embedding_dim = embedding_dim # Embedding dimensions

        #### Defining layers
        self.MHSA = torch.nn.ModuleList([HardAttentionHead(self.input_embedding_dim,self.embedding_dim) for _ in range(self.num_heads)])
        self.linear = torch.nn.Linear(self.embedding_dim*self.num_heads,self.embedding_dim)

    def forward(self,x):

        """
        Multi-headed masked self attention (MHSA)

        INPUTS:-
        1) x: Input of shape [N,T,D']
        
        OUPUTS:-
        1) out: Output of shape [N,T,D]
        2) attn: Attention weights of shape [N,H,T,T]
        """

        ##### Iteration over attention heads
        N,T = x.size(0), x.size(1)
        out = []
        attn = []

        for h in self.MHSA:
            out_curr, attn_curr = h(x) # Self attention
            out.append(out_curr)
            attn.append(attn_curr)

        out_curr = out_curr.detach().cpu()
        attn_curr = attn_curr.detach().cpu()

        del(out_curr, attn_curr)
        gc.collect()

        out = torch.cat(out,dim=-1) #[N,T,D*H]
        out = self.linear(out) # Linear operation [N,T,D*H] -> [N,T,D]

        attn = torch.stack(attn,dim=-1) # [N,T,T,H]
        attn = attn.permute((0,3,1,2)) # [N,H,T,T]
        
        return out, attn

##### Testing
#device = torch.device("cuda:0")
#input = torch.randn((1024,64,100),dtype=torch.float).to(device)
#mhsa = MHSA(12,100,64).to(device)
#attn_layer = HardAttentionHead(100,64).to(device)
#output, attn = mhsa(input)
#print(output.shape, attn.shape)
#attn = attn[0].detach().cpu().numpy()
#plt.imshow(attn)
#plt.show()