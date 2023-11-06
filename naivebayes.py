from torch import nn
from torch.nn import functional as F
import torch
import numpy as np
from collections import defaultdict


class NaiveBayes():
    def __init__(self):
        super(NaiveBayes, self).__init__()
        self.tcount = defaultdict(int)
        self.fcount = defaultdict(int)
        self.numt, self.numf = 0,0 #number of words 
        self.probt, self.probf = 0,0 

    def trainUni(self, inputs,lengths,labels):

        for i in range(0,len(inputs)):
            for j in range(0,lengths[i].item()):
                key = inputs[i][j].item()
                if labels[i] == 1:
                    self.numt += 1
                    self.tcount[key] += 1
                else:
                    self.numf += 1;
                    self.fcount[key] += 1

        self.probt += torch.sum(labels == 1).item()
        self.probf += torch.sum(labels == 0).item()
        
    
    def forward(self,inputs,lengths):
        logits = []
        for i in range(0,len(inputs)):
            tprob = np.log(self.probt)
            fprob = np.log(self.probf)
            for j in range(0,lengths[i].item()):
                key = inputs[i][j].item()
                tcount,fcount = max(self.tcount[key],10**-2),max(self.fcount[key],10**-2)
                tprob += (np.log(tcount) - np.log(self.numt))
                fprob += (np.log(fcount) - np.log(self.numf))
            logits.append(tprob - fprob)
    
        logits = torch.tensor(logits).type(torch.FloatTensor)
        return nn.Sigmoid()(logits)
    
    
            
        
                
             
      
