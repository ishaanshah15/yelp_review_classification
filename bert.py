from torch import nn
from torch.nn import functional as F
import torch
from transformers import BertForSequenceClassification
import numpy as np


class BERT(nn.Module):
    def __init__(self, device):
        super(BERT, self).__init__()
        '''
        Load the pre-trained BERT Language Model Head Model
        '''
        self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased').to(device)

    def forward(self, inputs,mask):
        outputs = self.bert.forward(inputs,attention_mask=mask)
        outputs = outputs[0]
        return outputs










