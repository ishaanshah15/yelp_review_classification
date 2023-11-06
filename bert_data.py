import numpy as np
from transformers import BertTokenizer
from preprocess import count_vocab,build_dataset
import torch
from transformers import *
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch import nn,optim
import pandas as pd
from torch.utils.data import Dataset

def load_bert_data(train_file,test_file,batch_sz,window_sz):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    pad_id = tokenizer.pad_token_id
    train_df = pd.read_csv(train_file,sep=',')
    test_df = pd.read_csv(test_file,sep=',')
    counts = {}
    count_vocab(counts,train_df['reviews'].tolist(),test_df['reviews'].tolist())
    tr_inputs,tr_lengths = bertprocess(train_df['reviews'].tolist(),tokenizer,counts,window_sz)
    te_inputs,te_lengths = bertprocess(test_df['reviews'].tolist(),tokenizer,counts,window_sz)
    train_loader = build_dataset(tr_inputs,tr_lengths,train_df,batch_sz,window_sz,True,pad_id,1)
    test_loader = build_dataset(te_inputs,te_lengths,test_df,batch_sz,window_sz,True,pad_id,1)
    return train_loader,test_loader,pad_id

def bertprocess(reviews,tokenizer,counts,window_sz):
    inputs = []
    lengths = []
    for review in reviews:
        # Should I use lower ?
        words = review.lower().split()[:window_sz]
        line = []
        for w in words:
            if counts[w] < 50: # UNK threshold
                line.append("UNK")
            else:
                line.append(w)
        line = " ".join(line)
        line = tokenizer.encode(line)
        inputs.append(torch.tensor(line))
        lengths.append(len(line))
    return inputs,lengths


