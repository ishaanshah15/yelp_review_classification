import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch import nn,optim
import pandas as pd
from torch.utils.data import Dataset

def load_data(train_file,test_file,batch_sz,window_sz):
    train_df = pd.read_csv(train_file,sep=',')
    test_df = pd.read_csv(test_file,sep=',')
    word2id,counts = {},{}
    count_vocab(counts,train_df['reviews'].tolist(),test_df['reviews'].tolist())
    tr_inputs,tr_lengths = preprocess(train_df['reviews'].tolist(),word2id,counts,window_sz)
    te_inputs,te_lengths = preprocess(test_df['reviews'].tolist(),word2id,counts,window_sz)
    train_loader = build_dataset(tr_inputs,tr_lengths,train_df,batch_sz,window_sz,True,0,0)
    test_loader = build_dataset(te_inputs,te_lengths,test_df,batch_sz,window_sz,True,0,0)
    vocab_sz = len(word2id) + 1 # plus 1 added below for "pad" word which has id = 0
    return train_loader,test_loader,vocab_sz

def count_vocab(counts,train_reviews,test_reviews):
    for reviews in [train_reviews,test_reviews]:
        for review in reviews:
            words = review.lower().split() # need to change this in bert_data
            for w in words:
                if w in counts:
                    counts[w] += 1
                else:
                    counts[w] = 1

def preprocess(reviews,word2id,counts,window_sz):
    inputs = []
    lengths = []
    for review in reviews:
        words = review.lower().split() # Should I use lower ?
        words = words[:window_sz]
        res = []
        for w in words:
            if counts[w] < 50: # UNK threshold
                res.append(add_word("UNK",word2id))
            else:
                res.append(add_word(w,word2id))
        inputs.append(torch.tensor(res))
        lengths.append(len(res))
    return inputs,lengths



def lengthen_seq(inputs,seq_len):
    long_seq = torch.zeros(seq_len,dtype=torch.int64) # Long Tensor
    size = len(inputs[0])
    long_seq[:size] = inputs[0]
    inputs[0] = long_seq
   
def build_dataset(inputs,lengths,df,batch_sz,window_size,shuffle,pad_id,model):
    lengthen_seq(inputs,window_size)
    pad_inputs = pad_sequence(inputs,batch_first=True,padding_value=pad_id)
    labels = df['labels'].tolist()
    ratings = df['ratings'].tolist()
    review_set = ReviewDataset(pad_inputs,lengths,ratings,labels,model)
    loader = DataLoader(review_set, batch_size=batch_sz, shuffle=shuffle)
    return loader

def add_word(word,word2id):
    if not word2id.get(word):
        vid = len(word2id) + 1
        word2id[word] = vid
    return word2id[word]

class ReviewDataset(Dataset):
    # TODO: Create masked Penn Treebank dataset.
    #       You can change signature of the initializer.
    def __init__(self,inputs,lengths,ratings,labels,model):
        super().__init__()
        
        self.inputs = inputs
        self.lengths = torch.tensor(lengths)
        self.ratings = torch.tensor(ratings)
        self.labels = torch.tensor(labels) == 1
        if model == 0:
            self.labels = self.labels.type(torch.LongTensor) #change back to Float
        else:
            self.labels = self.labels.type(torch.LongTensor)
      
    def __len__(self):
        """
        __len__ should return a the length of the dataset

        :return: an integer length of the dataset
        """
        # TODO: Override method to return length of dataset
        return len(self.labels)

    def __getitem__(self, idx):
        """
        __getitem__ should return a tuple or dictionary of the data at some
        index

        :param idx: the index for retrieval

        :return: tuple or dictionary of the data
        """
        # TODO: Override method to return the ith item in dataset
        
        
        return {"inputs": self.inputs[idx],
                "lengths":self.lengths[idx],
                "ratings": self.ratings[idx],
                "labels": self.labels[idx]}






