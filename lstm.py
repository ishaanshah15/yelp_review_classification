from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch


class LSTMCLS(nn.Module):
    def __init__(self, hyperparams):
        """
        The Model class implements the LSTM-LM model.
        Feel free to initialize any variables that you find necessary in the
        constructor.

        :param vocab_size: The number of unique tokens in the data
        :param rnn_size: The size of hidden cells in LSTM/GRU
        :param embedding_size: The dimension of embedding space
        """
        super().__init__()
        # TODO: initialize the vocab_size, rnn_size, embedding_size

        self.vocab_size = hyperparams['vocab_size']
        self.window_size = hyperparams['window_size']
        self.rnn_size = hyperparams['rnn_size']
        self.embedding_size = hyperparams['embedding_size']

        # TODO: initialize embeddings, LSTM, and linear layers

        self.embedding_layer = nn.Embedding(self.vocab_size,self.embedding_size)

        # Note: I can add linear layers here.
        self.lstm = nn.LSTM(self.embedding_size,self.rnn_size)

        self.fc_layer = nn.Linear(self.rnn_size*self.window_size,self.rnn_size)

        self.fc_layer2 = nn.Linear(self.rnn_size,2)
        #self.fc_layer2 = nn.Linear(self.rnn_size,1)
        

    def forward(self, inputs, lengths):

        """
        Runs the forward pass of the model.

        :param inputs: word ids (tokens) of shape (window_size, batch_size)
        :param lengths: array of actual lengths (no padding) of each input

        :return: the logits, a tensor of shape
                 (window_size, batch_size, vocab_size)
        """
        # TODO: write forward propagation

        # make sure you use pack_padded_sequence and pad_padded_sequence to
        # reduce calculation
        
        # Note inputs - (window_size, batch_size)
        # Note all entries of inputs must be integers in order to be embedded.

       
        # FINAL PROJECT: CHANGE TO WINDOW FIRST
        #print("input size",inputs.size())
        
        x = inputs.permute(1,0)

        window_size = x.size()[0]

        # NOTE do you want to add padding_idx?

        x = self.embedding_layer(x)

        # Note embedded_inputs - (window_size, batch_size, embedding_size)

        x = pack_padded_sequence(x,lengths,enforce_sorted = False)

        x,_ = self.lstm(x)

        # NOTE IMPORTANT: It says that batch elements will be ordered in decreasing order.
        # THIS MIGHT MESS SHIT UP ESPECIALLY IF BATCH LABELS ARE NOT SORTED

        x,_ = pad_packed_sequence(x,total_length=window_size)

        # above size is (window_sz,batch_sz,rnn_size)

        x = x.permute(1,0,2)
        x = x.flatten(start_dim=1,end_dim=2)


        # NOTE: SHOULD I BE ADDING ACTIVATION SOME WHERE

        x = F.relu(self.fc_layer(x))
        x = self.fc_layer2(x)
        #x = nn.Sigmoid()(self.fc_layer2(x)) 
        #x = x.flatten()
        
        return x


