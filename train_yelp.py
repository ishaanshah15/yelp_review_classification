from comet_ml import Experiment
from preprocess import ReviewDataset,load_data
from bert import BERT
from bert_data import load_bert_data
from transformers import BertTokenizer,BertForSequenceClassification
from lstm import LSTMCLS
from torch.utils.data import DataLoader
from torch import nn, optim
import torch
import numpy as np
import sklearn.metrics
import argparse
from tqdm import tqdm  # optional progress bar


hyperparams = {
    "batch_size": 40,
    "window_size":150,
    "rnn_size": 320,
    "embedding_size":360,
    "learning_rate": 0.001
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, train_loader,optimizer, experiment, hyperparams):
    
    """
    Training loop that trains classifier.

    Inputs:
    - model: A classifier model
    - train_loader: Dataloader of training data
    - experiment: comet.ml experiment object
    - hyperparams: Hyperparameters dictionary
    """
    
    
    model = model.train()
    with experiment.train():
        for e in range(hyperparams['num_epochs']):
            print("---------------epoch----------------- " + str(e))
            for batch_id,batch in enumerate(train_loader):
                inputs, labels = batch["inputs"].to(device), batch["labels"].to(device)
                optimizer.zero_grad()
                if hyperparams['model'] == 0:
                    lengths,ratings = batch["lengths"].to(device),batch["ratings"].to(device)
                    outputs = model.forward(inputs,lengths)
                    batch_loss = nn.CrossEntropyLoss()(outputs,labels)
                    probs = nn.Sigmoid()(outputs[:,1] - outputs[:,0])
                else:
                    mask = (inputs != hyperparams['pad_id']).type(torch.FloatTensor).to(device)
                    outputs = model.forward(inputs,mask)
                    batch_loss = nn.CrossEntropyLoss()(outputs,labels)
                    probs = nn.Sigmoid()(outputs[:,1] - outputs[:,0])
                batch_loss.backward()
                optimizer.step()
                print("---------------batch----------------- " + str(batch_id))
                print("batch loss",batch_loss.item())
                probs,labels = probs.detach().cpu(),labels.detach().cpu()
                batch_auc = sklearn.metrics.roc_auc_score(labels,probs)
                print("batch AUC",batch_auc)
                preds = (probs > 0.5).type(torch.LongTensor)
                num_correct = torch.sum(preds == labels,dtype=torch.float64)
                print("batch accuracy",num_correct/len(labels))


def test(model, test_loader,  experiment, hyperparams):
    # removed loss_fn, word2vec from definition
    """
    Testing loop for BERT model and logs perplexity and accuracy to comet.ml.

    Inputs:
    - model: A LSTM model
    - test_loader: Dataloader of training data
    - experiment: comet.ml experiment object
    - hyperparams: Hyperparameters dictionary
    """
    # lengths is number of masked inputs
    num_examples = len(test_loader.dataset)
    num_correct = 0
    test_auc = 0
    true_positives = 0
    pred_pos = 0
    pos_labels = 0
    model.eval()
    with torch.no_grad():
        # TODO: Write testing loop
        # NOTE: maybe dont need labels.to(device). Have done it everywhere so far.
        for batch_idx,batch in enumerate(test_loader):
            inputs, labels = batch["inputs"].to(device), batch["labels"].to(device)
            if hyperparams['model'] == 0:
                lengths,ratings = batch["lengths"].to(device),batch["ratings"].to(device)
                outputs = model.forward(inputs,lengths)
                probs = nn.Sigmoid()(outputs[:,1] - outputs[:,0])
            else:
                mask = (inputs != hyperparams['pad_id']).type(torch.FloatTensor).to(device)
                outputs = model.forward(inputs,mask) 
                probs = nn.Sigmoid()(outputs[:,1] - outputs[:,0])
            probs,labels = probs.detach().cpu(),labels.detach().cpu()
            test_auc = test_auc + sklearn.metrics.roc_auc_score(labels,probs)
            preds = (probs > 0.5).type(torch.LongTensor)
            num_correct += torch.sum(preds == labels,dtype=torch.float64)
            true_positives += torch.sum(preds[preds==0] == labels[preds==0],dtype=torch.float64)
            pred_pos += torch.sum(preds == 0,dtype=torch.float64)
            pos_labels += torch.sum(labels == 0,dtype=torch.float64)
         
        test_auc = test_auc/(batch_idx+1)
        accuracy = (num_correct/num_examples).item()
        precision = (true_positives/pred_pos)
        recall = (true_positives/pos_labels)
        f1 = (2*precision*recall)/(precision + recall)
    
    with experiment.test():
        print("test auc:", test_auc)
        print("test accuracy:", accuracy)
        print("test precision:", precision.item())
        print("test recall:", recall.item())
        print("test F1-score:", f1.item())
        experiment.log_metric("AUC", test_auc)
        experiment.log_metric("accuracy", accuracy)
        experiment.log_metric("precision", precision.item())
        experiment.log_metric("recall", recall.item())
        experiment.log_metric("F1-score", f1.item())
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_file")
    parser.add_argument("test_file")
    parser.add_argument("model")
    parser.add_argument("epochs")
    parser.add_argument("-l", "--load", action="store_true",
                        help="load model.pt")
    parser.add_argument("-s", "--save", action="store_true",
                        help="save model.pt")
    parser.add_argument("-T", "--train", action="store_true",
                        help="run training loop")
    parser.add_argument("-t", "--test", action="store_true",
                        help="run testing loop")
    args = parser.parse_args()
    hyperparams['model'] = int(args.model)
    hyperparams['num_epochs'] = int(args.epochs)

    # TODO: Make sure you modify the `.comet.config` file
    
    experiment = Experiment(api_key="8jP6l99HeQ0tH5MAIyk2ui0Qj",
                        project_name="final-project-cs-1460", workspace="qiki6bsb",log_code=False)
    experiment.log_parameters(hyperparams)

   
    train_file,test_file = args.train_file,args.test_file
    

    if hyperparams['model'] == 0:
        # TODO: Load dataset
        train_loader,test_loader,vocab_sz = load_data(train_file,test_file,
                                    hyperparams['batch_size'],hyperparams['window_size']) 

        print("vocab_size",vocab_sz)
        hyperparams["vocab_size"] = vocab_sz  # need this for embedding layer
        model = LSTMCLS(hyperparams).to(device)
    else:
        train_loader,test_loader,pad_id = load_bert_data(train_file,test_file,
                                  hyperparams['batch_size'],hyperparams['window_size'])
        hyperparams["pad_id"] = pad_id
        hyperparams["learning_rate"] = 0.00001 #learning rate might be too high
        model = BERT(device).to(device)
        
    optimizer = optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])

    if args.load:
        #model.load_state_dict(torch.load('./model.pt'))
        checkpoint = torch.load('checkpoint2.pt',map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    if args.train:
        train(model, train_loader, optimizer, experiment,
              hyperparams)
    if args.test:
        print("testing model .....")
        test(model, test_loader, experiment, hyperparams)
    if args.save:
        #torch.save(model.state_dict(), './model.pt')
        checkpoint = {'model':model.state_dict(),'optimizer':optimizer.state_dict()}
        torch.save(checkpoint, 'checkpoint2.pt')

