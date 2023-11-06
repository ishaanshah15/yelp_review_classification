from comet_ml import Experiment
from nb_process import ReviewDataset,load_data
from torch.utils.data import DataLoader
from torch import nn, optim
from naivebayes import NaiveBayes
import torch
import numpy as np
import sklearn.metrics
import argparse
from tqdm import tqdm  # optional progress bar


hyperparams = {
    "num_epochs": 1,
    "batch_size": 100,
    "window_size":500,
    "classifier": "NaiveBayes"
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, train_loader, experiment, hyperparams):
    
    """
    Training loop that trains classifier.

    Inputs:
    - model: A classifier model
    - train_loader: Dataloader of training data
    - experiment: comet.ml experiment object
    - hyperparams: Hyperparameters dictionary
    """
    for batch_id,batch in enumerate(train_loader):
        inputs,labels = batch["inputs"],batch["labels"]
        lengths = batch["lengths"]
        model.trainUni(inputs,lengths,labels)
    #print(model.probt,model.probf)
    #print(model.numt,model.numf)
    #print(len(model.tcount),len(model.fcount))


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
    for batch_idx,batch in enumerate(test_loader):
        inputs, labels = batch["inputs"], batch["labels"]
        lengths,ratings = batch["lengths"],batch["ratings"]
        probs = model.forward(inputs,lengths)
        #print(probs[:10])
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
    parser.add_argument("-l", "--load", action="store_true",
                        help="load model.pt")
    parser.add_argument("-s", "--save", action="store_true",
                        help="save model.pt")
    parser.add_argument("-T", "--train", action="store_true",
                        help="run training loop")
    parser.add_argument("-t", "--test", action="store_true",
                        help="run testing loop")
    args = parser.parse_args()

    # TODO: Make sure you modify the `.comet.config` file
    
    experiment = Experiment(api_key="8jP6l99HeQ0tH5MAIyk2ui0Qj",
                        project_name="final-project-cs-1460", workspace="qiki6bsb",log_code=False)
    experiment.log_parameters(hyperparams)

   
    train_file,test_file = args.train_file,args.test_file

        # TODO: Load dataset
    train_loader,test_loader,vocab_sz = load_data(train_file,test_file,
                                    hyperparams['batch_size'],hyperparams['window_size']) 

    print("vocab_size",vocab_sz)
    hyperparams["vocab_size"] = vocab_sz  # need this for embedding layer
    model = NaiveBayes()
    
        
    if args.train:
        train(model, train_loader, experiment,
              hyperparams)
    if args.test:
        print("testing model ....")
        test(model, test_loader, experiment, hyperparams)

