import os
import re
import sys
import time
import json
import random
import argparse
from tqdm import tqdm


import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.legacy.data import Field, LabelField, TabularDataset, Pipeline, BucketIterator
from torch.nn.utils.rnn import pack_padded_sequence

def seed_reset(SEED=0):
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

DEBUG_MODE=True
def print_debug(*args, **kwargs):
    global DEBUG_MODE
    if DEBUG_MODE:
        print(*args, **kwargs)

def _load_dataset(train_path="data/sentiment-analysis/Train.csv",
                  valid_path="data/sentiment-analysis/Valid.csv", 
                  test_path="data/sentiment-analysis/Test.csv"):
    print_debug("[I] LOAD DATASET")
    preprocess_pipeline = Pipeline(lambda x: re.sub(r'[^a-z]+', ' ', x))
    TEXT = Field(batch_first = True,
                include_lengths = True, 
                lower=True, 
                preprocessing=preprocess_pipeline)
    LABEL = LabelField(dtype = torch.float)
    train_data = TabularDataset(path=train_path, 
                                format='csv', 
                                fields=[('review', TEXT), ('sentiment', LABEL)], skip_header=True)
    valid_data = TabularDataset(path=valid_path, 
                                format='csv', 
                                fields=[('review', TEXT), ('sentiment', LABEL)], skip_header=True)
    test_data = TabularDataset(path=test_path, 
                                format='csv', 
                                fields=[('review', TEXT), ('sentiment', LABEL)], skip_header=True)

    TEXT.build_vocab(train_data)
    LABEL.build_vocab(train_data)
    vocab_size = len(TEXT.vocab)
    padding_idx = TEXT.vocab.stoi[TEXT.pad_token]

    print_debug("[O] LOAD DATASET")
    return train_data, valid_data, test_data, vocab_size, padding_idx

def check_elapsed_time(start_time, end_time):
    """Do not modify the code in this function."""
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def accuracy(prediction, label):
    """Do not modify the code in this function."""
    binary_prediction = torch.round(torch.sigmoid(prediction))
    correct = (binary_prediction == label).float()
    acc = correct.sum() / len(correct)
    return acc

class SentimentModel(nn.Module):
    def __init__(self, num_embeddings, padding_idx, embedding_dim, 
                 rnn_hidden_dim, 
                 rnn_num_layers, 
                 rnn_dropout, 
                 rnn_bidirectional,
                 fc_hidden_dim, 
                 fc_num_layers, 
                 fc_dropout):
        """ Build a SentimentModel model
        :param num_embeddings: the numebr of embeddings (vocab size)
        :param padding_idx: padding idx
        :param embedding_dim: (int) embedding dimension
        :param rnn_hidden_dim: (int) hidden dimension
        :param rnn_num_layers: (int) the number of recurrent layers
        :param rnn_dropout: (float) rnn_dropout rate
        :param rnn_bidirectional: (bool) is rnn_bidirectional
        :return output: type=torch.Tensor, shape=[batch size]
        """
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx = padding_idx)
        self.emb_dropout = nn.Dropout(rnn_dropout)
        self.num_embeddings = num_embeddings
        self.padding_idx = padding_idx
        self.embedding_dim = embedding_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.rnn_num_layers = rnn_num_layers
        self.rnn_dropout_rate = rnn_dropout
        self.rnn_bidirectional = rnn_bidirectional

        self.rnn = nn.GRU(input_size=self.embedding_dim,
                          hidden_size=self.rnn_hidden_dim,
                          num_layers=self.rnn_num_layers,
                          dropout=self.rnn_dropout_rate,
                          bidirectional=self.rnn_bidirectional,
                          batch_first=True
                          )
        self.fc_infeatures = self.rnn_hidden_dim
        if self.rnn_bidirectional:
            self.fc_infeatures += self.rnn_hidden_dim
        
        self.fc_hidden_dim = fc_hidden_dim 
        self.fc_num_layers = fc_num_layers 
        self.fc_dropout_rate = fc_dropout
        self.fcs = []
        input_fc, output_fc = self.rnn_hidden_dim, self.fc_hidden_dim
        for _ in range(self.fc_num_layers):
            self.fcs.append(nn.Linear(in_features=input_fc, out_features=output_fc))
            input_fc = output_fc
        self.stack_fc = nn.Sequential(*(self.fcs))
        self.fc_dropout = nn.Dropout(self.fc_dropout_rate)
        self.last_fc = nn.Linear(in_features=input_fc,
                            out_features=1)
        
    def forward(self, text, text_lengths):
        embedded = self.emb_dropout(self.embedding(text))
        packed_embedded = pack_padded_sequence(embedded, text_lengths.to('cpu'), batch_first = True)
        _, hidden = self.rnn(packed_embedded)
        if self.rnn_bidirectional:
          hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
          hidden = hidden[-1,:,:]
        output = self.fc_dropout(self.stack_fc(hidden))
        output = self.last_fc(output)
        output = output.squeeze(1)
        assert output.shape == torch.Size([text.shape[0]]) # batch_size
        return output


def train_one_epoch(model, train_data_iterator, optimizer, criterion):
    """ Complete train method
    :param model: SentimentModel model
    :param train_data_iterator: train dataset train_data_iterator
    :param optimizer: optimzer
    :param criterion: loss function

    :return output: train loss, train accuracy
    """
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.train()
    for batch in tqdm(train_data_iterator, desc="train"):
        optimizer.zero_grad()      
        (text, text_lengths), labels = batch.review, batch.sentiment        
        predict = model(text, text_lengths)
        loss = criterion(predict, labels)
        acc = accuracy(predict, labels) 
        loss.backward()
        optimizer.step()
        assert loss.shape == torch.Size([])
        assert acc.shape == torch.Size([])        
        total_epoch_loss += loss.item()
        total_epoch_acc += acc.item()
    epoch_loss = total_epoch_loss / len(train_data_iterator)
    epoch_acc = total_epoch_acc / len(train_data_iterator)
    return epoch_loss, epoch_acc


def evaluate(model, test_data_iterator, criterion):
    """ Complete evaluate method
    :param model: SentimentModel model
    :param test_data_iterator: dataset test_data_iterator
    :param criterion: loss function

    :return output: loss, accuracy
    """    
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_data_iterator, desc="evaluate"):
            (text, text_lengths), labels = batch.review, batch.sentiment
            predict = model(text, text_lengths)
            loss = criterion(predict, labels)
            acc = accuracy(predict, labels) 
            assert loss.shape == torch.Size([])
            assert acc.shape == torch.Size([])
            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()
    epoch_loss = total_epoch_loss / len(test_data_iterator)
    epoch_acc = total_epoch_acc / len(test_data_iterator)
    return epoch_loss, epoch_acc

def verify_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def fitness_sentiment_analysis(hyperparameter, train_data, valid_data, test_data, vocab_size, padding_idx, save_path="sentiment-analysis-model", verbose=True):
    print_debug("[I] fitness_sentiment_analysis")
    seed_reset()
    NUM_EMBEDDINGS = vocab_size
    PADDING_IDX = padding_idx
    param_dict = hyperparameter

    # create model directory
    model_save_path = os.path.join(save_path, param_dict['model_name'])
    verify_dir(model_save_path)

    # save param before training
    with open(os.path.join(model_save_path, "params.json"), 'w') as paramfile:
        json.dump(param_dict, paramfile, indent=2)

    model = SentimentModel(NUM_EMBEDDINGS, 
                PADDING_IDX,
                param_dict['embedding_dim'], 
                param_dict['rnn_hidden_dim'], 
                param_dict['rnn_num_layers'], 
                param_dict['rnn_dropout'], 
                param_dict['rnn_bidirectional'],
                param_dict['fc_hidden_dim'],
                param_dict['fc_num_layers'],
                param_dict['fc_dropout']
                )
    device = torch.device(param_dict['device'] if torch.cuda.is_available() else 'cpu')
    train_iter, val_iter = BucketIterator.splits(
                                    (train_data, valid_data), 
                                    batch_size = param_dict['batch_size'],
                                    sort_within_batch = True,
                                    sort_key=lambda x: len(x.review),
                                    device = device) 
    test_iter = BucketIterator(test_data, 
                                batch_size = param_dict['batch_size'],
                                sort_within_batch = True,
                                sort_key=lambda x: len(x.review),
                                device = device)

    end_train_time, start_train_time = None, None
    end_test_time, start_test_time = None, None
    train_loss, train_acc = None, None
    valid_loss, valid_acc = None, None
    test_loss, test_acc = None, None

    try:
        optimizer = optim.Adam(model.parameters(), lr=param_dict['learning_rate'])
        criterion = nn.BCEWithLogitsLoss()
        model = model.to(device)
        criterion = criterion.to(device)

        best_val_acc = 0
        start_train_time = time.time()
        for epoch in range(param_dict['num_epochs']):
            print(f'Epoch: {epoch+1:02}')
            start_time = time.time()
            train_loss, train_acc = train_one_epoch(model, train_iter, optimizer, criterion)
            valid_loss, valid_acc = evaluate(model, val_iter, criterion)   
            if best_val_acc < valid_acc : 
                best_val_acc = valid_acc
                torch.save(model.state_dict(), os.path.join(model_save_path, ('{}_best.pt'.format(param_dict['model_name']))))
            if verbose:
                end_time = time.time()
                epoch_mins, epoch_secs = check_elapsed_time(start_time, end_time)
                print(f'\nEpoch Time: {epoch_mins}m {epoch_secs}s')
                print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
                print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

        end_train_time = time.time()

        model.load_state_dict(torch.load(os.path.join(model_save_path, ('{}_best.pt'.format(param_dict['model_name'])))))    
        start_test_time = time.time()
        test_loss, test_acc = evaluate(model, test_iter, criterion)
        end_test_time = time.time()
        print(f'\t Test Loss: {test_loss:.3f} |  Test Acc: {test_acc*100:.2f}%')
    except:
        end_train_time, start_train_time = -1, -1
        end_test_time, start_test_time = -1, -1
        train_loss, train_acc = -1, -1
        valid_loss, valid_acc = -1, -1
        test_loss, test_acc = -1, -1

    # save result before return
    fitness_result = {
        'train_loss' : train_loss, 
        'train_acc' : train_acc, 
        'valid_loss' : valid_loss, 
        'valid_acc' : valid_acc, 
        'test_loss' : test_loss, 
        'test_acc' : test_acc,
        'train_time' : (end_train_time - start_train_time),
        'test_time' : (end_test_time - start_test_time)
    }
    with open(os.path.join(model_save_path, "results.json"), 'w') as resultfile:
        json.dump(fitness_result, resultfile, indent=2)

    return train_loss, train_acc, valid_loss, valid_acc, test_loss, test_acc

def set_hyperparameter_dict():
    param_dict = {
        'model_name': 'sa-1-1', # this is just identifier first '1' means generation and second '1' is just id
        'embedding_dim': 128, # can be modified
        # [8, 16, 32, 64, 128, 256, 512, 1024]
        # please set 1024 to be the maximum value
        'rnn_hidden_dim': 256, # can be modified
        # [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
        # please set 4096 to be the maximum value
        'rnn_num_layers': 2, # can be modified
        # [1,2,3,4,5,6,7,8]
        # please set 8 to be the maximum value
        'rnn_dropout': 0.5, # can be modified
        # [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
        # please set 0.85 to be the maximum value
        'rnn_bidirectional': False, # can be modified
        # [True, False]
        'fc_hidden_dim' : 128, # can be modified
        # [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
        # please set 4096 to be the maximum value
        'fc_num_layers' : 1, # can be modified
        # [1,2,3,4,5,6,7,8]
        # please set 8 to be the maximum value
        'fc_dropout' : 0.5, # can be modified
        # [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
        # please set 0.85 to be the maximum value
        'learning_rate': 1e-3, # can be modified
        # [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]
        # please set 0.01 to be the maximum value and 0.00001 to the minimum value
        'batch_size': 32,
        # actually, we can modify the batch size
        # but i think we do not have to modify the batch_size because it effects the training time. 
        # and also it has dependency with dataset
        'num_epochs': 1, 
        # i think we do not have to modify the num of epochs because it realy effects the training time
        # moreover, we will pick the model that has highest validation score during training

        'device':'cuda'
    }
    return param_dict


class SentimentAnalysisModel:

    def __init__(self):
        pass


    @staticmethod
    def build():
        hyperparameter = set_hyperparameter_dict()
        # load the dataset first so we dont have to load it if we want to train a model
        train_data, valid_data, test_data, vocab_size, padding_idx = _load_dataset(train_path="../data/sentiment-analysis/Train.csv",
                                                                                   valid_path="../data/sentiment-analysis/Valid.csv",
                                                                                   test_path="../data/sentiment-analysis/Test.csv") 

        train_loss, train_acc, valid_loss, valid_acc, test_loss, test_acc = fitness_sentiment_analysis(hyperparameter, train_data, valid_data, test_data, vocab_size, padding_idx, save_path="../sentiment-analysis-model")

        print(train_loss, train_acc, valid_loss, valid_acc, test_loss, test_acc)

        return 100

if __name__ == '__main__':
    SentimentAnalysisModel.build()