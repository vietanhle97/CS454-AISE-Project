import os
import re
import sys
import time
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

def _load_dataset():
    print_debug("[I] LOAD DATASET")
    preprocess_pipeline = Pipeline(lambda x: re.sub(r'[^a-z]+', ' ', x))
    TEXT = Field(batch_first = True,
                include_lengths = True, 
                lower=True, 
                preprocessing=preprocess_pipeline)
    LABEL = LabelField(dtype = torch.float)
    train_data = TabularDataset(path="data/sentiment-analysis/Train.csv", 
                                format='csv', 
                                fields=[('review', TEXT), ('sentiment', LABEL)], skip_header=True)
    valid_data = TabularDataset(path="data/sentiment-analysis/Valid.csv", 
                                format='csv', 
                                fields=[('review', TEXT), ('sentiment', LABEL)], skip_header=True)
    test_data = TabularDataset(path="data/sentiment-analysis/Test.csv", 
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
        for _ in range(self.fc_num_layers-1):
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

def fitness_sentiment_analysis(hyperparameter, train_data, valid_data, test_data, vocab_size, padding_idx, verbose=True):
    print_debug("[I] fitness_sentiment_analysis")
    seed_reset()
    NUM_EMBEDDINGS = vocab_size
    PADDING_IDX = padding_idx
    param_dict = hyperparameter
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

    start_fitness_time = time.time()
    train_loss, train_acc = None, None
    valid_loss, valid_acc = None, None
    test_loss, test_acc = None, None

    optimizer = optim.Adam(model.parameters(), lr=param_dict['learning_rate'])
    criterion = nn.BCEWithLogitsLoss()
    model = model.to(device)
    criterion = criterion.to(device)

    best_val_acc = 0
    for epoch in range(param_dict['num_epochs']):
        print(f'Epoch: {epoch+1:02}')
        start_time = time.time()
        train_loss, train_acc = train_one_epoch(model, train_iter, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, val_iter, criterion)   
        if best_val_acc < valid_acc : 
            best_val_acc = valid_acc
            torch.save(model.state_dict(), ('{}_best.pt'.format(param_dict['model_name'])))
        if verbose:
            end_time = time.time()
            epoch_mins, epoch_secs = check_elapsed_time(start_time, end_time)
            print(f'\nEpoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

    model.load_state_dict(torch.load(('{}_best.pt'.format(param_dict['model_name']))))    
    test_loss, test_acc = evaluate(model, test_iter, criterion)
    print(f'\t Test Loss: {test_loss:.3f} |  Test Acc: {test_acc*100:.2f}%')

    end_fitness_time = time.time()
    return train_loss, train_acc, valid_loss, valid_acc, test_loss, test_acc, start_fitness_time, end_fitness_time

def set_hyperparameter_dict():
    param_dict = {
        'model_name': 'sa-1-1',
        'embedding_dim': 128, # can be modified
        'rnn_hidden_dim': 256, # can be modified
        'rnn_num_layers': 2, # can be modified
        'rnn_dropout': 0.5, # can be modified
        'rnn_bidirectional': False, # can be modified
        'fc_hidden_dim' : 128, # can be modified
        'fc_num_layers' : 1, # can be modified
        'fc_dropout' : 0.5, # can be modified
        'batch_size': 32, # can be modified
        'num_epochs': 3, # can be modified
        'learning_rate': 1e-3, # can be modified
        'device':'cuda'
    }
    return param_dict

if __name__ == '__main__':
    # this is just the dummy hyperparameter
    hyperparameter = set_hyperparameter_dict()
    # load the dataset first so we dont have to load it if we want to train a model
    train_data, valid_data, test_data, vocab_size, padding_idx = _load_dataset() 

    train_loss, train_acc, valid_loss, valid_acc, test_loss, test_acc, start_fitness_time, end_fitness_time = fitness_sentiment_analysis(hyperparameter, train_data, valid_data, test_data, vocab_size, padding_idx)
    fitness_mins, fitness_secs = check_elapsed_time(start_fitness_time, end_fitness_time)
    print(f'\Fitness Time: {fitness_mins}m {fitness_secs}s')
    print(train_loss, train_acc, valid_loss, valid_acc, test_loss, test_acc)