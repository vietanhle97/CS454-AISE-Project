import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import random_split

import math
import json
import random 
import time

def seed_reset(SEED=0):
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

def load_dataset(batch_size, data_path="data/cifar10"):
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    try:
        print("not download train data")
        trainset = torchvision.datasets.CIFAR10(root=data_path, train=True,
                                                download=False, transform=transform)
    except:
        print("download train data")
        trainset = torchvision.datasets.CIFAR10(root=data_path, train=True,
                                                download=True, transform=transform)
    try:
        print("not download test data")
        testset = torchvision.datasets.CIFAR10(root=data_path, train=False,
                                            download=False, transform=transform)
    except:
        print("download test data")
        testset = torchvision.datasets.CIFAR10(root=data_path, train=False,
                                            download=True, transform=transform)
    # get image size
    img, target = trainset[0]
    input_channel, height, width = img.size()
    img_size = (height, width)

    # divide train and valid
    trainset_size = len(trainset)
    torch.manual_seed(42)
    val_size = int(trainset_size * 0.2)
    train_size = trainset_size  - val_size
    train_ds, val_ds = random_split(trainset, [train_size, val_size])

    # get classes
    classes = trainset.classes

    # loader
    trainloader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, 
                                                shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, 
                                            shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader, valloader, testloader, img_size, input_channel, classes

class ImageClassifierModel(nn.Module):
    def __init__(self, img_size, 
                 num_class, 
                 input_channel, 
                 conv_1_out_channels=32,
                 conv_1_bias=True,
                 conv_2_out_channels=64,
                 conv_2_bias=True,
                 conv_dropout=0.5,
                 fc_hidden_dim=64,
                 fc_num_layers=1,
                 fc_dropout=0.5
                 ):
        '''
        img_size: tuple of int (height, weight)
        config: dictionary of config
        '''
        super(ImageClassifierModel, self).__init__()
        # conv + maxpool + dropout 1
        self.conv1_in_channels = input_channel
        self.conv1_kernel_size = 3                              # config["conv_1_kernel_size"]
        self.conv1_stride = 1                                   # config["conv_1_stride"]
        self.maxpool1_kernel_size = 2                           # config["maxpool_1_kernel_size"]
        self.conv1_out_channels = conv_1_out_channels           # config["conv_1_out_channels"]
        self.conv1_bias = conv_1_bias                           # config["conv_1_bias"]

        self.conv1 = nn.Conv2d(in_channels=self.conv1_in_channels,
                               out_channels=self.conv1_out_channels, 
                               kernel_size=self.conv1_kernel_size, 
                               stride=self.conv1_stride,
                               bias=self.conv1_bias)
        self.maxpool1 = nn.MaxPool2d(self.maxpool1_kernel_size)

        # conv + maxpool + dropout 2
        self.conv2_kernel_size = 3                              # config["conv_2_kernel_size"]
        self.conv2_stride = 1                                   # config["conv_2_stride"]
        self.maxpool2_kernel_size = 2                           # config["maxpool_2_kernel_size"]
        self.conv2_out_channels = conv_2_out_channels           # config["conv_2_out_channels"]
        self.conv2_bias = conv_2_bias                           # config["conv_2_bias"]

        self.conv2 = nn.Conv2d(in_channels=self.conv1_out_channels,
                               out_channels=self.conv2_out_channels, 
                               kernel_size=self.conv2_kernel_size, 
                               stride=self.conv2_stride,
                               bias=self.conv2_bias)
        self.maxpool2 = nn.MaxPool2d(self.maxpool2_kernel_size)
        
        self.conv_act = nn.ReLU()
        self.cov_dropt_rate = conv_dropout                      # config["conv_dropout"]
        self.conv_dropout = nn.Dropout(self.cov_dropt_rate)

        # need to count the size of the image
        input_h, input_w = img_size
        h_conv_1 = math.floor(((input_h - (self.conv1_kernel_size - 1) - 1)/self.conv1_stride)+1) # output of conv
        w_conv_1 = math.floor(((input_w - (self.conv1_kernel_size - 1) - 1)/self.conv1_stride)+1) # output of conv
        h_conv_1 = math.floor(((h_conv_1 - (self.maxpool1_kernel_size - 1) - 1)/self.maxpool1_kernel_size)+1) # output of maxpool
        w_conv_1 = math.floor(((w_conv_1 - (self.maxpool1_kernel_size - 1) - 1)/self.maxpool1_kernel_size)+1) # output of maxpool
        
        h_conv_2 = math.floor(((h_conv_1 - (self.conv2_kernel_size - 1) - 1)/self.conv2_stride)+1) # output of conv
        w_conv_2 = math.floor(((w_conv_1 - (self.conv2_kernel_size - 1) - 1)/self.conv2_stride)+1) # output of conv
        h_conv_2 = math.floor(((h_conv_2 - (self.maxpool2_kernel_size - 1) - 1)/self.maxpool2_kernel_size)+1) # output of maxpool
        w_conv_2 = math.floor(((w_conv_2 - (self.maxpool2_kernel_size - 1) - 1)/self.maxpool2_kernel_size)+1) # output of maxpool

        # fc layer
        self.fc_hidden_dim = fc_hidden_dim                      # config["fc_hidden_dim"]
        self.fc_num_layers = fc_num_layers                      # config["fc_num_layers"]
        
        self.fcs = []
        input_fc, output_fc = (self.conv2_out_channels * h_conv_2 * w_conv_2), self.fc_hidden_dim
        for _ in range(self.fc_num_layers):
            self.fcs.append(nn.Linear(in_features=input_fc, out_features=output_fc))
            input_fc = output_fc
        self.stack_fc = nn.Sequential(*(self.fcs))
        self.fc_dropout_rate = fc_dropout                       # config["fc_dropout"]
        self.fc_dropout = nn.Dropout(self.fc_dropout_rate)
        self.num_class = num_class
        self.last_fc = nn.Linear(input_fc, self.num_class)
        self.fc_act = nn.ReLU()

    def forward(self, x):
        x = self.conv_act(self.conv1(x)) # conv 1
        x = self.maxpool1(x)
        x = self.conv_act(self.conv2(x)) # conv 2
        x = self.maxpool2(x)
        x = self.conv_dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc_dropout(self.fc_act(self.stack_fc(x)))
        x = self.last_fc(x)
        return x

def train_one_epoch(model, train_data_iterator, optimizer, criterion, device, verbose=False, epoch=0):
    total_epoch_loss = 0
    running_loss = 0
    total = 0
    correct = 0
    model.train()
    n_data = len(train_data_iterator)
    for i, data in enumerate(train_data_iterator, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        total_epoch_loss += loss.item()
        running_loss += loss.item()

        if verbose and (i % 2000 == 1999):    # print every 2000 mini-batches
            print('[%d, %5d/%5d] loss: %.3f' % (epoch + 1, i + 1, n_data, running_loss / 2000))
            running_loss = 0.0

    epoch_loss = total_epoch_loss / len(train_data_iterator)
    accuracy = correct / total
    return epoch_loss, accuracy

def evaluate(model, test_data_iterator, criterion, device):
    total_epoch_loss = 0
    total = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_data_iterator, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_epoch_loss += loss.item()
    
    epoch_loss = total_epoch_loss / len(test_data_iterator)
    accuracy = correct / total
    return epoch_loss, accuracy

def verify_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def check_elapsed_time(start_time, end_time):
    """Do not modify the code in this function."""
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def fitness_image_classification(hyperparameter, train_data, valid_data, test_data, classes, img_size=(28,28), input_channel=3, save_path="cifar10", verbose=True):
    seed_reset()
    # cifar 10
    param_dict = hyperparameter

    # create model directory
    model_save_path = os.path.join(save_path, param_dict['model_name'])
    verify_dir(model_save_path)

    # save param before training
    with open(os.path.join(model_save_path, "params.json"), 'w') as paramfile:
        json.dump(param_dict, paramfile, indent=2)

    model = ImageClassifierModel(img_size=img_size, 
                                 num_class=len(classes), 
                                 input_channel=input_channel, 
                                 conv_1_out_channels=param_dict["conv_1_out_channels"],
                                 conv_1_bias=param_dict["conv_1_bias"],
                                 conv_2_out_channels=param_dict["conv_2_out_channels"],
                                 conv_2_bias=param_dict["conv_2_bias"],
                                 conv_dropout=param_dict["conv_dropout"],
                                 fc_hidden_dim=param_dict["fc_hidden_dim"],
                                 fc_num_layers=param_dict["fc_num_layers"],
                                 fc_dropout=param_dict["fc_dropout"])
    device = torch.device(param_dict['device'] if torch.cuda.is_available() else 'cpu')
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    end_train_time, start_train_time = None, None
    end_test_time, start_test_time = None, None
    train_loss, train_acc = None, None
    valid_loss, valid_acc = None, None
    test_loss, test_acc = None, None

    print("START TRAINING")
    print("batch_size", param_dict["batch_size"])
    try:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=param_dict['learning_rate'], momentum=0.9)
        model = model.to(device)
        criterion = criterion.to(device)
        
        best_val_acc = 0
        start_train_time = time.time()
        for epoch in range(param_dict['num_epochs']):
            print(f'Epoch: {epoch+1:02}')
            start_time = time.time()
            train_loss, train_acc = train_one_epoch(model, train_data, optimizer, criterion, device, verbose=verbose, epoch=epoch)
            valid_loss, valid_acc = evaluate(model, valid_data, criterion, device)   
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
        test_loss, test_acc = evaluate(model, test_data, criterion, device)
        end_test_time = time.time()
        print(f'Test Loss: {test_loss:.3f} |  Test Acc: {test_acc*100:.2f}%')
    except KeyboardInterrupt as e:
        print("Keyboard Interrupt. Exit.")
        raise e
    except RuntimeError as e:
        if 'out of memory' in str(e):
            print('WARNING: ran out of memory. Skip train and evaluating this model')
            end_train_time, start_train_time = -1, -1
            end_test_time, start_test_time = -1, -1
            train_loss, train_acc = -1, -1
            valid_loss, valid_acc = -1, -1
            test_loss, test_acc = -1, -1
        else:
            raise e
    except:
        raise e

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
        "conv_1_out_channels":64,
        # [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
        "conv_1_bias":True,
        # [True, False]
        "conv_2_out_channels":64,
        # [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
        "conv_2_bias":True,
        # [True, False]
        "conv_dropout":0.1,
        # [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
        "fc_hidden_dim":64,
        # [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
        "fc_num_layers":2,
        # [1,2,3,4,5,6,7,8]
        "fc_dropout":0.0,
        # [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
        "learning_rate":0.001, 
        # [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]
        # ============= do not modify the below hyperparameter
        "batch_size":8, 
        "num_epochs":10, 
        'device':'cuda'
    }
    return param_dict

class ImageClassifier:

    def __init__(self):
        pass

    @staticmethod
    def load_data():

        # this one for Colab
        # project_path = os.getcwd() + "/drive/My Drive/CS454-AISE-Project" 

        project_path = os.getcwd()

        train_data, valid_data, test_data, img_size, input_channel, classes = load_dataset(4, data_path=project_path+"/data/cifar10")

        return {"train_data" : train_data, "valid_data": valid_data, "test_data": test_data, "img_size": img_size, "input_channel": input_channel, "classes": classes }


    @staticmethod
    def build(parameters, data):

        train_loss, train_acc, valid_loss, valid_acc, test_loss, test_acc = fitness_image_classification(parameters, data["train_data"], data["valid_data"], data["test_data"], data["classes"], img_size=data["img_size"], input_channel=data["input_channel"], save_path="../cifar10")

        print(train_loss, train_acc, valid_loss, valid_acc, test_loss, test_acc)

        return test_acc
# if __name__ == '__main__':
#     ImageClassifier.build()