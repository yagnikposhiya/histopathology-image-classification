"""
author: Dr. Sanskruti Patel, Yagnik Poshiya
github: @yagnikposhiya
organization: Charotar University of Science and Technology
"""

import torch
import json
import wandb
import pandas as pd
import torch.optim as optim
import pytorch_lightning as pl

from config import Config
from nn_arch.neural_network import Resnet18
from pytorch_lightning.loggers import WandbLogger
from chaoyang_data import trainChaoyangDataLoading
from nn_arch.neural_network_config import CustomModule
from torch.utils.data import TensorDataset, DataLoader, random_split


if __name__=='__main__':

    config = Config() # create an object of class Config
    model_resnet = Resnet18() # create an object of class Resnet18

    # initialize the weights & biases cloud server instance
    wandb.init(entity=config.ENTITY,
               project=config.PROJECT,
               anonymous=config.ANONYMOUS,
               group = config.GROUP[0],
               reinit=config.REINIT)

    train_json_file = config.JSON_FILEPATH_TRAIN # set train json filepath
    test_json_file = config.JSON_FILEPATH_TEST # set test json filepath

    # read train json file
    with open(train_json_file,'r') as f1:
        train_data = json.load(f1) # load train json file content

    # read test json file
    with open(test_json_file,'r') as f2:
        test_data = json.load(f2) # load test json file content

    train_dataframe = pd.DataFrame(train_data) # transform train json data into pandas dataframe
    test_dataframe = pd.DataFrame(test_data) # transform test json data into pandas dataframe

    print('Data samples from train set: \n{}'.format(train_dataframe.head(10))) # first 10 train data samples
    print('Data samples from test set: \n{}'.format(test_dataframe.head(10))) # first 10 test data samples``

    X_train, Y_train = trainChaoyangDataLoading(train_dataframe, config.ROOT_PATH) # load whole training set with labels

    train_dataset = TensorDataset(X_train,Y_train) # assume X_train & Y_train are already loaded as torch tensors

    # split training set into train & validation set
    total_samples = len(train_dataset) # total number of data samples
    train_size = int(config.SPLIT_RATIO * total_samples) # set train set size
    val_size = total_samples - train_size # set validation set size
    train_dataset, val_dataset = random_split(train_dataset,[train_size,val_size]) # split dataset
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=config.SHUFFLE) # create an object of DataLoader class for training set
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=config.SHUFFLE) # create an object of DataLoader class for validation set


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # set computing device
    model = model_resnet.MODEL # load resnet neural network
    model = model.to(device) # move model arch to available computing device
    criterion = config.LOSS # set loss/criterion
    optimizer = optim.Adam(model.parameters(),lr=config.LEARNING_RATE) # set optimizer

    wandb_logger = WandbLogger(log_model=config.LOG_MODEL) # set wandb logger

    trainer = pl.Trainer(max_epochs=config.MAX_EPOCHS, logger=wandb_logger) # initialize the pytorch lightning trainer

    model = CustomModule(model, criterion, optimizer, config.NUM_CLASSES) # create an object of CustomModule class & set the model configuration

    trainer.fit(model, train_loader, val_loader) # train the neural network

    wandb.finish() # close the weights & biases cloud instance

    print('Training finished.')