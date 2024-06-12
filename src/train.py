"""
author: Dr. Sanskruti Patel, Yagnik Poshiya
github: @yagnikposhiya
organization: Charotar University of Science and Technology
"""

import os
import torch
import json
import wandb
import pandas as pd
import torch.optim as optim
import pytorch_lightning as pl

from PIL import Image
from config import Config
from torchinfo import summary
from torchvision import transforms
from gpu_config.check import check_gpu_config
from utils.featuremaps import FeatureExtractor
from pytorch_lightning.loggers import WandbLogger
from chaoyang_data import trainChaoyangDataLoading
from nn_arch.neural_network_config import CustomModule
from torch.utils.data import TensorDataset, DataLoader, random_split
from utils.utils import num_unique_labels, samples_per_category, model_selection, is_directory_existed, model_save_path

from torchvision.models import resnet18


if __name__=='__main__':

    check_gpu_config() # check available compute modules (gpus) configuration

    config = Config() # create an object of class Config

    user_choice, nn_arch_name, nn_arch = model_selection() # get user choice, nn architecture name, and neural network architecture

    # initialize the weights & biases cloud server instance
    wandb.init(entity=config.ENTITY,
               project=config.PROJECT,
               anonymous=config.ANONYMOUS,
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

    train_dataframe = train_dataframe.head(20) # extract some portion of train dataframe
    test_dataframe = test_dataframe.head(20) # extract some portion of test dataframe

    print('- Data samples from train set: \n{}'.format(train_dataframe.head(10))) # first 10 train data samples
    print('- Data samples from test set: \n{}'.format(test_dataframe.head(10))) # first 10 test data samples``
    print('- Categories & Samples per category for training set:')
    num_unique_labels(train_dataframe) # the number of unique labels
    samples_per_category(train_dataframe) # the number of data samples per class/category

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
    model = nn_arch.MODEL # load resnet neural network
    model = model.to(device) # move model arch to available computing device
    criterion = config.LOSS # set loss/criterion
    optimizer = optim.Adam(model.parameters(),lr=config.LEARNING_RATE) # set optimizer

    wandb_logger = WandbLogger(log_model=config.LOG_MODEL) # set wandb logger

    trainer = pl.Trainer(max_epochs=config.MAX_EPOCHS, logger=wandb_logger) # initialize the pytorch lightning trainer

    model = CustomModule(model, criterion, optimizer, config.NUM_CLASSES) # create an object of CustomModule class & set the model configuration

    print('- Model summary: \n')
    summary(model,(1,3,512,512)) # model summary; input shape is extracted @ data loading time...................

    print('Training started...')

    trainer.fit(model, train_loader, val_loader) # train the neural network

    print('Training finished.')

    is_directory_existed(config.MODEL_SAVE_ROOT_PATH) # call function and check target directory exists or not
    current_metrics = trainer.callback_metrics # extract callback metrics i.e. for validation loss and validation accuracy
    model_name, path = model_save_path(config.MODEL_SAVE_ROOT_PATH, nn_arch_name, config.MAX_EPOCHS, current_metrics['valid_acc'].item(), current_metrics['valid_loss'].item()) # set model name & get model save path
    torch.save(model, path) # save model @ specified path

    print('- Trained model saved as {}'.format(model_name)) # saved model successfully


    wandb.finish() # close the weights & biases cloud instance