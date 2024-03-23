"""
author: Dr. Sanskruti Patel, Yagnik Poshiya
github: @yagnikposhiya
organization: Charotar University of Science and Technology
"""

import torch.nn as nn

class Config():
    def __init__(self) -> None:

        # Data loading parameters
        self.ROOT_PATH = '/data/home/20CE114/workstation/lab/histopathology-image-classification/data/raw/chaoyang-data/' # set root path for dataset
        self.JSON_FILEPATH_TRAIN = '/data/home/20CE114/workstation/lab/histopathology-image-classification/data/raw/chaoyang-data/train.json' # set json filepath for train directory
        self.JSON_FILEPATH_TEST = '/data/home/20CE114/workstation/lab/histopathology-image-classification/data/raw/chaoyang-data/test.json' # set json filepath for test directory

        # Model training parameters
        self.NUM_CLASSES = 4 # set number of categories in dataset
        self.BATCH_SIZE = 4 # set batch size 
        self.LEARNING_RATE = 0.001 # set learning rate
        self.SPLIT_RATIO = 0.8 # set train-validation split ratio
        self.SHUFFLE = False # set boolean value for data shuffling
        self.LOSS = nn.CrossEntropyLoss() # set loss/criterion
        self.MAX_EPOCHS = 10 # set total nuber of epochs

        # Log visualization parameters
        self.LOG_MODEL = 'all' # set log model type for wandb

        # For model training
        self.LOG_NAME_TRAIN_LOSS = 'train_loss' # set log name for loss
        self.LOG_NAME_TRAIN_ACC = 'train_acc' # set log name for accuracy 
        self.LOG_NAME_TRAIN_PRECISION = 'train_mean_precision' # set log name for precision 
        self.LOG_NAME_TRAIN_RECALL = 'train_mean_recall' # set log name for recall
        self.LOG_NAME_TRAIN_F1 = 'train_mean_f1_score' # set log name for f1 score
        self.LOG_NAME_TRAIN_TP = 'train_tp' # set log name for true positives
        self.LOG_NAME_TRAIN_FP = 'train_fp' # set log name for false positives
        self.LOG_NAME_TRAIN_FN = 'train_fn' # set log name for false negatives
        self.LOG_NAME_TRAIN_TN = 'train_tn' # set log name for true negatives

        # For model validation
        self.LOG_NAME_VALID_LOSS = 'valid_loss' # set log name for loss
        self.LOG_NAME_VALID_ACC = 'valid_acc' # set log name for accuracy 
        self.LOG_NAME_VALID_PRECISION = 'valid_mean_precision' # set log name for precision 
        self.LOG_NAME_VALID_RECALL = 'valid_mean_recall' # set log name for recall
        self.LOG_NAME_VALID_F1 = 'valid_mean_f1_score' # set log name for f1 score
        self.LOG_NAME_VALID_TP = 'valid_tp' # set log name for true positives
        self.LOG_NAME_VALID_FP = 'valid_fp' # set log name for false positives
        self.LOG_NAME_VALID_FN = 'valid_fn' # set log name for false negatives
        self.LOG_NAME_VALID_TN = 'valid_tn' # set log name for true negatives

        # Weights & biases config
        self.ENTITY = 'neuralninjas' # set team/organization name for wandb account
        self.PROJECT = 'research-work' # set project name
        self.GROUP = [ # set gropu name
            'resnet-18'
            ]
        self.REINIT = True # set boolean value for re-initialization 
        self.ANONYMOUS = 'allow' # set anonymous value type

