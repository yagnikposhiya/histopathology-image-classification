"""
author: Dr. Sanskruti Patel, Yagnik Poshiya
github: @yagnikposhiya
organization: Charotar University of Science and Technology
"""

import os
import torch
import pandas as pd

from PIL import Image
from torchvision import transforms


def trainChaoyangDataLoading(train_df:pd.core.frame.DataFrame, root_path:str) -> torch.Tensor:
    '''
    This function is used to load chaoyang training data only.

    Parameters:
    - train_df (pd.core.frame.DataFrame): dataframe contains training data samples
    - root_path (str): root path for train and test directories
    
    Returns:
    - (torch.Tensor): data samples and their labels
    '''

    # initialize lists
    X_train = [] # to store images
    Y_train = [] # to store labels

    # define a transformation to apply to each image
    transform = transforms.Compose([transforms.ToTensor()]) # convert image to tensor

    # iterate through train dataframe
    for index, row in train_df.iterrows():
        image_path = os.path.join(root_path,row['name']) # set an image path; 'name' is column in the dataframe
        image_label = row['label'] # set an image label; 'label' is column in the dataframe

        image = Image.open(image_path) # load an image
        image = transform(image) # apply defined transformation on an image

        X_train.append(image) # append an image array
        Y_train.append(image_label) # append an image label

    X_train = torch.stack(X_train) # convert image array list to tensor
    Y_train = torch.tensor(Y_train) # convert image label list to tensor

    print('- Shape of X_train: {}'.format(X_train.shape)) # X_train shape
    print('- Shape of Y_train: {}'.format(Y_train.shape)) # Y_train shape

    return X_train, Y_train



