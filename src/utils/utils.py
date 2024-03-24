"""
author: Dr. Sanskruti Patel, Yagnik Poshiya
github: @yagnikposhiya
organization: Charotar University of Science and Technology
"""

import os
import pandas as pd
from nn_arch.neural_network import Resnet18, Resnet34, Resnet50, Resnet101, Resnet152

def num_unique_labels(df:pd.core.frame.DataFrame) -> None:
    '''
    This function is used to count total number of categories existed in the specific set.
    
    Parameters:
    - df (pd.core.frame.DataFrame): dataframe contains data samples

    Returns:
    - (None)
    '''

    print('-- Number of classes/categories/labels: {}'.format(df['label'].nunique())) # the number of unique labels

def samples_per_category(df:pd.core.frame.DataFrame) -> None:
    '''
    This function is used to count data samples per each existing class/category.

    Parameters:
    - df (pd.core.frame.DataFrame): dataframe contains data samples

    Returns:
    - (None)
    '''

    print('-- The number of data samples per class/category/label: \n{}'.format(df['label'].value_counts())) # the number of data samples per class/category

def is_directory_existed(directory_path:str) -> None:
    '''
    This function is used to check target directory exist or not in the provided path.
    If target directory does not exist in the path then create it.

    Parameters:
    - directory_path (str): path for target directory

    Returns:
    - (None)
    '''

    if not os.path.exists(directory_path): # check if target directory exists
        os.makedirs(directory_path) # create target directory @ specified path
        print('- Directory is successfully created at: {}'.format(directory_path))
    else:
        print('- Directory already exists at: {}'.format(directory_path))

def get_choice(min_choice:int, max_choice:int) -> int:
    '''
    This function is used to take user's choice as an input for further processing.

    Parameters:
    - min_choice (int): min. integer value user can provide
    - max_choice (int): max. integer value user can provide

    Returns:
    - (int): user's input/choice
    '''

    while (True):
        try:
            choice = int(input('* Enter your choice: '))
            if min_choice<= choice <= max_choice: # check if choice lies between a range
                return choice
            else:
                print('Invalid choice. Please enter a number between {min_choice} and {max_choice}.'.format(min_choice=min_choice, max_choice=max_choice))
            
        except:
            print("Invalid input. Please enter a valid integer.")

    

def model_selection():
    '''
    This function provides facility to user to select neural network architecture for training.

    Parameters:
    - (None)

    Returns:
    - (int): user's choice
    - (str): name of neural network architecture selected based on user's choice
    - (unknown): neural network without pre-trained weights
    '''

    # create a dictionary for user choices
    choices = {
        'choice_no':[0, 1, 2, 3, 4],
        'nn_arch':['ResNet-18', 'ResNet-34', 'ResNet-50', 'ResNet-101', 'ResNet-152']
    }

    dataframe = pd.DataFrame(choices) # convert choices dict into dataframe
    print(dataframe) # show choices to user

    user_choice = get_choice(0, len(dataframe)-1) # get user's choice
    selected_nn_arch = dataframe.loc[dataframe['choice_no'] == user_choice, 'nn_arch'].values[0]

    if selected_nn_arch == 'ResNet-18':
        nn_arch = Resnet18()
    elif selected_nn_arch == 'ResNet-34':
        nn_arch = Resnet34()
    elif selected_nn_arch == 'ResNet-50':
        nn_arch = Resnet50()
    elif selected_nn_arch == 'ResNet-101':
        nn_arch = Resnet101()
    elif selected_nn_arch == 'ResNet-152':
        nn_arch = Resnet152()

    return user_choice, selected_nn_arch, nn_arch
