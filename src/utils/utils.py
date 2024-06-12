"""
author: Dr. Sanskruti Patel, Yagnik Poshiya, Rupesh Garsondiya
github: @yagnikposhiya, @Rupeshgarsondiya
organization: Charotar University of Science and Technology
"""

import os
import pandas as pd
from nn_arch.neural_network import Resnet18, Resnet34, Resnet50, Resnet101, Resnet152
from nn_arch.neural_network import Densenet121, Densenet161, Densenet169, Densenet201
from nn_arch.neural_network import Mobilenetv2, Mobilenetv3_small, Mobilenetv3_large
from nn_arch.neural_network import Efficientnet_b0, Efficientnet_b1, Efficientnet_b2, Efficientnet_b3
from nn_arch.neural_network import Efficientnet_b4, Efficientnet_b5, Efficientnet_b6, Efficientnet_b7

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
        'choice_no':[0, 1, 2, 3, 4, 
                     5, 6, 7, 8,
                     9, 10, 11,
                     12, 13, 14, 15,
                     16, 17, 18, 19],
        'nn_arch':['ResNet-18', 'ResNet-34', 'ResNet-50', 'ResNet-101', 'ResNet-152',
                   'DenseNet-121', 'DenseNet-161', 'DenseNet-169', 'DenseNet-201',
                   'MobileNetV2', 'MobileNetV3-Small', 'MobileNetV3-Large',
                   'Efficientnet-b0', 'Efficientnet-b1', 'Efficientnet-b2', 'Efficientnet-b3',
                   'Efficientnet-b4', 'Efficientnet-b5', 'Efficientnet-b6', 'Efficientnet-b7']
    }

    dataframe = pd.DataFrame(choices) # convert choices dict into dataframe
    print(dataframe.to_string(index=False)) # show choices to user without indices store by dataframe

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
    elif selected_nn_arch == 'DenseNet-121':
        nn_arch = Densenet121()
    elif selected_nn_arch == 'DenseNet-161':
        nn_arch = Densenet161()
    elif selected_nn_arch == 'DenseNet-169':
        nn_arch = Densenet169()
    elif selected_nn_arch == 'DenseNet-201':
        nn_arch = Densenet201()
    elif selected_nn_arch == 'MobileNetV2':
        nn_arch = Mobilenetv2()
    elif selected_nn_arch == 'MobileNetV3-Small':
        nn_arch = Mobilenetv3_small()
    elif selected_nn_arch == 'MobileNetV3-Large':
        nn_arch = Mobilenetv3_large()
    elif selected_nn_arch == 'Efficientnet-b0':
        nn_arch = Efficientnet_b0()
    elif selected_nn_arch == 'Efficientnet-b1':
        nn_arch = Efficientnet_b1()
    elif selected_nn_arch == 'Efficientnet-b2':
        nn_arch = Efficientnet_b2()
    elif selected_nn_arch == 'Efficientnet-b3':
        nn_arch = Efficientnet_b3()
    elif selected_nn_arch == 'Efficientnet-b4':
        nn_arch = Efficientnet_b4()
    elif selected_nn_arch == 'Efficientnet-b5':
        nn_arch = Efficientnet_b5()
    elif selected_nn_arch == 'Efficientnet-b6':
        nn_arch = Efficientnet_b6()
    elif selected_nn_arch == 'Efficientnet-b7':
        nn_arch = Efficientnet_b7()

    print('- {} architecture is selected.'.format(selected_nn_arch))

    return user_choice, selected_nn_arch, nn_arch

def model_save_path(root_path:str, nn_arch_name:str, epoch:int, val_acc:float, val_loss:float) -> str:
    '''
    This function is used to create model name and joint it with root path to create path for saving a trained model.

    Parameters:
    - root_path (str: root path for directory in which model will be saved
    - nn_arch_name (str): name of neural network architecture selected by user
    - epoch (int): number of epochs on which nn arch is trained
    - val_acc (float): validation accuracy for last step of last epoch
    - val_loss (float): validation loss for last step of last epoch

    Returns:
    - (str): model name only
    - (str): path including model name @ which model will be saved
    '''

    model_name = f'{str(nn_arch_name)}_{str(epoch)}_{str(val_acc)}_{str(val_loss)}.pt' # generate model name
    path = os.path.join(root_path, model_name) # generate whole path including model name

    return model_name, path
