"""
author: Dr. Sanskruti Patel, Yagnik Poshiya
github: @yagnikposhiya
organization: Charotar University of Science and Technology
"""

import os
import pandas as pd

def num_unique_labels(df:pd.core.frame.DataFrame) -> None:
    '''
    This function is used to count total number of categories existed in the specific set.
    
    Parameters:
    - df (pd.core.frame.DataFrame): dataframe contains data samples

    Retunrs:
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

    print('-- The number of data samples per class/category/label: {}'.format(df['label'].value_counts())) # the number of data samples per class/category
