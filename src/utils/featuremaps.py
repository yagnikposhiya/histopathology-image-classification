"""
author: Dr. Sanskruti Patel, Yagnik Poshiya
github: @yagnikposhiya
organization: Charotar University of Science and Technology
"""

import os
import torch

from torchvision import utils
from utils.utils import is_directory_existed

class FeatureExtractor():
    def __init__(self, model_path:str, feature_map_directory: str, nn_arch_name:str, val_acc:float, val_loss:float) -> None:
        self.model = torch.load(model_path, map_location=torch.device('cpu')) # laod the model from given path
        self.model.eval() # evaluate the model, turns off certain operations like dropout & batch normalization layers
        self.feature_maps = {} # initialize the feature maps

        self.feature_map_directory = feature_map_directory # set path to save feature maps at specified directory
        self.nn_arch_name = nn_arch_name # set neural network architecture name
        self.val_acc = str(val_acc) # set validation accuracy in the string format
        self.val_loss = str(val_loss) # set validation loss in the string format

        # register hook on each layer to capture feature maps
        for name, layer in self.model.named_modules():
            layer.register_forward_hook(self.get_feature_maps(name))

    def get_feature_maps(self, name):
        '''
        Register a hook to capture feature maps for a specific layer in the model.

        Parameters:
        - name (str): The name of the layer for which feature maps will be captured

        Returns:
        - (unknown): A hook function that captures the output of the specified layer
        '''

        def hook(model, input, output):
            self.feature_maps[name] = output.detach() # store specific layer's feature map as an output with layer's name as a key

        return hook
        
    def extract_feature_maps(self, input_image) -> None:
        '''
        This function is used to extract feature maps from an image

        Parameters:
        - input_image (torch.Tensor): an input imgae of type torch.Tensor

        Returns:
        - (None)
        '''

        with torch.no_grad(): # disable gradient computation
            _ = self.model(input_image)
        
    def print_feature_maps(self) -> None:
        '''
        This function is used to save generated feature maps

        Parameters:
        - (None)

        Returns:
        - (None)
        '''

        for name, fmap in self.feature_maps.items(): # iterate over the feature maps and save each pair (original and normalized) as image

            normalized_fmap = (fmap - fmap.min())/(fmap.max() - fmap.min()) # normalie the feature map tensor to range [0,1]

            is_directory_existed(self.feature_map_directory) # check if directory exists, if not create it

            original_image_path = os.path.join(self.feature_map_directory, f'{self.nn_arch_name}_{self.val_acc}_{self.val_loss}_{name}_original_feature_map.jpg') # generate an image path
            normalized_image_path = os.path.join(self.feature_map_directory, f'{self.nn_arch_name}_{self.val_acc}_{self.val_loss}_{name}_normalized_feature_map.jpg') # generate an image path
            
            utils.save_image(fmap, original_image_path) # save original feature map
            utils.save_image(normalized_fmap, normalized_image_path) # save normalized feature map




