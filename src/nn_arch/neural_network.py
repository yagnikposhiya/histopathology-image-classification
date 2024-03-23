"""
author: Dr. Sanskruti Patel, Yagnik Poshiya
github: @yagnikposhiya
organization: Charotar University of Science and Technology
"""

from config import Config
from torchvision.models import resnet18

config = Config() # create an object of class Config

class Resnet18():
    def __init__(self):
        self.MODEL = resnet18(pretrained=False, num_classes=config.NUM_CLASSES) # did not use pretrained model