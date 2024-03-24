"""
author: Dr. Sanskruti Patel, Yagnik Poshiya
github: @yagnikposhiya
organization: Charotar University of Science and Technology
"""

from config import Config
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152

config = Config() # create an object of class Config

class Resnet18():
    def __init__(self):
        self.MODEL = resnet18(pretrained=False, num_classes=config.NUM_CLASSES) # did not use pretrained model

class Resnet34():
    def __init__(self):
        self.MODEL = resnet34(pretrained=False, num_classes=config.NUM_CLASSES) # did not use pretrained model

class Resnet50():
    def __init__(self):
        self.MODEL = resnet50(pretrained=False, num_classes=config.NUM_CLASSES) # did not use pretrained model

class Resnet101():
    def __init__(self):
        self.MODEL = resnet101(pretrained=False, num_classes=config.NUM_CLASSES) # did not use pretrained model

class Resnet152():
    def __init__(self):
        self.MODEL = resnet152(pretrained=False, num_classes=config.NUM_CLASSES) # did not use pretrained model       