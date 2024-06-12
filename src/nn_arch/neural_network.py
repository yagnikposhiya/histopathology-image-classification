"""
author: Dr. Sanskruti Patel, Yagnik Poshiya, Rupesh Garsondiya
github: @yagnikposhiya, @Rupeshgarsondiya
organization: Charotar University of Science and Technology
"""

from config import Config
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.models import densenet121, densenet161, densenet169, densenet201
from torchvision.models import mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large
from torchvision.models import efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3
from torchvision.models import efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7
# EfficientNet Paper: https://arxiv.org/abs/1905.11946

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

class Densenet121():
    def __init__(self):
        self.MODEL = densenet121(pretrained=False, num_classes=config.NUM_CLASSES) # did not use pretrained model

class Densenet161():
    def __init__(self):
        self.MODEL = densenet161(pretrained=False, num_classes=config.NUM_CLASSES) # did not use pretrained model

class Densenet169():
    def __init__(self):
        self.MODEL = densenet169(pretrained=False, num_classes=config.NUM_CLASSES) # did not use pretrained model

class Densenet201():
    def __init__(self):
        self.MODEL = densenet201(pretrained=False, num_classes=config.NUM_CLASSES) # did not use pretrained model

class Mobilenetv2():
    def __init__(self):
        self.MODEL = mobilenet_v2(pretrained=False,num_classes=config.NUM_CLASSES) # did not use pretrained model

class Mobilenetv3_small():
    def __init__(self):
        self.MODEL = mobilenet_v3_small(pretrained=False,num_classes=config.NUM_CLASSES) # did not use pretrained model

class Mobilenetv3_large():
    def __init__(self):
        self.MODEL = mobilenet_v3_large(pretrained=False,num_classes=config.NUM_CLASSES) # did not use pretrained model

class Efficientnet_b0():
    def __init__(self):
        self.MODEL = efficientnet_b0(pretrained=False,num_classes=config.NUM_CLASSES) # did not use pretrained model
    
class Efficientnet_b1():
    def __int__(self):
        self.MODEL = efficientnet_b1(pretrained=False, num_classes=config.NUM_CLASSES) # did not use prtrained model

class Efficientnet_b2():
    def __int__(self):
        self.MODEL = efficientnet_b2(pretrained=False, num_classes=config.NUM_CLASSES) # did not use prtrained model

class Efficientnet_b3():
    def __int__(self):
        self.MODEL = efficientnet_b3(pretrained=False, num_classes=config.NUM_CLASSES) # did not use prtrained model

class Efficientnet_b4():
    def __int__(self):
        self.MODEL = efficientnet_b4(pretrained=False, num_classes=config.NUM_CLASSES) # did not use prtrained model

class Efficientnet_b5():
    def __int__(self):
        self.MODEL = efficientnet_b5(pretrained=False, num_classes=config.NUM_CLASSES) # did not use prtrained model

class Efficientnet_b6():
    def __int__(self):
        self.MODEL = efficientnet_b6(pretrained=False, num_classes=config.NUM_CLASSES) # did not use prtrained model

class Efficientnet_b7():
    def __int__(self):
        self.MODEL = efficientnet_b7(pretrained=False, num_classes=config.NUM_CLASSES) # did not use prtrained model