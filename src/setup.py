"""
author: Dr. Sanskruti Patel, Yagnik Poshiya
github: @yagnikposhiya
organization: Charotar University of Science and Technology
"""

from setuptools import setup, find_packages

setup(
    name='research-work', # name of the package
    version='1.0.0', # version of the package
    description='Performance analysis of existing neural network architectures available in the PyTorch framework on Chaoyang Data',
    author='Yagnik Poshiya', # package author name
    author_email='yagnikposhiya.updates@gmail.com', # package author mail
    url='', # url of package repository
    packages=find_packages(), # automatically find packages in 'src' directory
    install_requires=[ # list of dependencies required by package
        'opencv-python',
        'wandb',
        'scikit-learn',
        'pandas',
        'numpy',
        'pillow',
        'pytorch_lightning',
        'torch',
        'torchvision'
    ],
    classifiers=[ # list of classifiers describing package
        'Programming Language :: Python :: 3.10.12'
    ]
)