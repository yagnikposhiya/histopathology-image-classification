"""
author: Dr. Sanskruti Patel, Yagnik Poshiya
github: @yagnikposhiya
organization: Charotar University of Science and Technology
"""

import torch
import subprocess


def check_gpu_config():
    '''
    This function id used to check GPUs available for model training.

    Parameters:
    - 

    Returns:
    -
    '''

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count() # get total number of gpus available
        print('- Number of gpus available: {}'.format(num_gpus))

        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i) # get gpu name
            print('- GPU name: {}'.format(gpu_name))

        command = 'nvidia-smi' # set a command
        result = subprocess.run(command, shell=True, capture_output=True, text=True) # execute command

        if result.returncode == 0:
            print(result.stdout) # command execution output
        else:
            print('- Error message: \n{}'.format(result.stderr)) # command execution error message


    else:
        print('- CUDA is not available. Using CPU instead.')
