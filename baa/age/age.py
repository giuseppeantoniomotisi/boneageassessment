# Copyright (C) 2024 motisigiuseppeantonio@yahoo.it
#
# For the license terms see the file LICENSE, distributed along with this
# software.
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

"""
This Python script is designed for Bone Age Assessment in Machine Learning. It uses a
combination of hyperparameter tuning, model creation, training, and evaluation processes.
Here's a breakdown of what the code does:

1. Imports: The script imports necessary modules such as argparse for parsing command-line
arguments, json for handling JSON files, logging for logging messages, os for operating
system-related functionalities, and modules from the local project (model.py and utils.py).
2. Configuration: Sets up logging configurations to control the verbosity and format of log
messages.
3. Process Function: Defines a function named process that takes a dictionary of parameters
as input. This function handles the entire workflow including hyperparameter tuning, model
creation, training, and evaluation.
4. Argument Parsing: Parses command-line arguments using argparse. The script can accept
parameters either from a JSON file (--macro) or through keyboard input (--keyboard_input).
If neither option is provided, it uses default values.
5. Parameter Handling: Reads parameters from either a JSON file or keyboard input, or uses
default values if no input method is specified.
6. Execution: Calls the process function with the parameters obtained from the argument parsing
step.

In summary, this script provides a streamlined workflow for Bone Age Assessment, allowing for
easy experimentation with different hyperparameters and input methods.
"""
import argparse
import json
import logging
import os
from model import BoneAgeAssessment
from model import BaaModel
from utils import extract_info

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def process(params: dict):
    """Process function for hyperparameter tuning, model creation, training, and evaluation.

    This function performs the following steps:
    1. Updates hyperparameters including batch size and number of epochs.
    2. Creates a Bone Age Assessment model using VGG16 architecture with L2 regularization.
    3. Compiles the created model.
    4. Starts training the model.
    5. Evaluates the model using the best weights of the last trained model.

    Args:
        **params (dict): A dictionary containing the following parameters:
            - 'Learning rate': Float value specifying the learning rate for the optimizer.
            - 'Regularization factor': Float value specifying the regularization factor.
            - 'Batch size': Tuple of integers specifying the batch sizes for training.
            - 'Number of epochs': Integer specifying the number of epochs for training.

    Returns:
        None
    """
    if params is None:
        params = {
            'Learning rate': 1e-05,
            'Regularization factor': 1e-04,
            'Batch size': (32, 32, 1396),
            'Number of epochs': 20}

    # Updates hyperparameters
    baa = BoneAgeAssessment()
    baa.__update_batch_size__(params.get('Batch size', ()), key='all')
    baa.__update_epochs__(params.get('Number of epochs', 0))

    # Show info
    baa.__show_info__()

    # Now create the model
    model = BaaModel(summ=False).vgg16regression_l2(params.get('Regularization factor', 0.0))

    # Compile the model
    baa.compiler(model)

    # Start training
    baa.training_evaluation(model)

    # Test the model with best weights of last model
    weights_name = 'best_model.keras'
    path_to_weights = os.path.join(extract_info('main'), 'baa', 'age', 'weights', weights_name)
    baa.model_evaluation(path_to_weights)

if __name__ == '__main__':
    # Example of parameters structure
    # parameters = {
    #   'Learning rate': ...,  # Fill in your learning rate
    #   'Regularization factor': ...,  # Fill in your regularization factor
    #   'Batch size': (..., ..., ...),  # Fill in your batch sizes
    #   'Number of epochs': ...  # Fill in your number of epochs
    #    }

    parser = argparse.ArgumentParser(description='Bone Age Assessment Process - ML module')
    parser.add_argument('--macro', type=str, help='Path to the parameters JSON file')
    parser.add_argument('--keyboard_input', action='store_true', help='Input parameters via keyboard')

    args = parser.parse_args()

    if args.macro:
        with open(args.macro, 'r') as file:
            parameters = json.load(file)
    elif args.keyboard_input:
        parameters = {
            'Learning rate': float(input("Enter Learning rate: ")),
            'Regularization factor': float(input("Enter Regularization factor: ")),
            'Batch size': tuple(map(int, input("Enter Batch size (comma separated): ").split(','))),
            'Number of epochs': int(input("Enter Number of epochs: "))
        }
    else:
        parameters = None
        logger.warning("No input method specified. Default values are used.")

    # Call the process function with the defined parameters
    process(parameters)
