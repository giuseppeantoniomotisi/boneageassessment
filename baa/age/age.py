import argparse
import json
import logging
import os
from model import BoneAgeAssessment, BaaModel
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
    if params == None:
        params = {
            'Learning rate': 1e-05,
            'Regularization factor': 1e-04,
            'Batch size': (32, 32, 1396),
            'Number of epochs': 20}
    # Updates hyperparameters
    baa = BoneAgeAssessment()
    baa.__update_batch_size__(params.get('Batch size', ()))
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
    WEIGHTS_NAME = 'best_model.keras'
    PATH_TO_WEIGHTS = os.path.join(extract_info('main'), 'baa', 'age', 'weights', WEIGHTS_NAME)
    baa.model_evaluation(PATH_TO_WEIGHTS)

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
