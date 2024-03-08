import os
from model import BoneAgeAssessment, BaaModel
from utils import extract_info

def process(**params: dict):
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
    baa_instance = BoneAgeAssessment()
    baa_instance.__update_batch_size__(params.get('Batch size', ()))
    baa_instance.__update_epochs__(params.get('Number of epochs', 0))

    # Show info
    baa_instance.__show_info__()

    # Now create the model
    model = BaaModel(summ=False).vgg16regression_l2(params.get('Regularization factor', 0.0))

    # Compile the model
    baa_instance.compiler(model)

    # Start training
    baa_instance.training_evaluation(model)

    # Test the model with best weights of last model
    WEIGHTS_NAME = 'best_model.keras'
    PATH_TO_WEIGHTS = os.path.join(extract_info('main'), 'baa', 'age', 'weights', WEIGHTS_NAME)
    baa_instance.model_evaluation(PATH_TO_WEIGHTS)

if __name__ == '__main__':
    # Define your parameters
    parameters = {
        'Learning rate': ...,  # Fill in your learning rate
        'Regularization factor': ...,  # Fill in your regularization factor
        'Batch size': (..., ..., ...),  # Fill in your batch sizes
        'Number of epochs': ...  # Fill in your number of epochs
    }

    # Call the process function with the defined parameters
    process(**parameters)



