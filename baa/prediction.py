import argparse
import os
from utils import extract_info
from preprocessing.preprocessing import Preprocessing
from age.model import BoneAgeAssessment, BaaModel

def prediction(**args: dict) -> str:
    """Predict the age of an individual based on an input image.

    This function performs the following steps:
    1. Preprocesses the input image.
    2. Loads the Bone Age Assessment model and makes a prediction based on the preprocessed image.

    Args:
        **args (dict): A dictionary containing the following parameters:
            - 'image name': String representing the name of the input image file.
            - 'path_to_image': String representing the path to the input image file.
            - 'save': Boolean indicating whether to save the preprocessed image.

    Returns:
        str: A string representing the prediction result.
    """
    if args is None:
        raise ValueError("**args is a dictionary containing: image name','path_to_image','save'")
    image = os.path.join(args['path name'], args['image name'])

    # First preprocess the image
    prep_instance = Preprocessing()
    preprocessed_image = prep_instance.preprocessing_image(image_name=image, save=args['save'])

    # Now load model and make prediction
    baa_instance = BoneAgeAssessment()
    prediction = baa_instance.prediction(preprocessed_image, show=True, save=True, image_id=2)
    # prediction uses the best_model.keras
    return prediction

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict the age of an individual based on an input image.')
    parser.add_argument('image_name', type=str, help='Name of the input image file')
    parser.add_argument('path_to_image', type=str, help='Path to the input image file')
    parser.add_argument('--save', action='store_true', help='Whether to save the preprocessed image')

    args = parser.parse_args()

    prediction_result = prediction(args.image_name, args.path_to_image, args.save)
    print(prediction_result)
