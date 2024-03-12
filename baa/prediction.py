import argparse
import os
from utils import extract_info
from preprocessing.tools import Preprocessing
from age.model import BoneAgeAssessment

def process(image_name:str, image_path:str, save:bool=True) -> str:
    """Predict the age of an individual based on an input image.

    This function performs the following steps:
    1. Preprocesses the input image.
    2. Loads the Bone Age Assessment model and makes a prediction based on the preprocessed image.

    Args:
        image_name (str): String representing the name of the input image file.
        image_path (str): String representing the path to the input image file.
        save (str, optional): Boolean indicating whether to save the preprocessed image.

    Returns:
        str: A string representing the prediction result.
    """
    image = os.path.join(image_path, image_name)

    # First preprocess the image
    prep_instance = Preprocessing()
    preprocessed_image = prep_instance.preprocessing_image(image_name=image, save=save)

    # Now load model and make prediction
    baa_instance = BoneAgeAssessment()
    prediction = baa_instance.prediction(preprocessed_image, show=True, save=save, image_id=image_name[:-4])
    # prediction uses the best_model.keras
    return prediction

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict the age of an individual based on an input image.')
    parser.add_argument('image_name', type=str, help='Name of the input image file')
    parser.add_argument('image_path', type=str, help='Path to the input image file')
    parser.add_argument('--save', action='store_true', help='Indicate whether to save the preprocessed image')

    args = parser.parse_args()
    prediction = process(args.image_name, args.image_path, args.save)
    print(prediction)
