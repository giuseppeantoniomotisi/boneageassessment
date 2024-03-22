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

"""Predict the age of an individual based on an input image.
"""
import argparse
import os
import cv2
from utils import extract_info
from preprocessing.tools import Preprocessing
from age.model import BoneAgeAssessment

def process(image_name:str, image_path:str, save:bool=True, prep:bool=True) -> str:
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
    if prep:
        prep_instance = Preprocessing()
        img = prep_instance.preprocessing_image(image_path=image, save=save, show=False)
        img /= 255.

    else:
        image = os.path.join(image_path, image_name)
        img = cv2.imread(image)
        img /= 255.

    # Now load model and make prediction
    baa_instance = BoneAgeAssessment()
    prediction = baa_instance.prediction(img, show=True, save=save, image_id=image_name[:-4])
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
