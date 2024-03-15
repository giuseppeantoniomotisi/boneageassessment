"""
Bone Age Assessment System

This script provides a modular system for Bone Age Assessment (BAA). 
It integrates various modules such as data preprocessing, machine learning, and prediction. 
Users can configure the behavior of the system using a JSON configuration file.

Usage:
    python script.py --macro <path_to_parameters_json>

Parameters:
    --macro <path_to_parameters_json>: Path to the JSON configuration file containing parameters
    for the BAA process.

Example:
    python script.py --macro config.json
"""
import argparse
import json
import os
import sys
sys.path.append(os.path.join(sys.path[0], 'RSNA'))
sys.path.append(os.path.join(sys.path[0], 'preprocessing'))
sys.path.append(os.path.join(sys.path[0], 'age'))

import utils
from RSNA import rsna
from RSNA import tools_rsna
from preprocessing import preprocessing
from age import age
import prediction

def preprocessing_module(opt:bool):
    """
    Executes data preprocessing tasks.

    Args:
        opt (bool): If True, executes preprocessing tasks.

    Raises:
        FileNotFoundError: If required files/folders are not found.
    """
    print("Preliminary operation module is running. Please wait.")
    utils.write_info()
    if opt:
        print("We are manipulating dataset. Please wait.")
        rsna.process()
        print("Done!")
        print("We are processing images. Please wait.")
        preprocessing.process()
        print("Done!")
    else:
        utils.open_downloads()
        if 'dataset.zip' in os.listdir(os.getcwd()):
            print("We are unzipping dataset folder. Please wait.")
            tools_rsna.unzip_folder('dateset.zip')
            print("Done!")
        elif 'dataset' in os.listdir(os.getcwd()):
            pass
        else: 
            raise FileNotFoundError("no file named dataset.zip or folder named dataset was found.")
    utils.houdini()
    print("Done!")

def machinelearning_module(opt:bool, hyperparameters_json):
    """
    Executes machine learning tasks.

    Args:
        opt (bool): If True, executes machine learning tasks.
    """
    if opt:
        print("Machine Learning module is running. Please wait.")
        hyper = json.load(hyperparameters_json)
        age.process(hyper)
        print("Done!")
    else:
        pass

def prediction_module(opt:bool,name:str,path:str):
    """
    Executes prediction tasks.

    Args:
        opt (bool): If True, executes prediction tasks.
        name (str): Name of the image file.
        path (str): Path to the image file.

    Returns:
        prediction_result (type): The result of prediction.
    """
    if opt:
        if not os.path.exists(os.path.join(path, name)):
            raise FileNotFoundError("image was not found! Please check it.")
        prediction_result = prediction.process(name,path)
        prediction_result = 0
        return prediction_result
    else:
        pass

def baa(info_json):
    """
    Executes the complete Bone Age Assessment process.

    Args:
        info_json (dict): Dictionary containing configuration parameters.
    """
    if info_json == None:
        info_json = {
            'RSNA': False,
            'training': False,
            'prediction': False,
            'image_name': 'image.png',
            'path_to_image': '../../',
        }
    preprocessing_module(info_json['RSNA'])
    machinelearning_module(info_json['training'])
    image, path = info_json['image'], info_json['path_to_image']
    prediction_module(info_json['prediction'], image, path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bone Age Assessment Process')
    parser.add_argument('--macro', type=str, help='Path to the parameters JSON file')
    args = parser.parse_args()

    if args.macro:
        with open(args.macro, 'r') as file:
            parameters = json.load(file)
    else:
        raise ImportError("check if a json file with correct macros is in directory.")

    baa(parameters)
