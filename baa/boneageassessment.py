"""
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
    """_summary_

    Args:
        opt (bool): _description_

    Raises:
        FileNotFoundError: _description_
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

def machinelearning_module(opt:bool):
    """_summary_

    Args:
        opt (bool): _description_
    """
    if opt:
        print("Machine Learning module is running. Please wait.")
        age.process()
        print("Done!")
    else:
        pass

def prediction_module(opt:bool,name:str,path:str):
    """_summary_

    Args:
        opt (bool): _description_
        name (str): _description_
        path (str): _description_

    Returns:
        _type_: _description_
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
    """_summary_
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
