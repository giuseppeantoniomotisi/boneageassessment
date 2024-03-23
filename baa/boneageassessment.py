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
Bone Age Assessment System

This script provides a modular system for Bone Age Assessment (BAA). 
It integrates various modules such as data preprocessing, machine learning, and prediction. 
Users can configure the behavior of the system using a JSON configuration file.

Usage:
`
    python baa/boneageassessment.py --macro <path_to_parameters_json>

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
import shutil
sys.path.append(os.path.join(sys.path[0], 'RSNA'))
sys.path.append(os.path.join(sys.path[0], 'preprocessing'))
sys.path.append(os.path.join(sys.path[0], 'age'))

import utils
from RSNA import rsna
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
    downloads_dir = utils.get_downloads()
    
    if opt:
        print("We are manipulating dataset. Please wait.")
        rsna.process()
        utils.houdini()
        print("Done!")
        print("We are processing images. Please wait.")
        preprocessing.process()
        print("Done!")
        print("We are splitting dataset. Please wait.")
        rsna.split()
        print("Done!")
        print("We are balancing dataset per year. Please wait.")
        rsna.balance()
        print("Done!")

    else:
        downloads_dir = utils.get_downloads()

        if 'dataset.zip' in os.listdir(downloads_dir):
            if os.name != 'posix':
                error = "you work on Windows. The shell command 'unzip' works only for MACOS and Linux.\n"
                error += "Please unzip by your UtilityCompressor and retry."
                raise NotImplementedError(error)

            print("We are unzipping dataset folder. Please wait.")
            os.system(f"unzip {os.path.join(downloads_dir, 'dataset.zip')}")
            if os.path.exists('__MACOSX'):
               os.system(f"rm -r '__MACOSX')")
            os.remove(os.path.join(downloads_dir, 'dataset.zip'))
            print("Done!")

        elif 'dataset' in os.listdir(downloads_dir):
            pass

        elif os.path.exists(os.path.join(os.getcwd(), 'dataset')):
            pass
        
        elif 'dataset_lite.zip' in os.listdir(downloads_dir):
            if os.name != 'posix':
                error = "you work on Windows. The shell command 'unzip' works only for MACOS and Linux.\n"
                error += "Please unzip by your UtilityCompressor and retry."
                raise NotImplementedError(error)

            print("We are unzipping dataset_lite folder. Please wait.")
            src = os.path.join(downloads_dir, 'dataset_lite.zip')
            dest = os.path.join(os.getcwd(), 'dataset')
            #shutil.unpack_archive(src, dest, format="zip")
            os.system(f"unzip {src}")
            os.remove(os.path.join(downloads_dir, 'dataset_lite.zip'))
            print("Done!")

        else: 
            raise FileNotFoundError("no file named dataset.zip or folder named dataset was found.")

    if 'weights.zip' in os.listdir(downloads_dir):
        if os.name != 'posix':
            error = "you work on Windows. The shell command 'unzip' works only for MACOS and Linux.\n"
            error += "Please unzip by your UtilityCompressor and retry."
            raise NotImplementedError(error)

        print("We are unzipping weights folder. Please wait.")
        os.system(f"unzip {os.path.join(downloads_dir, 'weights.zip')} -d {os.path.join(os.getcwd(), 'baa', 'age')}")
        # if os.path.exists(os.path.join(os.getcwd(), 'baa', 'age', '__MACOSX')):
        #     shutil.rmtree(os.path.join(os.getcwd(), 'baa', 'age', '__MACOSX'))
        os.remove(os.path.join(downloads_dir, 'weights.zip'))
        print("Done!")

    elif 'weights' in os.listdir(downloads_dir):
        pass
    
    elif os.path.exists(os.path.join(os.getcwd(), 'baa', 'age', 'weights')):
        pass

    else: 
        raise FileNotFoundError("no file named weights.zip or folder named weights was found.")
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
        prediction_result = prediction.process(name, path)
        print(prediction_result)
        prediction_result = 0
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
            'Training and testing model': False,
            'Path to hyperparameters.json': '../baa/age/age_macro.json',
            'New prediction': False,
            'New image name': 'image.png',
            'Path to new image': '../../',
        }

    preprocessing_module(info_json['RSNA'])
    machinelearning_module(info_json['Training and testing model'], info_json["Path to hyperparameters.json"])
    image, path = info_json['New image name'], info_json['Path to new image']
    prediction_module(info_json['New prediction'], image, path)


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
