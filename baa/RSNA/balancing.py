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
Balancing module defines a class BalancingDataset that is designed to balance a dataset by augmenting
the training data with additional samples. The process involves creating a balanced CSV file and,
optionally, generating augmented images.

This method balances the dataset by calling create_bal_csv to create the balanced CSV file.
If create_images is set to True, it further calls create_images to generate and save augmented
images.

In summary, this script is designed to balance a dataset by creating a balanced CSV file and
optionally generating augmented images for training. The augmentation involves introducing random
rotations to the images.
"""
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2 as cv
from tools_rsna import extract_info_as_dict

class BalancingDataset:
    """
    BalancingDataset class handles the process of balancing a dataset by augmenting the
    training data with additional samples and creating balanced CSV and image files.

    Attributes:
        path_to_csv (str): Path to the directory containing CSV files.
        path_to_images (str): Path to the directory containing raw images.
        path_to_train (str): Path to the processed training folder.
        csv_name (str): Name of the training CSV file.
        dataframe (pd.DataFrame): Pandas DataFrame containing the training data.

        key_id (str): Column name for identifiers in the DataFrame.
        key_value (str): Column name for values used for categorization.

    Methods:
        find_element(limit_value, num_classes) -> tuple:
            Categorizes elements in a DataFrame into classes based on their values, and calculates
            delta values
            based on the histogram of key_values column and limit_value.

        create_bal_csv():
            Creates a balanced CSV file by augmenting the training data with additional samples and
            saves it.

        rotation(image, angle) -> np.ndarray:
            Rotates the image by a given angle.

        create_images():
            Creates augmented images based on the balanced CSV file and saves them to the
            processed training folder.

        balance(create_images: bool = False):
            Balances the dataset by creating a balanced CSV file and optionally generating
            augmented images.

    """
    def __init__(self):
        """
        Initializes BalancingDataset class with default paths and dependencies.
        """
        # find all dependicies
        info = extract_info_as_dict()
        self.path_to_csv = info['labels']
        self.path_to_images = info['raw']
        self.path_to_train = os.path.join(info['processed'],'train')
        self.csv_name = 'train.csv'

        if os.path.exists(os.path.join(self.path_to_csv, self.csv_name)):
            dataframe = pd.read_csv(os.path.join(self.path_to_csv, self.csv_name))
            dataframe.loc[:,'angle'] = np.zeros(dataframe.shape[0])
            self.dataframe = dataframe

        self.key_id = 'id'
        self.key_value = 'boneage'
        
    def __update_df__(self, new_dataframe_path):
        dataframe = pd.read_csv(new_dataframe_path)
        dataframe.loc[:,'angle'] = np.zeros(dataframe.shape[0])
        self.dataframe = dataframe

    def find_element(self, limit_value, num_classes) -> tuple:
        """
        Summary:
            The function find_element seems designed to categorize elements in a DataFrame into
            classes based on their values in a particular column, key_values.
            Additionally, it calculates the delta values based on the histogram of key_values
            column and limit_value.

        Args:
            dataframe (pd.DataFrame): A pandas DataFrame containing the data.
            key_id (str): The name of the column in the DataFrame containing the identifiers.
            key_values (str): The name of the column in the DataFrame containing the values to be
            categorized.
            limit_value (int): The maximum number of elements allowed in each class.
            num_classes (int): The number of classes to categorize the data into.

        Raises:
            TypeError: If dataframe is not of type pd.DataFrame.
            TypeError: If limit_value is not of type int.
            TypeError: If num_classes is not of type int.
            ValueError: If key_id or key_values are not found in the DataFrame.
            ValueError: If limit_value is less than the maximum histogram value.

        Returns:
            tuple: A tuple containing a dictionary representing classes and an array representing
            the delta values.
                The dictionary (dict): Keys represent the classes, and values represent the
                elements in each class.
                The array (numpy.ndarray): Represents the difference between the limit_value and
                the histogram values.
        """
        # Check self.dataframe type
        if type(self.dataframe) != pd.DataFrame:
            type_error = f"dataframe's expeted type is" + \
            "<class 'pandas.core.frame.DataFrame'>,not {type(self.dataframe)}"
            raise TypeError(type_error)

        # Check limit_value type
        if type(limit_value) != int:
            raise TypeError(f"limit_value's expeted type is <class 'int'>, not {type(limit_value)}")

        # Check num_classes type
        if type(num_classes) != int:
            raise TypeError(f"num_classes's expeted type is <class 'int'>, not {type(num_classes)}")

        # Create empty lists to append image id
        classes_value = [[] for _ in range(num_classes)]
        classes_key = np.arange(0, num_classes)
        classes = dict(zip(classes_key, classes_value))

        # Calculate histogram
        hist, edges = np.histogram(self.dataframe[self.key_value], bins=num_classes)

        # Check if limit_value is greater than the maximum bin value
        if limit_value <= np.max(hist):
            raise ValueError(f"limit_value must be greater or equal to max_hist: {np.max(hist)}!")

        # Define how much is distant the desired number of images
        delta_array = limit_value - hist

        # Assign elements to classes based on their values
        for i in range(len(self.dataframe)):
            for j in range(num_classes):
                if edges[j] <= self.dataframe[self.key_value][i] <= edges[j + 1]:
                    classes[j].append(self.dataframe[self.key_id][i])
                    break

        return classes, delta_array

    def create_bal_csv(self, limit):
        """
        Creates a balanced CSV file by augmenting the training data with additional
        samples and saves it.
        """
        num = 19
        list_of_names, delta_balanced = self.find_element(limit, num)

        temp_id, temp_boneage, temp_gender, temp_bn, temp_angles = [], [], [], [], []
        for i in range(len(list_of_names)):
            print(f'Creating {delta_balanced[i]} files for [{i}, {i+1}) age range ...')
            for j in range(delta_balanced[i]):
                # Draw name to transform
                selected_name = np.random.choice(list_of_names[i])
                idx = np.where(selected_name == self.dataframe['id'])[0]

                # Generate a random angle to rotate image
                angle = np.random.rand() * 360

                angle_name = str("%.3f" % angle).replace('.', '')
                new_name = f'{angle_name}{selected_name}'  # Adding angle to the name

                # Add name to csv file
                temp_id.append(new_name)
                temp_angles.append(angle)
                temp_boneage.append(self.dataframe['boneage'].values[idx][0])
                temp_gender.append(self.dataframe['male'].values[idx][0])

            print('Done!')

        temp_df = pd.DataFrame({'id':temp_id,
                                'boneage':temp_boneage,
                                'male':temp_gender,
                                'angle':temp_angles})

        dataframe_aug = pd.concat([self.dataframe, temp_df], ignore_index=True)
        dataframe_aug_name = os.path.join(self.path_to_csv, 'augmented.csv')
        dataframe_aug.to_csv(dataframe_aug_name, index=False)

    @staticmethod
    def rotation(image, angle) -> np.ndarray:
        """
        Summary:
            Rotate the image by a given angle.

        Args:
            image (numpy.ndarray): The input image.
            angle (float): The angle by which the image should be rotated.

        Returns:
            numpy.ndarray: The rotated image.
        """
        size_reverse = np.array(image.shape[1::-1]) # swap x with y
        M = cv.getRotationMatrix2D(tuple(size_reverse / 2.), angle, 1.)
        MM = np.absolute(M[:,:2])
        size_new = MM @ size_reverse
        M[:,-1] += (size_new - size_reverse) / 2.
        return cv.warpAffine(image, M, tuple(size_new.astype(int)))
    
    def create_images(self):
        """
        Creates augmented images based on the balanced CSV file and saves them to the processed
        training folder.
        """
        from_directory = self.path_to_images
        to_directory = self.train_dir
        dataframe_aug_name = os.path.join(self.path_to_csv, 'augmented.csv')
        dataframe = pd.read_csv(dataframe_aug_name)
        iterations = dataframe.shape[0]

        for i in tqdm(range(iterations)):
            name = dataframe['id'][i]
            input_path = os.path.join(from_directory, name)
            output_path = os.path.join(to_directory, name)

            if os.path.exists(output_path):
                 pass
            else:
                if dataframe['angle'][i] == 0.:
                    shutil.copyfile(input_path, output_path)

                else:
                    angle = dataframe['angle'][i]

                    angle_name = str("%.3f" % angle).replace('.','')
                    name = dataframe['id'][i].replace(angle_name, '')
                    input_path = os.path.join(from_directory, name)

                    name = dataframe['id'][i]
                    output_path = os.path.join(to_directory, name)

                    image = cv.imread(input_path, 1)
                    rotated_image = self.otation(image, angle)

                    plt.figure(frameon=False)
                    plt.imshow(rotated_image, cmap='gray')
                    plt.axis('off')
                    plt.tight_layout()
                    plt.savefig(output_path)
                    plt.close()

                    del angle_name, image, rotated_image
    
    def balance(self, create_images:bool=False):
        """
        Balances the dataset by creating a balanced CSV file and optionally generating
        augmented images.

        Args:
            create_images (bool): If True, generates and saves augmented images.
        """
        # First create csv file
        self.create_bal_csv(limit=1500)
        
        # Then creates image
        if create_images:
            self.create_images()

if __name__ == '__main__':
    BalancingDataset().balance(create_images=False)
