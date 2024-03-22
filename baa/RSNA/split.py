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
Split module facilitates the splitting of a dataset into training, validation, and
test sets. Additionally, it organizes images corresponding to these sets into separate directories.
Original splitting is set:

- 70% for training data,
- 20% for validation data,
- 10% per test data.

If you decide to change splitting ratio, first check dimension of dataset and then use the
updat_split method.
"""
import os
import sys
import shutil
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.join(sys.path[0], ''))
from utils import extract_info

class Split:
    """
    Class for splitting a dataset into training, validation, and test sets,
    and organizing corresponding images into separate directories.

    Attributes:
        labels (str): Path to the directory containing label-related information.
        processed (str): Path to the directory where processed data will be stored.
        dataset (pd.DataFrame): DataFrame containing dataset information loaded from 'dataset.csv'.
        train_val (float): Ratio of training and validation data (default is 0.9).
        test (float): Ratio of test data (default is 0.1).
    """
    def __init__(self):
        """
        Initializes the Split class with default attributes and loads the dataset.
        """
        self.temp = os.path.join(extract_info('raw'), 'temp')
        self.labels = extract_info('labels')
        self.processed = extract_info('processed')

        if os.path.exists(os.path.join(self.labels, 'dataset.csv')):
            self.dataset = pd.read_csv(os.path.join(self.labels, 'dataset.csv'))
            # Better shuffle dataset by rows!
            self.dataset = self.dataset.sample(frac=1).reset_index(drop=True)

        self.train = 0.7  # training
        self.val = 0.2 # validation
        self.test = 0.1  # test

    def update_split(self, new_train: float, new_val: float, new_test: float):
        """
        Updates the train_val and test ratios.

        Args:
            new_train (float): New ratio for training data.
            new_val (float): New ratio for validation data.
            new_test (float): New ratio for test data.

        Raises:
            ValueError: If the sum of new_train, new_val, and new_test is not equal to 1,
            or if any of the ratios is not between 0 and 1.
            TypeError: If new_train, new_val, or new_test is not a float.
        """
        # Check if the sum of ratios is equal to 1
        if new_train + new_val + new_test != 1.:
            raise ValueError("The sum of train, val, and test ratios must equal 1.")

        # Check if each ratio is between 0 and 1
        if not all(0 <= ratio <= 1 for ratio in (new_train, new_val, new_test)):
            raise ValueError("Ratios must be between 0 and 1.")

        # Check if each ratio is a float
        if not all(isinstance(ratio, float) for ratio in (new_train, new_val, new_test)):
            raise TypeError("Ratios must be floats.")

        self.train = new_train
        self.val = new_val
        self.test = new_test

    def splitting(self):
        """
        Splits the dataset into training, validation, and test sets, and organizes images into
        separate directories.

        You must check dimension of dataframe before use it; then choose a proper splitter.
        """
        # Splitting dataset
        splitter_test = int(len(self.dataset) * (self.train+self.val))
        splitter_train = int(len(self.dataset) * self.train)
        splitter_val = int(len(self.dataset) * self.val)
        ds_training_validation = self.dataset[:splitter_test]
        ds_test = self.dataset[splitter_test:len(self.dataset)]

        # Better shuffle one more!
        ds_training_validation = ds_training_validation.sample(frac=1).reset_index(drop=True)
        ds_training = ds_training_validation[:splitter_train]
        ds_validation = ds_training_validation[splitter_train:splitter_val+splitter_train]

        # Saving datasets ...
        ds_training.to_csv(os.path.join(self.labels, 'train.csv'), index=False)
        ds_validation.to_csv(os.path.join(self.labels, 'validation.csv'), index=False)
        ds_test.to_csv(os.path.join(self.labels, 'test.csv'), index=False)

        # Copying images to their respective directories
        train = os.path.join(self.processed, 'train')
        val = os.path.join(self.processed, 'validation')
        test = os.path.join(self.processed, 'test')

        for element in tqdm(os.listdir(self.temp)):
            if element in self.dataset['id'].values:
                input_path = os.path.join(self.temp, element)
                if element in ds_training['id'].values:
                    output_path = os.path.join(train, element)
                    if os.path.exists(output_path):
                        pass
                    else:
                        shutil.move(input_path, output_path)
                elif element in ds_validation['id'].values:
                    output_path = os.path.join(val, element)
                    if os.path.exists(output_path):
                        pass
                    else:
                        shutil.move(input_path, output_path)
                elif element in ds_test['id'].values:
                    output_path = os.path.join(test, element)
                    if os.path.exists(output_path):
                        pass
                    else:
                        shutil.move(input_path, output_path)
            else:
                pass

        os.rmdir(self.temp)

    def hist(self, dataframe: pd.DataFrame, col_index: str, nbins: int):
        """
        Generates a histogram for a specific column in the given DataFrame.

        Args:
            dataframe (pd.DataFrame): DataFrame containing data for histogram.
            col_index (str): Name of the column to be used for the histogram.
            nbins (int): Number of bins for the histogram.
        """
        dataframe[col_index].hist(bins=nbins)

if __name__ == '__main__':
    # Create an instance of the Split class
    splitter = Split()

    # Perform the splitting operation
    splitter.splitting()

    print("Splitting operation completed successfully!")
