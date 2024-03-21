"""
Checker module defines a class called Checker which is responsible for checking the contents of
specified folders and CSV files.

Overall, the Checker class is designed to provide a simple mechanism for verifying the contents of
specific folders and CSV files, which can be useful in various data processing and validation
tasks.
"""
import os
import pandas as pd
from tools_rsna import extract_info_as_dict

class Checker:
    """The class Checker is a simple class for evaluate dataset transformation."""
    def __init__(self):
        """
        Initialize Checker class with default paths and dependencies.

        The class extracts information from the initialization module (__init__)
        and sets default directories and CSV file names.
        """
        info = extract_info_as_dict()
        self.path_to_processed = info['processed']
        self.path_to_csv = info['labels']
        self.default_dir = ['train', 'validation', 'test']
        self.default_csv = ['train.csv', 'validation.csv', 'test.csv']

    def update_dir(self, new_default_dir):
        """
        Update the default directories with the provided list.

        Args:
            new_default_dir (list): New list of default directories.
        """
        self.default_dir = new_default_dir

    def update_csv(self, new_default_csv):
        """
        Update the default CSV file names with the provided list.

        Args:
            new_default_csv (list): New list of default CSV file names.
        """
        self.default_csv = new_default_csv

    def check(self):
        """
        Check the contents of specified folders and CSV files.

        Iterates through the default directories and CSV files, printing
        the number of elements in each folder and the number of rows in
        each CSV file.
        """
        folders = self.default_dir
        csvs = self.default_csv

        print('---')
        for i in enumerate(csvs):
            folder_path = os.path.join(self.path_to_processed, folders[i])
            print(f'The {folders[i]} folder contains {len(os.listdir(folder_path))} elements')

            csv_path = os.path.join(self.path_to_csv, csvs[i])
            dataframe = pd.read_csv(csv_path)
            print(f'The {csvs[i]} contains {dataframe.shape[0]} rows')
            print('---')

if __name__ == '__main__':
    Checker().check()
