"""
Merge module code defines functions and a class related to merging CSV files and images for
a dataset project.
Overall, the code is designed to facilitate the management and organization of datasets by merging
separate files into a single dataset file and consolidating images into a single folder.
"""
import os
import pandas as pd
from tools_rsna import extract_info_as_dict

def switch_columns(dataframe: pd.DataFrame, column_name1: str, column_name2: str) -> pd.DataFrame:
    """
    Simple function to switch two columns in a pandas.DataFrame.

    Args:
        dataframe (pandas.DataFrame): Original dataframe.
        column_name1 (str): First column's name.
        column_name2 (str): Second column's name.

    Returns:
        pandas.DataFrame: The old dataframe with two columns swapped.
    """
    i = list(dataframe.columns)
    a, b = i.index(column_name1), i.index(column_name2)
    i[b], i[a] = i[a], i[b]
    dataframe = dataframe[i]
    return dataframe

def change_columns_names(dataframe: pd.DataFrame, new_names: list) -> None:
    """
    Simple function to change column names.

    Args:
        dataframe (pandas.DataFrame): Original dataframe.
        new_names (list): List of strings which contains new columns' names.
    """
    dataframe.columns = new_names

class Merge:
    """
    Class for merging training and validation CSV files into a single dataset CSV file.

    Attributes:
        path_to_train (str): Path to the training CSV file.
        path_to_val (str): Path to the validation CSV file.
    """
    def __init__(self):
        """
        Initializes the Merge class with default attributes for paths to training and
        validation CSV files.
        """
        info = extract_info_as_dict()
        self.path_to_train = info['train.csv']  # Default path to the training CSV file
        # Default path to the training images folder
        self.path_to_train_imgs = info['train_images_path']
        self.path_to_val = info['val.csv']  # Default path to the validation CSV file
        # Default path to the validation images folder
        self.path_to_val_imgs = info['val_images_path']
        # Default path for the dataset project (Desktop/boneageassessment/)
        self.main_dir = info['main_dir']
        self.raw = info['raw']
        self.labels = info['labels']

    def merge_csv(self):
        """
        Merges the training and validation CSV files into a single dataset CSV file.

        Reads the training and validation CSV files, performs necessary data transformations,
        concatenates them into a single dataset, and saves the merged dataset CSV file.
        Removes the original training and validation CSV files after merging.
        """
        # Read training and validation CSV files
        train = pd.read_csv(self.path_to_train)
        val = pd.read_csv(self.path_to_val)

        # Switch columns and change column names for validation data
        if list(val.columns) == ['Image ID', 'male', 'Bone Age (months)']:
            val = switch_columns(val, 'male', 'Bone Age (months)')
            change_columns_names(val, ['id', 'boneage', 'male'])

        # Concatenate training and validation datasets
        merged_dataset = pd.concat([train, val], ignore_index=True)

        # Save the merged dataset CSV file
        merged_name = os.path.join(self.labels, 'dataset.csv')
        merged_dataset['id'] = merged_dataset['id'].apply(lambda x: \
            str(x) + '.png' if '.png' not in str(x) else str(x))
        merged_dataset.to_csv(merged_name, index=False)

        # Remove CSV files for training and validation
        os.remove(self.path_to_train)
        os.remove(self.path_to_val)

    def merge_images(self):
        """
        Merges images from training and validation folders to a single folder.
        """
        input_folders = [self.path_to_train_imgs,
                         self.path_to_val_imgs[0],
                         self.path_to_val_imgs[1]]
        output_folder = self.raw
        for input_folder in input_folders:
            for element in os.listdir(input_folder):
                source = os.path.join(input_folder, element)
                destination = os.path.join(output_folder, element)
                os.replace(source, destination)

    def merge(self):
        """
        Merges both CSV files and images.
        """
        self.merge_csv()
        self.merge_images()

if __name__ == '__main__':
    # Create an instance of the Merge class
    merger = Merge()
    
    # Merge CSV files
    merger.merge_csv()
    
    # Merge images
    merger.merge_images()
    
    print("Merge operation completed successfully!")
