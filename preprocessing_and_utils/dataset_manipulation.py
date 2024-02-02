"""This Python code defines a function named dataset_gender_splitter that organizes image
datasets based on gender information provided in a CSV file. It also provides the functionality
to either split or merge the dataset based on the split parameter.

Here's a breakdown of how the code works:

The function dataset_gender_splitter takes three parameters:

send_dir: The directory path where the images and CSV file are located.
csv_file: The name of the CSV file containing information about the images.
split: A boolean parameter indicating whether to split the dataset based on gender.

It reads the CSV file into a DataFrame using pandas.read_csv().
It creates two directories for male and female images inside the send_dir directory if
they don't already exist. If split is True, it iterates through the rows of the DataFrame
and moves the image files to the appropriate directory based on the gender information
provided in the CSV file. If split is False, it reverses the process by moving the images 
back to the source directory from their respective male or female directories.
If merging the dataset, it removes the empty male and female directories.

In the if __name__ == '__main__': block, the code specifies the sender directory
(SENDER) and the CSV filename (FILENAME) and then calls the dataset_gender_splitter function
with the specified parameters.

Usage:
To use this code, you need to provide the directory where the images and CSV file are located,
the name of the CSV file, and optionally specify whether to split or merge the dataset based
on gender information. You can call the function dataset_gender_splitter with appropriate
parameters to organize your dataset based on gender.
"""

import os
import pandas as pd

def dataset_gender_splitter(send_dir: str, csv_file: str, split: bool = True):
    """
    Organizes image datasets based on gender information provided in a CSV file.

    This function reads a CSV file containing gender information associated with image files.
    It organizes the image files into separate directories based on gender.
    If `split` is True, it splits the dataset into separate directories for male and female images.
    If `split` is False, it merges the dataset, moving images from male and female directories back
    to the source directory.

    Args:
    - send_dir (str): The directory path where the images and CSV file are located.
    - csv_file (str): The name of the CSV file containing information about the images.
    - split (bool, optional): If True, split the dataset based on gender; if False, merge it.
    Default is True.

    Returns:
    None
    """
    # Read the CSV file into a DataFrame
    dataset = pd.read_csv(os.path.join(send_dir, csv_file))

    # Create directories for male and female images if they don't exist
    male_dir = os.path.join(send_dir, 'male')
    female_dir = os.path.join(send_dir, 'female')

    if not os.path.exists(male_dir):
        os.makedirs(male_dir)
    if not os.path.exists(female_dir):
        os.makedirs(female_dir)

    if split:
        # Split dataset based on gender
        for _, row in dataset.iterrows():
            # Extract filename and determine gender
            filename = f"{row['id']}.png"
            # Determine the destination directory based on gender
            dest_dir = male_dir if row['male'] else female_dir
            # Move the image file to the appropriate directory
            os.replace(os.path.join(send_dir, filename), os.path.join(dest_dir, filename))
    else:
        # Merge dataset
        for _, row in dataset.iterrows():
            # Extract filename and determine gender
            filename = f"{row['id']}.png"
            # Determine the destination directory based on gender
            dest_dir = male_dir if row['male'] else female_dir
            # Move the image file to the appropriate directory
            os.replace(os.path.join(dest_dir, filename), os.path.join(send_dir, filename))

        # Remove empty male and female directories
        os.remove(male_dir)
        os.remove(female_dir)


if __name__ == '__main__':
    SENDER = '../boneage-training-dataset' #boneage-training-dataset
    FILENAME = 'train.csv' #csv file
    dataset_gender_splitter(send_dir=SENDER, csv_file=FILENAME, split=True)
