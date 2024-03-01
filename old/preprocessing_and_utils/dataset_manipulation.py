"""
"""
import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2 as cv
from imageprocessing import ImageProcessing

def find_element(dataframe: pd.DataFrame, key_id: str, key_values: str, limit_value: int, num_classes: int) -> tuple:
    """
    Summary:
        The function find_element seems designed to categorize elements in a DataFrame into classes based on
        their values in a particular column, key_values. Additionally, it calculates the delta values based on
        the histogram of key_values column and limit_value

    Args:
        dataframe (pd.DataFrame): A pandas DataFrame containing the data.
        key_id (str): The name of the column in the DataFrame containing the identifiers.
        key_values (str): The name of the column in the DataFrame containing the values to be categorized.
        limit_value (int): The maximum number of elements allowed in each class.
        num_classes (int): The number of classes to categorize the data into.

    Raises:
        TypeError: If dataframe is not of type pd.DataFrame.
        TypeError: If limit_value is not of type int.
        TypeError: If num_classes is not of type int.
        ValueError: If key_id or key_values are not found in the DataFrame.
        ValueError: If limit_value is less than the maximum histogram value.

    Returns:
        tuple: A tuple containing a dictionary representing classes and an array representing the delta
        values.
               - The dictionary (dict): Keys represent the classes, and values represent the elements
               in each class.
               - The array (numpy.ndarray): Represents the difference between the limit_value and the histogram values.
    """
    # Check dataframe type
    if type(dataframe) != pd.DataFrame:
        raise TypeError(f"dataframe's expeted type is <class 'pandas.core.frame.DataFrame'>, not {type(dataframe)}")

    # Check limit_value type
    if type(limit_value) != int:
        raise TypeError(f"limit_value's expeted type is <class 'int'>, not {type(limit_value)}")

    # Check num_classes type
    if type(num_classes) != int:
        raise TypeError(f"num_classes's expeted type is <class 'int'>, not {type(num_classes)}")

    # Check if key_id and key_values exist in the dataframe
    if key_id not in dataframe.columns or key_values not in dataframe.columns:
        raise ValueError("key_id or key_values not found in the pandas.DataFrame!")

    # Create empty lists to append image id
    classes_value = [[] for _ in range(num_classes)]
    classes_key = np.arange(0, num_classes)
    classes = dict(zip(classes_key, classes_value))

    # Calculate histogram
    hist, edges = np.histogram(dataframe[key_values], bins=num_classes)

    # Check if limit_value is greater than the maximum bin value
    if limit_value < np.max(hist):
        raise ValueError(f"limit_value must be greater than max_hist: {np.max(hist)}!")

    # Define how much is distant the desired number of images
    delta_array = limit_value - hist

    # Assign elements to classes based on their values
    for i in range(len(dataframe)):
        for j in range(num_classes):
            if edges[j] <= dataframe[key_values][i] <= edges[j + 1]:
                classes[j].append(dataframe[key_id][i])  # Adjusted the key from j + 1 to j
                break

    return classes, delta_array

def dataset_organizer(dataframe:pd.DataFrame,from_directory,to_directory,balancing:bool=False) -> None:
    """
    Process images based on data from a training dataframe.

    Args:
    - dataframe: DataFrame containing image information. Dataframe just contains names of new files
    and angles columns, so it is necessary to run find_element before.
    - from_directory: Directory containing source images.
    - to_directory: Directory to save processed images.
    
    Returns:
    None
    """
    if balancing:
        for i in tqdm(range(len(dataframe))):
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
                    angle_name = str("%.3f" % angle).replace('.', '')
                    name = dataframe['id'][i].replace(angle_name, '')
                    input_path = os.path.join(from_directory, name)
                    name = dataframe['id'][i]
                    output_path = os.path.join(to_directory, name)

                    image = cv.imread(input_path, 1)
                    rotated_image = ImageProcessing().rotate_image(image, angle)

                    plt.figure(frameon=False)
                    plt.imshow(rotated_image, cmap='gray')
                    plt.axis('off')
                    plt.tight_layout()
                    plt.savefig(output_path)
                    plt.close()

                    del angle_name, image, rotated_image
    else:
        for i in tqdm(range(len(dataframe))):
            name = dataframe['id'][i]
            input_path = os.path.join(from_directory, name)
            output_path = os.path.join(to_directory, name)
            if os.path.exists(output_path):
                 pass
            else:
                shutil.copyfile(input_path, output_path)

def dataset_gender_splitter(send_dir: str, csv_file: str, split: bool = True) -> None:
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
    print(0)
