import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
#import tensorflow.keras.backend as K

def dataset_gender_splitter(training_path: str, csv_file: str, split = True):
    if split == True:
        # Create directories for male and female images if they don't exist
        male_dir = os.path.join(training_path, 'male')
        female_dir = os.path.join(training_path, 'female')

        if not os.path.exists(male_dir):
            os.makedirs(male_dir)
        if not os.path.exists(female_dir):
            os.makedirs(female_dir)

        # Read the CSV file into a DataFrame
        ds = pd.read_csv(os.path.join(training_path, csv_file))

        # Iterate through each row of the DataFrame
        for _, row in ds.iterrows():
        # Extract filename and determine gender
            filename = f"{row['id']}.png"
         # Determine the destination directory based on gender
            dest_dir = male_dir if row['male'] else female_dir
        # Move the image file to the appropriate directory
            os.replace(os.path.join(training_path, filename), os.path.join(dest_dir, filename))
