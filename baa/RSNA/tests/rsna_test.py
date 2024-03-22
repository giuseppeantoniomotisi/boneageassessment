import os
import shutil
import sys
import unittest
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append(sys.path[0].replace('/tests',''))
sys.path.append(sys.path[0].replace('RSNA/tests',''))
import tools_rsna
import merge
import split
import balancing
import checker

class TestTools(unittest.TestCase):
    def test_create_directories(self):
        # Test if the directories are created properly
        tools_rsna.create_directories()
        # Check if the directories exist
        self.assertTrue(os.path.exists("dataset"))
        self.assertTrue(os.path.exists("dataset/IMAGES"))
        self.assertTrue(os.path.exists("dataset/IMAGES/labels"))
        self.assertTrue(os.path.exists("dataset/IMAGES/raw"))
        self.assertTrue(os.path.exists("dataset/IMAGES/processed"))
        self.assertTrue(os.path.exists("dataset/IMAGES/processed/train"))
        self.assertTrue(os.path.exists("dataset/IMAGES/processed/validation"))
        self.assertTrue(os.path.exists("dataset/IMAGES/processed/test"))
    
class TestMerge(unittest.TestCase):
    def setUp(self):
        # Set up test data
        self.test_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        self.merge_instance = merge.Merge()

    def test_switch_columns(self):
        # Test if columns are switched correctly
        switched_df = merge.switch_columns(self.test_df, 'A', 'B')
        self.assertEqual(list(switched_df.columns), ['B', 'A'])

    def test_change_columns_names(self):
        # Test if column names are changed correctly
        test_df = self.test_df.copy()  # Make a copy of test_df
        merge.change_columns_names(test_df, ['C', 'D'])
        self.assertEqual(list(test_df.columns), ['C', 'D'])

    def test_merge_csv(self):
        # Test if CSV files are merged correctly
        # Mock path to training and validation CSV files
        self.merge_instance.path_to_train = 'test_train.csv'
        self.merge_instance.path_to_val = 'test_val.csv'
        
        # Create test CSV files
        test_train_df = pd.DataFrame({'id': [1, 2, 3],
                                   'boneage': [1, 2, 3],
                                   'male': [1, 2, 3]})
        test_val_df = pd.DataFrame({'id': [4, 5, 6],
                                   'boneage': [4, 5, 6],
                                   'male': [4, 5, 6]})
        test_train_df.to_csv('test_train.csv', index=False)
        test_val_df.to_csv('test_val.csv', index=False)

        # Perform merging
        self.merge_instance.merge_csv()

        # Check if merged file exists and original files are removed
        self.assertTrue(os.path.exists(os.path.join(self.merge_instance.labels, 'dataset.csv')))
        self.assertFalse(os.path.exists('test_train.csv'))
        self.assertFalse(os.path.exists('test_val.csv'))

        # Clean up
        os.remove(os.path.join(self.merge_instance.labels, 'dataset.csv'))

    def test_merge_images(self):
        # Test if images are merged correctly
        # Mock paths to image folders
        self.merge_instance.path_to_train_imgs = 'test_train_images'
        self.merge_instance.path_to_val_imgs = ['test_val_images1', 'test_val_images2']

        # Create test image folders if they don't exist
        for folder in [self.merge_instance.path_to_train_imgs] + self.merge_instance.path_to_val_imgs:
            if not os.path.exists(folder):
                os.makedirs(folder)

        # Perform image merging
        self.merge_instance.merge_images()

        # Check if images are merged correctly
        self.assertFalse(os.path.exists(os.path.join(self.merge_instance.raw, 'test_train_images')))
        self.assertFalse(os.path.exists(os.path.join(self.merge_instance.raw, 'test_val_images1')))
        self.assertFalse(os.path.exists(os.path.join(self.merge_instance.raw, 'test_val_images2')))

        # Clean up
        for folder in [self.merge_instance.path_to_train_imgs] + self.merge_instance.path_to_val_imgs:
            os.rmdir(folder)
            
class TestSplit(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for testing
        self.temp_dir = os.path.join(os.getcwd(), 'temp_test')
        os.makedirs(self.temp_dir, exist_ok=True)

        # Create a temporary dataset.csv file for testing
        self.temp_dataset_csv = os.path.join(self.temp_dir, 'dataset.csv')
        pd.DataFrame({'id': ['img1.png', 'img2.png'], 'label': [0, 1]}).to_csv(self.temp_dataset_csv, index=False)

        # Initialize the Split class with temporary paths
        self.splitter = split.Split()
        self.splitter.labels = self.temp_dir
        self.splitter.raw = self.temp_dir
        self.splitter.processed = self.temp_dir
        self.splitter.dataset = pd.read_csv(self.temp_dataset_csv)

    def test_update_split(self):
        # Test if update_split method updates the split ratios correctly
        self.splitter.update_split(0.6, 0.2, 0.2)
        self.assertEqual(self.splitter.train, 0.6)
        self.assertEqual(self.splitter.val, 0.2)
        self.assertEqual(self.splitter.test, 0.2)

    # def test_splitting(self):
    #     # Test if splitting method creates the expected CSV files and directories
    #     self.splitter.splitting()
    #     test_csv_path = os.path.join(self.temp_dir, 'test.csv')
    #     train_csv_path = os.path.join(self.temp_dir, 'train.csv')
    #     val_csv_path = os.path.join(self.temp_dir, 'validation.csv')

    #     self.assertTrue(os.path.exists(test_csv_path))
    #     self.assertTrue(os.path.exists(train_csv_path))
    #     self.assertTrue(os.path.exists(val_csv_path))

    #     # Check if the total number of rows in the CSV files equals the original dataset length
    #     original_dataset_len = len(self.splitter.dataset)
    #     test_csv_len = len(pd.read_csv(test_csv_path))
    #     train_csv_len = len(pd.read_csv(train_csv_path))
    #     val_csv_len = len(pd.read_csv(val_csv_path))
    #     self.assertEqual(test_csv_len + train_csv_len + val_csv_len, original_dataset_len)

    def tearDown(self):
        # Clean up temporary directory and files
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

class TestBalancingDataset(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for testing
        self.temp_dir = os.path.join(os.getcwd(), 'temp_test')
        os.makedirs(self.temp_dir, exist_ok=True)

        # Create a temporary CSV file for testing
        self.temp_csv = os.path.join(self.temp_dir, 'train.csv')
        with open(self.temp_csv, 'w') as f:
            f.write('id,boneage,male\nimg1.png,10,1\nimg2.png,15,0\nimg3.png,20,1\nimg4.png,25,1\n')
        
        self.balancer = balancing.BalancingDataset()

    def test_rotation(self):
        # Initialize the BalancingDataset class with temporary paths and dependencies
        self.balancer.path_to_images = self.temp_dir

        # Create a dummy image for testing rotation
        test_image = np.zeros((399,399,3))

        # Test the rotation method
        rotated_image = self.balancer.rotation(test_image, 45)

        # Check if the rotated image is of the correct type
        self.assertIsInstance(rotated_image, np.ndarray)

    def tearDown(self):
        # Clean up temporary directory and files
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

if __name__ == '__main__':
    unittest.main()
