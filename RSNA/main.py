# Importing necessary modules
from __init__ import create_directories, check_path
from merge import Merge
from split import Split
from balancing import BalancingDataset
from checker import Checker

def preparatory():
    """
    Function to perform preparatory tasks before merging and splitting datasets.
    """
    # Create necessary directories
    create_directories()
    # Check if the path is valid

def merge_and_split():
    """
    Function to merge and split datasets.
    """
    # Merge datasets
    Merge().merge()
    # Split datasets
    Split().splitting()

def balance():
    """
    Function to balance the dataset.
    """
    # Balance the dataset, creating images if required
    BalancingDataset().balance(create_images=False)

def check():
    """
    Function to perform dataset checking.
    """
    # Check the integrity of the dataset
    Checker().check()

if __name__ == '__main__':
    # Perform preparatory tasks
    preparatory()
    # Merge and split datasets
    merge_and_split()
    # Balance the dataset
    balance()
    # Check the integrity of the dataset
    check()

