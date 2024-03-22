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
rsna.py is the main script that must be run to obtain the new dataset structure.
"""
from tools_rsna import create_directories, clean_workspace, check_path
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
    check_path()

def merge():
    """
    Function to merge datasets.
    """
    # Merge datasets
    Merge().merge()
    
def split():
    """
    Function to split datasets.
    """
    # Split datasets
    Split().splitting()

def balance():
    """
    Function to balance the dataset.
    """
    # Balance the dataset, creating images if required
    BalancingDataset().balance(create_images=True)

def check():
    """
    Function to perform dataset checking.
    """
    # Check the integrity of the dataset
    Checker().check()

def process():
    # Perform preparatory tasks
    preparatory()
    # Merge and split datasets
    merge() # Forse dovrei fare prima il preprocessing dello splitting!!!
    # Balance the dataset
    # balance() Il balancing solo dopo il preprocessing!!!
    # Check the integrity of the dataset
    # check()
    # Now clean workspace
    clean_workspace()

if __name__ == '__main__':
    process()
