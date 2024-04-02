"""
Some useful functions.

This module contains a collection of utility functions for various tasks such as file operations, directory manipulation,
and data processing.

Functions:
    - append_in_json(key: str, value, json_file: str): Append a key-value pair in a JSON file.
    - write_in_macro(): Write the current working directory path into a JSON file.
    - open_boneageassessment(naive: bool = True): Open the boneageassessment directory.
    - unzip_sh(path_to_zip: str): Unzip a file using the system's shell.
    - open_downloads(): Open the Downloads directory based on the current platform.
    - get_downloads(): Get the Downloads directory path based on the current platform.
    - switch_folders(folder1_path: str, folder2_path: str): Switch the contents of two folders.
    - houdini(opt: str = 'dataset'): Switch two folders with the same name.
    - write_info(): Write paths of main directories in boneageassessment.
    - extract_info(key: str): Extract information from a CSV file.

"""
import os
import json
import pwd
import platform
import shutil
from warnings import warn
from csv import reader
from tqdm import tqdm

def append_in_json(key: str, value, json_file: str):
    """Append key-value pair in a JSON file.

    Args:
        key (str): Key to append.
        value: Value corresponding to the key.
        json_file (str): Path to the JSON file.
    """
    new_line = {key: value}
    with open(json_file, 'r') as f:
        temp = json.load(f)
    temp.update(new_line)
    with open(json_file, 'w') as f:
        json.dump(temp, f, ensure_ascii=False, indent=4)

def write_in_macro():
    """Write current working directory path into a JSON file."""
    v = os.getcwd()
    k = "Path to boneageassessment"
    macro = os.path.join(v, 'baa', 'macro.json')
    append_in_json(key=k, value=v, json_file=macro)

def open_boneageassessment(naive: bool = True):
    """Open the boneageassessment directory.

    Args:
        naive (bool, optional): Whether to open naively. Defaults to True.
    """
    if naive:
        os.chdir(os.getcwd())
    else:
        raise NotImplementedError

def unzip_sh(path_to_zip):
    """Unzip a file using the system's shell.

    Args:
        path_to_zip (str): Path to the ZIP file.
    """
    if os.name == 'posix':
        os.system(f"unzip {path_to_zip}")

def open_downloads():
    """Open the Downloads directory based on the current platform."""
    username = pwd.getpwuid(os.getuid()).pw_name
    if platform.system() == 'Windows':
        path = 'C:/Users/username/Downloads'.replace('/username/', f'/{username}/')
    elif platform.system() == 'Darwin':
        path = '/Users/username/Downloads'.replace('/username/', f'/{username}/')
    elif platform.system() == 'Linux':
        path = '/home/username/Downloads'.replace('/username/', f'/{username}/')
    os.chdir(path)

def get_downloads():
    """Get the Downloads directory path based on the current platform."""
    username = pwd.getpwuid(os.getuid()).pw_name
    if platform.system() == 'Windows':
        path = 'C:/Users/username/Downloads'.replace('/username/', f'/{username}/')
    elif platform.system() == 'Darwin':
        path = '/Users/username/Downloads'.replace('/username/', f'/{username}/')
    elif platform.system() == 'Linux':
        path = '/home/username/Downloads'.replace('/username/', f'/{username}/')
    return path

def switch_folders(folder1_path, folder2_path):
    """Switch the contents of two folders.

    Args:
        folder1_path (str): Path to the first folder.
        folder2_path (str): Path to the second folder.
    """
    temp_dir = os.path.join(os.path.dirname(folder1_path), 'temp')
    os.makedirs(temp_dir, exist_ok=True)

    for item in tqdm(os.listdir(folder1_path)):
        item_path = os.path.join(folder1_path, item)
        shutil.move(item_path, temp_dir)

    for item in tqdm(os.listdir(folder2_path)):
        item_path = os.path.join(folder2_path, item)
        shutil.move(item_path, folder1_path)

    for item in tqdm(os.listdir(temp_dir)):
        item_path = os.path.join(temp_dir, item)
        shutil.move(item_path, folder2_path)

    os.rmdir(temp_dir)

def houdini(opt: str = 'dataset'):
    """Switch two folders with the same name.

    Args:
        opt (str, optional): Type of folder to switch. Defaults to 'dataset'.
            Options: 'dataset' and 'weights'.
    """
    if opt == 'dataset':
        extract_info('main')
        new_loc = os.path.join(os.getcwd(), 'dataset')

        down = get_downloads()
        old_loc = os.path.join(down, 'dataset')

        shutil.move(old_loc, new_loc)
    elif opt == 'weights':
        extract_info('main')
        new_loc_w = os.path.join(os.getcwd(), 'baa', 'age', 'weights')

        down = get_downloads()

        old_loc_w = os.path.join(down, 'weights')
        shutil.move(old_loc_w, new_loc_w)
    else:
        raise KeyError("only 'dataset' and 'weights' are supported. Please check input.")

def write_info():
    """Write paths of main directories in boneageassessment."""
    open_boneageassessment()
    main_baa_dir = os.getcwd()
    baa_dir = os.path.join(main_baa_dir, 'baa')
    rsna = os.path.join(baa_dir, 'RSNA')
    preprocessing = os.path.join(baa_dir, 'preprocessing')
    age = os.path.join(baa_dir, 'age')

    IMAGES_dir = os.path.join(main_baa_dir, 'dataset', 'IMAGES')
    labels_dir = os.path.join(IMAGES_dir, 'labels')
    raw_dir = os.path.join(IMAGES_dir, 'raw')
    processed_dir = os.path.join(IMAGES_dir, 'processed')
    train_dir = os.path.join(processed_dir, 'train')
    validation_dir = os.path.join(processed_dir, 'val')
    test_dir = os.path.join(processed_dir, 'test')

    info = {'main': main_baa_dir,
            'baa': baa_dir,
            'IMAGES': IMAGES_dir,
            'labels': labels_dir,
            'raw': raw_dir,
            'processed': processed_dir,
            'train': train_dir,
            'validation': validation_dir,
            'test': test_dir,
            'rsna': rsna,
            'preprocessing': preprocessing,
            'age': age}

    filename = os.path.join(baa_dir, 'info.csv')
    with open(filename, 'w+', newline='\n') as fp:
        fp.write("Dictionary key,Path to folder\n")
        for key in info.keys():
            fp.write(f"{key},{info[key]}\n")

def extract_info(key):
    """Extract information from a CSV file.

    Args:
        key (str): Key to extract.

    Returns:
        str: Path to the folder.
    """
    open_boneageassessment()
    filename = os.path.join(os.getcwd(), 'baa', 'info.csv')
    with open(filename, newline='\n') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            for element in row:
                if key == element:
                    return row[1]

def dataset_process():
    downloads_dir = get_downloads()
    if 'dataset.zip' in os.listdir(downloads_dir):
        if os.name != 'posix':
            error = "you work on Windows. The shell command 'unzip' works only for MACOS and Linux.\n"
            error += "Please unzip by your UtilityCompressor and retry."
            raise NotImplementedError(error)

        os.system(f"unzip {os.path.join(downloads_dir, 'dataset.zip')}")
        if os.path.exists('__MACOSX'):
            os.system(f"rm -r '__MACOSX')")
        os.remove(os.path.join(downloads_dir, 'dataset.zip'))

    elif 'dataset' in os.listdir(downloads_dir):
        houdini(opt='dataset')

    elif os.path.exists(os.path.join(os.getcwd(), 'dataset')):
        pass
        
    elif 'dataset_lite.zip' in os.listdir(downloads_dir):
        # lite version of dataset is about 3.5 Gbyte
        if os.name != 'posix':
            error = "you work on Windows. The shell command 'unzip' works only for MACOS and Linux.\n"
            error += "Please unzip by your UtilityCompressor and retry."
            raise NotImplementedError(error)

        src = os.path.join(downloads_dir, 'dataset_lite.zip')
        dest = os.path.join(os.getcwd(), 'dataset')
            #shutil.unpack_archive(src, dest, format="zip")
        os.system(f"unzip {src}")
        os.remove(os.path.join(downloads_dir, 'dataset_lite.zip'))

    else:
        # if you decide not to download dataset, a empty folder will be created
        ds = os.path.join(os.getcwd(), 'dataset')
        os.makedirs(ds, exist_ok=True)
        warn("Any version of dataset was found. An empty directory was created.")

def weights_process():
    downloads_dir = get_downloads()
    if 'weights.zip' in os.listdir(downloads_dir):
        if os.name != 'posix':
            error = "you work on Windows. The shell command 'unzip' works only for MACOS and Linux.\n"
            error += "Please unzip by your UtilityCompressor and retry."
            raise NotImplementedError(error)

        os.system(f"unzip {os.path.join(downloads_dir, 'weights.zip')} -d {os.path.join(os.getcwd(), 'baa', 'age')}")
        os.remove(os.path.join(downloads_dir, 'weights.zip'))
        print("Done!")

    elif 'weights' in os.listdir(downloads_dir):
        houdini(opt="weights")

    elif 'weights_essential.zip' in os.listdir(downloads_dir):
        if os.name != 'posix':
            error = "you work on Windows. The shell command 'unzip' works only for MACOS and Linux.\n"
            error += "Please unzip by your UtilityCompressor and retry."
            raise NotImplementedError(error)

        os.system(f"unzip {os.path.join(downloads_dir, 'weights_essential.zip')} -d {os.path.join(os.getcwd(), 'baa', 'age')}")
        os.remove(os.path.join(downloads_dir, 'weights_essential.zip'))
    
    elif os.path.exists(os.path.join(os.getcwd(), 'baa', 'age', 'weights')):
        pass

    else: 
        raise FileNotFoundError("no file named weights.zip or folder named weights was found.")
