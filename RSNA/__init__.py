"""
__init__.py is used to create folders so that the correct paths used by the application can
be constructed. Specifically, it switches from the subdivision proposed by RSNA (training,
validation-1, validation-2), to a hierarchical structure of the type:

../Desktop
-- boneageassessment
    |-- IMAGES
        |-- labels
        |-- processed
            |-- all-images
            |-- test
            |-- train
            |-- val
        |-- raw

You can easily find your new dataset in the desktop!
"""
import os
import pwd
import platform
import zipfile

def open_desktop():
    """
    Open Desktop directory based on the current platform.
    """
    username = pwd.getpwuid(os.getuid()).pw_gecos
    if platform.system() == 'Windows':
        path = 'C:/Users/username/Desktop'.replace('/username/', f'/{username}/')
    elif platform.system() == 'Darwin':
        path = '/Users/username/Desktop'.replace('/username/', f'/{username}/')
    elif platform.system() == 'Linux':
        path = '/home/username/Desktop'.replace('/username/', f'/{username}/')
    os.chdir(path)

def open_downloads():
    """
    Open Downloads directory based on the current platform.
    """
    username = pwd.getpwuid(os.getuid()).pw_gecos
    if platform.system() == 'Windows':
        path = 'C:/Users/username/Downloads'.replace('/username/', f'/{username}/')
    elif platform.system() == 'Darwin':
        path = '/Users/username/Downloads'.replace('/username/', f'/{username}/')
    elif platform.system() == 'Linux':
        path = '/home/username/Downloads'.replace('/username/', f'/{username}/')
    os.chdir(path)

def create_directories():
    """
    Create necessary directories.
    """
    open_desktop()
    current_dir = os.getcwd()

    # Creating directories
    main_baa_dir = os.path.join(current_dir, 'boneageassessment')
    os.makedirs(main_baa_dir, exist_ok=True)

    images_dir = os.path.join(main_baa_dir, 'IMAGES')
    os.makedirs(images_dir, exist_ok=True)

    labels_dir = os.path.join(images_dir, 'labels')
    os.makedirs(labels_dir, exist_ok=True)

    raw_dir = os.path.join(images_dir, 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    
    processed_dir = os.path.join(images_dir, 'processed')
    os.makedirs(processed_dir, exist_ok=True)

    train_dir = os.path.join(processed_dir, 'train')
    os.makedirs(train_dir, exist_ok=True)

    validation_dir = os.path.join(processed_dir, 'validation')
    os.makedirs(validation_dir, exist_ok=True)

    test_dir = os.path.join(processed_dir, 'test')
    os.makedirs(test_dir, exist_ok=True)

def unzip_current_in_folders():
    """
    Unzip all files in the current directory.
    """
    zip_files = list(filter(lambda x: '.zip' in x, os.listdir(os.getcwd())))
    if len(zip_files) == 0:
        pass
    else:
        for zip_file in zip_files:
            name = os.path.join(os.getcwd(), zip_file[:-4])
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(name)
                os.remove(name + '.zip')

def unzip_folder(zip_file):
    """
    Unzip a specified zip file.
    """
    name = zip_file[:-4]
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(name)
        os.remove(name + '.zip')

def check_path():
    """
    Check and update the RSNA paths.
    """
    open_desktop()
    path_file = os.path.join(os.getcwd(), 'boneageassessment', 'original_rsna_paths.txt')
    if os.path.exists(path_file):
        os.remove(path_file)
    with open(path_file, 'a') as input_file:
        input_file.write('# This txt file contains all RSNA paths\n')

    open_downloads()
    elements = list(map(lambda x: os.path.join(os.getcwd(), x), ['train.csv',
                                                                 'Bone Age Validation Set',
                                                                 'boneage-training-dataset']))
    for element in elements:
        if os.path.exists(element):
            if os.path.isdir(element):
                with open(path_file, 'a') as input_file:
                    input_file.write(f'{os.path.join(element)}\n')
                contents = os.listdir(element)
                for content in contents:
                    content = os.path.join(element, content)
                    if '.zip' in content:
                        print(content)
                        unzip_folder(content)
                        with open(path_file, 'a') as input_file:
                            input_file.write(f'{os.path.join(element, content[:-4])}\n')
                    elif '.csv' in content:
                        with open(path_file, 'a') as input_file:
                            input_file.write(f'{os.path.join(element, content)}\n')
                    elif os.path.isdir(content):
                        for subcontent in os.listdir(content):
                            if os.path.isdir(os.path.join(content,subcontent)):
                                with open(path_file, 'a') as input_file:
                                    input_file.write(f'{os.path.join(content,subcontent)}\n')
                            else:
                                pass
                    else:
                        pass
            else:
                with open(path_file, 'a') as input_file:
                    input_file.write(f'{element}\n')
        else:
            raise FileNotFoundError(f"{element} is not in the Download directory!")

def extract_info() -> dict:
    """
    Assigns paths to various datasets from the original RSNA paths file.

    Returns:
    - output_dict (dict): A dictionary containing paths to different datasets.
    """
    # Open the Desktop directory
    open_desktop()
    main = os.path.join(os.getcwd(),'boneageassessment')

    # Get the path of the original RSNA paths file
    input_file = os.path.join(main, 'original_rsna_paths.txt')

    # Initialize the output dictionary with empty lists for different paths
    output_dict = {'train.csv' : [],
                   'train_images_path' : [],
                   'val.csv' : [],
                   'val_images_path' : [],
                   'main_dir' : main,
                   'IMAGES' : os.path.join(main,'IMAGES'),
                   'labels' : os.path.join(main,'IMAGES','labels'),
                   'raw' : os.path.join(main,'IMAGES','raw'),
                   'processed' : os.path.join(main,'IMAGES','processed')}

    # Read lines from the input file and store them in a temporary list
    temp = []
    with open(input_file, 'r') as input:
        temp = input.readlines()

    # Iterate through each line in the temporary list
    for line in temp:
        # Check for keywords in each line and append paths accordingly to the output dictionary
        if 'train.csv' in line:
            output_dict['train.csv'].append(line.replace('\n', ''))
        if 'boneage-training-dataset' in line:
            output_dict['train_images_path'].append(line.replace('\n', ''))
        if 'boneage-validation-dataset' in line:
            output_dict['val_images_path'].append(line.replace('\n', ''))
        if 'Validation Dataset.csv' in line:
            output_dict['val.csv'].append(line.replace('\n', ''))

    return output_dict

if __name__ == '__main__':
    #create_directories()
    #check_path()
    print(extract_info())
