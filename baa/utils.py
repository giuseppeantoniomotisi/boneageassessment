"""Some useful functions.
"""
import os
import pwd
import platform
import shutil
from csv import reader
from tqdm import tqdm
from RSNA.tools_rsna import unzip_folder

def unzip_sh(path_to_zip):
    if os.name == 'posix':
        os.system(f"unzip {path_to_zip}")

def open_boneageassessment():
    """
    Open boneageassessment directory.
    """
    os.chdir(os.getcwd())

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

def switch_folders(folder1_path, folder2_path):
    # Create a temporary directory to hold the contents of folder1
    temp_dir = os.path.join(os.path.dirname(folder1_path), 'temp')
    os.makedirs(temp_dir, exist_ok=True)

    # Move contents of folder1 to the temporary directory
    for item in tqdm(os.listdir(folder1_path)):
        item_path = os.path.join(folder1_path, item)
        shutil.move(item_path, temp_dir)

    # Move contents of folder2 to folder1
    for item in tqdm(os.listdir(folder2_path)):
        item_path = os.path.join(folder2_path, item)
        shutil.move(item_path, folder1_path)

    # Move contents of temporary directory to folder2
    for item in tqdm(os.listdir(temp_dir)):
        item_path = os.path.join(temp_dir, item)
        shutil.move(item_path, folder2_path)

    # Remove temporary directory
    os.rmdir(temp_dir)

def houdini(opt:str='dataset'):
    """
    Houdini function switch two folders with the same name. This function is used to
    be sure that in your repository you can find everything.
    
    Args:
        opt (str, optional): optional argument to select kind of folder to be switch.
        Only options are: 'dataset' and 'weights'. Otherwise, the function raise a KeyError.
    """
    if opt == 'dataset':
        open_boneageassessment()
        new_loc = os.path.join(os.getcwd(),'dataset')

        open_downloads()
        old_loc = os.path.join(os.getcwd(),'dataset')
    
        shutil.move(old_loc, new_loc)
    elif opt == 'weights':
        open_boneageassessment()
        new_loc = os.path.join(os.getcwd(),'baa','age','weights')

        open_downloads()
        old_loc = os.path.join(os.getcwd(),'weights')

        shutil.move(old_loc, new_loc)
    else:
        raise KeyError("only 'dataset' and 'weights' are supported. Please check input.")

def write_info():
    """
    Write paths of main directories in boneageassessment.
    """
    open_boneageassessment()
    main_baa_dir = os.getcwd()
    baa_dir = os.path.join(main_baa_dir,'baa')

    # age, RSNA, preprocessing modules
    rsna = os.path.join(baa_dir,'RSNA')
    preprocessing = os.path.join(baa_dir,'preprocessing')
    age = os.path.join(baa_dir,'age')
    # app = os.path.join(main_baa_dir,'app')

    # Get paths of images
    IMAGES_dir = os.path.join(main_baa_dir,'dataset','IMAGES')
    labels_dir = os.path.join(IMAGES_dir,'labels')
    raw_dir = os.path.join(IMAGES_dir,'raw')
    processed_dir = os.path.join(IMAGES_dir,'processed')
    train_dir = os.path.join(processed_dir,'train')
    validation_dir = os.path.join(processed_dir,'validation')
    test_dir = os.path.join(processed_dir,'test')
    
    # Write info into a dict
    info = {'main':main_baa_dir,
            'baa':baa_dir,
            'IMAGES':IMAGES_dir,
            'labels':labels_dir,
            'raw':raw_dir,
            'processed':processed_dir,
            'train':train_dir,
            'validation':validation_dir,
            'test':test_dir,
            'rsna':rsna,
            'preprocessing':preprocessing,
            'age':age,
            #'app':app,
            }

    filename = os.path.join(baa_dir, 'info.csv')
    with open(filename, 'w+', newline='\n') as fp:
        fp.write("Dictionary key,Path to folder\n")
        for key in info.keys():
            fp.write(f"{key},{info[key]}\n")

def extract_info(key):
    open_boneageassessment()
    filename = os.path.join(os.getcwd(),'baa','info.csv')
    with open(filename, newline='\n') as file:
        # Create a CSV reader object
        csv_reader = reader(file)
        
        for row in csv_reader:
            for element in row:
                if key == element:
                    return row[1]

if __name__ == '__main__':
    write_info()
