import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from PIL import Image

def square(image:np.ndarray) -> np.ndarray:
        """
        Pad an image with zeros to ensure it is square-shaped.

        Args:
            image (np.ndarray): The input image to resize.

        Returns:
            np.ndarray: The squared image array.
        """
        dim = np.max([image.shape[0], image.shape[1]])
        canvas = np.zeros((dim, dim))
        canvas[:image.shape[0], :image.shape[1]] = image
        return canvas

def resize(image:np.ndarray, dimension: tuple) -> np.ndarray:
        """
        Resize an image to the specified dimensions.

        Args:
        - image: The input image to resize.
        - dimension: The target dimensions (height, width).

        Returns:
        - np.ndarray: The resized image.
        """
        return cv.resize(image, dimension)

def preprocessing_image(image):
  image_ = np.copy(image)
  #image_ = ImageProcessing().histogram_equalization(image_)
  image_ = square(image_)
  image_ = resize(image_, (256, 256))
  image_ = image_/255.
  return image_

def process(from_dir, target_dir):
  for filename in os.listdir(from_dir):
    from_path = os.path.join(from_dir, filename)
    to_path = os.path.join(target_dir, filename)
    
    image = cv.imread(from_path)
    image_ = preprocessing_image(image)
    plt.imsave(to_path, image_)
    
if __name__ == '__main__':
  from_train_data_dir = '/Users/Giuseppe/Documents/vscode/exam/dataset/boneage-training-dataset/'
  # from_validation_data_dir = '/Users/Giuseppe/Documents/vscode/exam/dataset/boneage-validation-dataset/boneage-validation-dataset/'
  # from_test_data_dir = '/Users/Giuseppe/Documents/vscode/exam/dataset/boneage-test-dataset/'

  # target_train_data_dir = '/Users/Giuseppe/Documents/vscode/exam/dataset/boneage-training-dataset/resized/'
  # target_validation_data_dir = '/Users/Giuseppe/Documents/vscode/exam/dataset/boneage-validation-dataset/boneage-validation-dataset/resized/'
  # target_test_data_dir = '/Users/Giuseppe/Documents/vscode/exam/dataset/boneage-test-dataset/resized/'

  # process(from_train_data_dir, target_train_data_dir)
  # process(from_validation_data_dir, target_validation_data_dir)
  # process(from_train_data_dir, target_train_data_dir)
  
  image = "1377.png"
  INPUT_FILE = os.path.join(from_train_data_dir, image)
  img = cv.imread(INPUT_FILE, 0)
  image_ = preprocessing_image(img)
  
  image_PIL = Image.fromarray(np.uint8(image_*255), mode = 'L')
  image_PIL.save('/Users/Giuseppe/Desktop/pippo.png')
  
  #zero padding per forma quadrata 
  #resize -> 256x256
  #histogram equalization 
  #normalization -> [0, 1]