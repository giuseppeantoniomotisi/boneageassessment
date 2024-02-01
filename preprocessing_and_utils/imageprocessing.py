import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


class ImageProcessing:

    def __init__(self):
        pass
        

    @staticmethod
    def zero_padding(image:np.ndarray) -> np.ndarray:
        """
        Pad an image with zeros to ensure it has dimensions of at least 2048x2048.

        Args:
        - image (np.ndarray): The input image to resize

        Returns:
        - np.ndarray: The padded image array.
        """
        if image.shape[0] < 2048 and image.shape[1] < 2048:
            canvas = np.zeros((2048, 2048))
            canvas[:image.shape[0], :image.shape[1]] = image
        else:
            canvas = np.zeros((2048 * 2, 2048 * 2))
            canvas[:image.shape[0], :image.shape[1]] = image
            canvas = cv.resize(canvas, (2048, 2048))
        return canvas

    @staticmethod
    def square(image:np.ndarray) -> np.ndarray:
        """
        Pad an image with zeros to ensure it is square-shaped.

        Args:
            image (np.ndarray): The input image to resize.

        Returns:
            np.ndarray: The squared image array.
        """
        dim = max(image.shape[0], image.shape[1])
        canvas = np.zeros((dim, dim))
        canvas[:image.shape[0], :image.shape[1]] = image
        return canvas
    
    @staticmethod
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
    
    @staticmethod
    def rotation(image:np.ndarray, angle:float) -> np.ndarray:
        """
        Rotate the image by a given angle.

        Args:
            image (numpy.ndarray): The input image.
            angle (float): The angle by which the image should be rotated.

        Returns:
            numpy.ndarray: The rotated image.
        """
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
        return result

    @staticmethod
    def flipping(image:np.ndarray) -> np.ndarray:
        """
        Flip the image vertically.

        Args:
            image (numpy.ndarray): The input image.

        Returns:
            numpy.ndarray: The horizontally flipped image.
        """
        return np.rot90(image, k=2)

    @staticmethod
    def brightness(image:np.ndarray, threshold: float) -> np.ndarray:
        """
        Adjust the brightness of the image.

        Args:
            image (numpy.ndarray): The input image.
            threshold (float): The brightness adjustment threshold.

        Raises:
            ValueError: If the threshold is not in the range (0, 1).

        Returns:
            numpy.ndarray: The image with adjusted brightness.
        """
        # Check if threshold is within valid range
        if not 0 < threshold < 1:
            raise ValueError("Threshold must be a float number in the range (0, 1)")

        # Apply thresholding directly using numpy array operations
        mask = image > threshold * 255
        brighter = image.copy()
        brighter[~mask] = 0

        return brighter

    @staticmethod
    def translation(image:np.ndarray, x_translation:int, y_translation:int) -> np.ndarray:
        """
        Translate the image by given pixel values along x and y axes.

        Args:
            image (numpy.ndarray): The input image.
            x_translation (int): The translation value along the x-axis.
            y_translation (int): The translation value along the y-axis.

        Returns:
            numpy.ndarray: The translated image.
        """
        x, y = x_translation, y_translation
        width, height = image.shape[:2]
        translated_image = np.zeros_like(image)
        translated_image[x:width, y:height] = image[:width-x, :height-y]
        return translated_image
    
    @staticmethod
    def normalization(image: np.ndarray) -> np.ndarray:
        """
        Performs min-max normalization on the input image.

        Args:
            image (np.ndarray): The input image to be normalized.

        Returns:
            np.ndarray: The normalized image.
        """
        normalized = (image - np.min(image)) / (np.max(image) - np.min(image))
        return normalized
    
    @staticmethod
    def transpose(image: np.ndarray) -> np.ndarray:
        """
        Transposes the input image.

        Args:
            image (np.ndarray): The input image to be transposed.

        Returns:
            np.ndarray: The transposed image.

        Raises:
            ValueError: If the input image is not a valid numpy array.
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Input 'image' must be a valid numpy array.")

        return image.transpose()
    
    @staticmethod
    def histogram_equalization(image: np.ndarray) -> np.ndarray:
        """
        Performs histogram equalization on the input image.

        Args:
            image (np.ndarray): The input image to undergo histogram equalization.

        Returns:
            np.ndarray: The image after histogram equalization (placeholder implementation returns the original image).
        """
        equalized_image = cv.equalizeHist(image)
        return equalized_image
    
    @staticmethod
    def clahe(image:np.ndarray) -> np.ndarray:
        clahe=cv.createCLAHE(clipLimit=40)
        equalized_image=cv.equalizeHist(image)
        clahe_image=clahe.apply(equalized_image)
        return clahe_image
    
    @staticmethod
    def show(image: np.ndarray):
        """
        Displays the input image using matplotlib.

        Args:
            image (np.ndarray): The input image to be displayed.
        """
        plt.axis(False)
        plt.imshow(image, cmap='gray')
        plt.colorbar()
        plt.tight_layout()
        plt.show()