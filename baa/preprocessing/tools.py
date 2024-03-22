# Copyright (C) 2024 g.fanciulli2@studenti.unipi.it
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
After data manipulation and data augmentation, what we need to do is to preprocess our images.
This is step has two main purposes: one is to reduce the dataset size and the second is to clean
as much as possible the images, making the background black and our subjects (the hands) shiny,
i.e. with a higher level of gray. This expedient would create a dataset that would be more readable
for our deep learning networks.
Let's start by importing all the packages that we need to perform our preprocessing and then we
define the functions that we are going to use throughout this step.

Requirements: os, csv, numpy, matplolib, cv2 and mediapipe
"""
import os
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import cv2
import sys
sys.path.append(os.path.join(os.getcwd(),'baa'))
import mediapipe as mp
from utils import extract_info

class Preprocessing:
    def __init__(self):
        self.main = extract_info('main')
        self.baa = extract_info('baa')
        self.raw = extract_info('raw')
        self.labels = extract_info('labels')
        os.makedirs(os.path.join(self.raw, 'temp'), exist_ok=True)
        self.temp = os.path.join(self.raw, 'temp')
        self.train = extract_info('train')
        self.validation = extract_info('validation')
        self.test = extract_info('test')

    @staticmethod
    def in_box(vertex_x:int, vertex_y:int, image_width:int, image_height:int) -> tuple:
        """
        Takes as input the x and y coordinates of a point and checks whether it is inside or
        outside a rectangle (for example an image).
        If the point is inside the box, the function returns a tuple containing the x coordinate
        as the first element and the y coordinate as the second element. Otherwise, if the point
        is outside the box, it gets redefined on the border of the box and then the function
        returns the tuple with the corrected coordinates.

        Args:
            - vertex_x (int): x coordinate of point
            - vertex_y (int): y cooridnate of point
            - image_width (int): width of the box (image)
            - image_hieght (int): height of the box (image)

        Returns:
            tuple (int, int): x and y coordinates of the point (may be corrected).
        """
        if vertex_x < 0:
            vertex_x = 0
        elif vertex_x >= image_width:
            vertex_x = image_width - 1
        if vertex_y < 0:
            vertex_y = 0
        elif vertex_y >= image_height:
            vertex_y = image_height - 1

        vertex = (vertex_x, vertex_y)

        return vertex

    def rectangle(self,frame:np.ndarray, coord:list):
        """
        Takes as input a 2-D array (e.g. an image) and a 2-D list contianing the x and y
        coordinates of two points, the upper left and the bottom right vertex of a rectangle.
        Starting from this two points, it moves them by a fraction of the frame shape (frac)
        and then with the in_box function check if they are inside frame (if they are not,
        the function changes their coordinates). Returns two tuples, the first containing x
        and y cooridnates of the top left vertex of an enlarged rectangle and the second one 
        are the x and y coordinates of the bottom right vertex of the same rectangle.

        Args:
            - frame (np.ndarray): 2-D array (image)
            - coord (list): 2-D list containing x and y coordinates of top left and bottom right vertex

        Returns: 
            tuple (int, int), tuple (int, int): coordinates of 'enlarged' vertices of rectangle
        """
        image_height, image_width = frame.shape[0], frame.shape[1]
        frac = 0.15
        elong = image_height*frac
        widen = image_width*frac
        top_left_x = int(min(coord[0])-widen)
        top_left_y = int(max(coord[1])+elong)
        top_left = self.in_box(top_left_x, top_left_y, image_width, image_height)
        bottom_right_x = int(max(coord[0])+widen)
        bottom_right_y = int(min(coord[1])-elong)
        bottom_right = self.in_box(bottom_right_x, bottom_right_y, image_width, image_height)

        return top_left, bottom_right

    @staticmethod
    def cut_peak(h:np.ndarray, index:int, frac:float, filename:str) -> int:
        """
        Finds index corresponding to approximate end of background contribution in an X-ray image.
        It takes as inputs the array histogram of the image, the index to which correpsonds the
        maximum of the histogram, the peak fraction where to cut the maximum peak and the name of
        the image file. By descending along the peak to its right side (therefore by assuming that
        the maximum of the histogram is in the black shade region) the algorithm finds the index
        (new_index) where the histogram value is less than the maximum of the histogram divided
        by the peak fraction chosen. If the new_index is further than 10 levels of gray,it is
        automatically set as index + 5 levels of gray. The code also checks if the index found is
        too large (larger than the array size) and if it is the case, it puts the new_index = 0
        and prints the name of the problematic image.

        Args:
            - h (np.ndarray): histogram array of some image
            - index (int): index (and gray level) corresponding to histogram maximum
            - frac (float): fraction of the histogram maximum where background ends
            - filename (str): name of the image file

        Returns:
            int: the index found, corresponding to the last level of background
        """
        i = 1
        if(index+i >= h.size):
            print(f'{filename} could not be processed!')
            return 0
        while(h[index+i] > h[index]/frac):
            i += 1
            if(index+i >= h.size):
                i = -index
                print(f'{filename} could not be processed!')
                break
        new_index = index + i
        if(new_index > 10 + index):
            new_index = index + 5
        return new_index

    @staticmethod
    def square(image:np.ndarray) -> np.ndarray:
        """
        It takes as input an image and outputs the image with same height and width,
        by choosing as the square length side the greater between height and width
        of the original image. The image is centered and where there is no original
        image, the new image is black (gray level = 0).

        Arg:
            - image (np.ndarray): array of the input image

        Returns:
            np.ndarray: squared image
         """
        im_height = image.shape[0]
        im_width = image.shape[1]
        if(im_height >= im_width):
            dim = im_height
            canvas = np.zeros((dim, dim, 3))
            canvas[:im_height, int(0.5*(dim-im_width)):int(0.5*(dim+im_width)), :] = image
        else:
            dim = im_width
            canvas = np.zeros((dim, dim, 3))
            canvas[int(0.5*(dim-im_height)):int(0.5*(dim+im_height)), :image.shape[1], :] = image
        return canvas

    @staticmethod
    def resize(image:np.ndarray, dimension: tuple) -> np.ndarray:
        """See documentation of OpenCV cv2.resize function
        """
        return cv2.resize(image, dimension)

    @staticmethod
    def histogram_equalization(image:np.ndarray) -> np.ndarray:
        """Performs the histogram equalization of some image.
        It takes as input an image and equalizes its histogram with the method 
        presented here: https://en.wikipedia.org/wiki/Histogram_equalization.

        Args:
            - image (np.ndarray): array of some image

        Returns:
            np.ndarray: array of equalized image
        """
        histim = image.reshape(image.size)
        h,e, = np.histogram(histim, range(0,256))
        cdf = np.cumsum(h)
        cdf_min = np.min(cdf[np.nonzero(cdf)])
        cdf = np.append(cdf, 0.)
        eq_im = np.round((cdf[image.astype(int)] - cdf_min)/(image.size - cdf_min) * 255)

        return eq_im

    def brightness_aug(self,image:np.ndarray, filename:str) -> np.ndarray:
        """This function was only used for augmented images (rotated), in order to
        remove the white frame and the black background that was originated from the
        rotation operation. The background gets set at the most occurred gray level
        excluding pitch black (gray level = 0) and complete white (gray level = 255).

        Args:
            - image (np.ndarray): array of augmented image
            - filename (str): name of the image file

        Returns:
            np.ndarray: array of corrected image
        """

        brighter = image.copy()
        histim = brighter.reshape(brighter.size)
        h, e = np.histogram(histim, range=(1,254), bins=253)
        index = self.cut_peak(h, np.where(h==max(h))[0][0], 10., filename)

        the_mask = image > index
        mask = image < 255
        brighter[~mask] = np.where(h==max(h))[0][0]
        brighter[~the_mask] = np.where(h==max(h))[0][0]

        return brighter

    def brightness(self,image:np.ndarray,filename:str) -> np.ndarray:
        """Remove the background from the hand X-ray image we are delaing with.
        By using the cut_peak function finds the index corresponding to the right
        base of the higher peak of the image histogram (by assuming that this
        peak is always corresponding to the background pixels). Then every pixel
        value that is less than the found index or greater than 254 is set to 0.

        Args:
            - image (np.ndarray): array of some image
            - filename (str): name of the image file

        Returns:
            np.ndarray: array of corrected image
        """
        histim = image.reshape(image.size)
        h,e  = np.histogram(histim, range=(0,256), bins=256)
        index = self.cut_peak(h, np.where(h==max(h))[0][0], 10., filename)

        mask1 = image < 254
        mask2 = image > index

        brighter = image.copy()
        brighter[~mask1] = 0
        brighter[~mask2] = 0

        return brighter
        
    def process(self, image:np.ndarray, filename:str):
        """This function makes use of brightness, histogram_equalization,
        square and resize functions to preprocess the image. This is the optimized
        sequence that gives back an approximately segmented hand X-ray image.

        Args:
        - image (np.ndarray): array of image to preprocess
        - filename (str): name of the image file

        Returns:
        np.ndarray: array of preprocessed image
        """
        image_ = np.copy(image)
        #image_ = brightness_aug(image_, filename)
        image_ = self.brightness(image_, filename)
        image_ = self.histogram_equalization(image_)
        image_ = self.square(image_)
        image_ = self.resize(image_, (399, 399)) 

        return image_


    def preprocessing_image(self, image_path:str, show:bool):
        """Takes the hand x-ray image as input and makes use of mediapipe package
        in order to find the hand in the image. If found, first crops the image with
        a bounding box around the hand and then applies the process method to the
        cropped image; finally it shows/saves it. Alternatively, if it cannot find 
        hands, it just applies the process method to the image and then it shows/saves
        it.

        Args:
            image_name (str): path to the image file
            show (bool): True if user wants the image to be shown, False otherwise

        Raises:
            Warning: if image doesn't get cropped, the program gives a warning.
        """
        # Source path of raw image in boneageassessment/dataset/IMAGES/raw/
        # Destination path of processed image in boneageassessment/dataset/IMAGES/processed/train/
        image_name = os.path.splitext(os.path.basename(image_path))[0] + '.png'
        
        mp_hands = mp.solutions.hands
        hand = mp_hands.Hands()

        coord = [[],[]]

        frame = plt.imread(image_path)
        if type(frame) is not np.ndarray:
            raise TypeError(f'{image_name} was not correctly converted in a numpy array!')
        image_width = frame.shape[1]
        image_height = frame.shape[0]
        results = hand.process(frame)
        hand_landmark = results.multi_hand_landmarks

        # If one hand is detected tha landmark coordinates are saved
        if hand_landmark:
            for landmarks in hand_landmark:
                # Here is How to Get All the Coordinates
                for ids, landmrk in enumerate(landmarks.landmark):
                    coord[0].append(landmrk.x * image_width)
                    coord[1].append(landmrk.y * image_height)
                    # Here we crop and process the image and then save it
            top_left, bottom_right = self.rectangle(frame, coord)
            cropped_frame = frame[bottom_right[1]:top_left[1], top_left[0]:bottom_right[0]]
            processed_frame = self.process(cropped_frame, image_path)
            if show:
                cv2.imshow(processed_frame)

        else:
            print('The image was not cropped since no hand was detected!')
            processed_frame = self.process(frame, image_name)
            if show:
                cv2.imshow(processed_frame)

        return processed_frame
        
    def preprocessing_directory(self):
        """Takes the hand x-ray images contained in some directory as input and makes
        use of mediapipe package in order to find the hand in the images. If found, 
        first crops the images with a bounding box around the hand and then applies the
        process method to the cropped images; finally it saves them. Alternatively, if 
        it cannot find hands, it just applies the process method to the image and then 
        it saves it.
        """
        # Source path of raw image in boneageassessment/dataset/IMAGES/raw/
        loading_path = self.raw
        # Destination path of processed image in boneageassessment/dataset/IMAGES/processed/train/
        saving_path = self.temp
        
        mp_hands = mp.solutions.hands
        hand = mp_hands.Hands()
        hands = 0

        coord = [[],[]]

        # Obtain files in loading_path
        images = os.listdir(loading_path)

        # Filtering only images to prevent errors
        images = list(filter(lambda x: x.endswith('.png'), images))

        for filename in tqdm(images):

            from_path = os.path.join(loading_path, filename)
            to_path = os.path.join(saving_path, filename)
            frame = cv2.imread(from_path)
            if type(frame) is not np.ndarray:
                raise TypeError(f'{filename} was not correctly converted in a numpy array!')
            image_width = frame.shape[1]
            image_height = frame.shape[0]
            results = hand.process(frame)
            hand_landmark = results.multi_hand_landmarks

        
            # If one is detected tha landmark coordinates are saved
            if hand_landmark:
                hands += 1
                for landmarks in hand_landmark:
                    # Here is How to Get All the Coordinates
                    for ids, landmrk in enumerate(landmarks.landmark):
                        coord[0].append(landmrk.x * image_width)
                        coord[1].append(landmrk.y * image_height)
                        # Here we crop and process the image and then save it
                        top_left, bottom_right = self.rectangle(frame, coord)
                        cropped_frame = frame[bottom_right[1]:top_left[1], top_left[0]:bottom_right[0]]
                        processed_frame = self.process(cropped_frame, from_path)
                        cv2.imwrite(to_path, processed_frame)
            else:
                #If no hand gets detected, the image only gets processed
                processed_frame = self.process(frame, filename)
                cv2.imwrite(to_path, processed_frame)
            coord = [[],[]]
            # print(f'- {hands} hands images were correctly found and cropped')
            # print(f'- Total number of images = {len(os.listdir(loading_path))}')
            # print(f'- Cropping{100*hands/len(os.listdir(loading_path))}...')
