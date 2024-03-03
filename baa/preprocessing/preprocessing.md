# Dataset: image preprocessing

After data manipulation and data augmentation, what we need to do is to preprocess our images. This is step has two main purposes: one is to reduce the dataset size and the second is to clean as much as possible the images, making the background black and our subjects (the hands) shiny, i.e. with a higher level of gray. This expedient would create a dataset that would be more readable for our deep learning networks.
Let's start by importing all the packages that we need to perform our preprocessing and then we define the functions that we are going to use throughout this step.

Requirements: os, csv, numpy, matplolib, cv2 and mediapipe

## Install mediapipe
Mediapipe is an open-source framework developed by Google that provides a flexible and easy-to-use pipeline for building machine learning (ML) pipelines for various multimedia processing tasks. It is primarily designed to work with media data such as images, video, and audio.

In Python, the Mediapipe library offers a set of pre-trained models and building blocks for tasks such as:

- Hand tracking: Detecting and tracking hands in images and video.
- Pose estimation: Estimating human poses in images and video.
- Object detection and tracking: Identifying and tracking objects in images and video streams.
- Facial recognition and landmarks: Detecting faces, facial landmarks, and facial expressions.
- Holistic detection: Combining multiple tasks like pose, hand, and face detection together.

```bash
pip install mediapipe
```

In our case, the module `mediapipe` is used to recognize hand, to create bounding box and finally to crop hands.

## Preprocessing class
The `Preprocessing` class cointains all useful methods for our image preprocessing. Here, you can find a quickly overview of its methods:

- `__init`

Initializes the class findings all directories and files which are used in preprocessing pipeline.

- `in_box`

Takes as input the x and y coordinates of a point and checks whether it is inside or outside a rectangle (for example an image). If the point is inside the box, the function returns a tuple containing the x coordinate as the first element and the y coordinate as the second element. Otherwise, if the point is outside the box, it gets redefined on the border of the box and then the function returns the tuple with the corrected coordinates.

- `rectangle`

Takes as input a 2-D array (e.g. an image) and a 2-D list contianing the x and y coordinates of two points, the upper left and the bottom right vertex of a rectangle. Starting from this two points, it moves them by a fraction of the frame shape (frac) and then with the in_box function check if they are inside frame (if they are not, the function changes their coordinates). Returns two tuples, the first containing x and y cooridnates of the top left vertex of an enlarged rectangle and the second one are the x and y coordinates of the bottom right vertex of the same rectangle.

- `cut_peak`

Finds index corresponding to approximate end of background contribution in an X-ray image. It takes as inputs the array histogram of the image, the index to which correpsonds the maximum of the histogram, the peak fraction where to cut the maximum peak and the name of the image file. By descending along the peak to its right side (therefore by assuming that the maximum of the histogram is in the black shade region) the algorithm finds the index (new_index) where the histogram value is less than the maximum of the histogram divided by the peak fraction chosen. If the new_index is further than 10 levels of gray,it is automatically set as index + 5 levels of gray. The code also checks if the index found is too large (larger than the array size) and if it is the case, it puts the new_index = 0 and prints the name of the problematic image.

- `square`

It takes as input an image and outputs the image with same height and width, by choosing as the square length side the greater between height and width of the original image. The image is centered and where there is no original image, the new image is black (gray level = 0).

- `resize`

See documentation of OpenCV cv2.resize function at https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html#ga47a974309e9102f5f08231edc7e7529d.

- `histogram_equalization`

Performs the histogram equalization of some image. It takes as input an image and equalizes its histogram with the method
presented here: https://en.wikipedia.org/wiki/Histogram_equalization.

- `brightness`
Remove the background from the hand X-ray image we are delaing with. By using the cut_peak function finds the index corresponding to the right base of the higher peak of the image histogram (by assuming that this peak is always corresponding to the background pixels). Then every pixel value that is less than the found index or greater than 254 is set to 0.

- `brightness_aug`

This function was only used for augmented images (rotated), in order to remove the white frame and the black background that was originated from the rotation operation. The background gets set at the most occurred gray level excluding pitch black (gray level = 0) and complete white (gray level = 255).

- `preprocessing_image`

This function makes use of brightness (classic and/or augmented version), histogram_equalization square and resize functions to preprocess the image. This is the optimized sequence that gives back an approximately segmented hand X-ray image.

- `preprocessing_directory`

This function makes use of brightness (classic and/or augmented version), histogram_equalization square and resize functions to preprocess all images inside a directory. This is the optimized sequence that gives back an approximately segmented hand X-ray image.
