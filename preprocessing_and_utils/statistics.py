"""This code defines a Python class named Statistic which contains several functions
for computing various statistical metrics commonly used in tasks like image segmentation.
Let's break down the code and understand its functionality:

Initialization (__init__ method):
The __init__ method initializes instances of the Statistic class. However, in the provided
code, it's an empty method (pass). No initialization logic is present in this method.

Dice Coefficients (dice_coefficients method):
This method computes the Dice similarity coefficient between the true and predicted values.
It takes y_true and y_pred as input, which represent the true labels and predicted labels, 
respectively. The Dice coefficient is computed using the formula:

(2 * intersection + smooth) / (union + smooth).

It uses the Keras backend functions (K.flatten, K.sum) to compute the intersection and union.

Dice Coefficients Loss (dice_coefficients_loss method):
This method computes the Dice loss, which is the negative of the Dice coefficient.
It simply calls the dice_coefficients method and returns the negative value.

Intersection over Union (IoU) (iou method):
This method calculates the Intersection over Union (IoU) between the true and predicted values.
It computes the intersection and union of y_true and y_pred using the Keras backend functions.
IoU is computed using the formula:

(intersection + smooth) / (summerize - intersection + smooth).

Jaccard Distance (jaccard_distance method):
This method computes the Jaccard distance between the true labels and the predicted labels.
It first flattens the true and predicted labels using K.flatten.
Then, it computes the IoU using the iou method and returns the negative value.
"""
from keras import backend as K

class Statistic:
    """
    This class cointains some useful statistic function.
    """

    def __init__(self) -> None:
        pass

    def dice_coefficients(self, y_true, y_pred, smooth=100):
        """
        Computes the Dice similarity coefficient between the true and predicted values.
        ...
        """
        y_true_flatten = K.flatten(y_true)
        y_pred_flatten = K.flatten(y_pred)

        intersection = K.sum(y_true_flatten * y_pred_flatten)
        union = K.sum(y_true_flatten) + K.sum(y_pred_flatten)
        return (2 * intersection + smooth) / (union + smooth)

    def dice_coefficients_loss(self, y_true, y_pred, smooth=100):
        """
        The Dice loss function for image segmentation models.
        ...
        """
        return -self.dice_coefficients(y_true, y_pred, smooth)

    def iou(self, y_true, y_pred, smooth=100):
        """
        Calculates the Intersection over Union (IoU) between the true and predicted values.
        ...
        """
        intersection = K.sum(y_true * y_pred)
        summerize = K.sum(y_true + y_pred)
        iou = (intersection + smooth) / (summerize - intersection + smooth)
        return iou

    def jaccard_distance(self, y_true, y_pred):
        """
        Function to compute the Jaccard distance between the true labels and the predicted labels.
        ...
        """
        y_true_flatten = K.flatten(y_true)
        y_pred_flatten = K.flatten(y_pred)
        return -self.iou(y_true_flatten, y_pred_flatten)