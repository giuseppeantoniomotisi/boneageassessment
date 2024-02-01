import tensorflow.keras.backend as K

class Statistic:
    
    def __init__(self) -> None:
        pass

    def dice_coefficients(y_true, y_pred, smooth=100):
        """
        Computes the Dice similarity coefficient between the true and predicted values.
        ...
        """
        y_true_flatten = K.flatten(y_true)
        y_pred_flatten = K.flatten(y_pred)

        intersection = K.sum(y_true_flatten * y_pred_flatten)
        union = K.sum(y_true_flatten) + K.sum(y_pred_flatten)
        return (2 * intersection + smooth) / (union + smooth)

    def dice_coefficients_loss(y_true, y_pred, smooth=100):
        """
        The Dice loss function for image segmentation models.
        ...
        """
        return -self.dice_coefficients(y_true, y_pred, smooth)

    def iou(y_true, y_pred, smooth=100):
        """
        Calculates the Intersection over Union (IoU) between the true and predicted values.
        ...
        """
        intersection = K.sum(y_true * y_pred)
        sum = K.sum(y_true + y_pred)
        iou = (intersection + smooth) / (sum - intersection + smooth)
        return iou

    def jaccard_distance(y_true, y_pred):
        """
        Function to compute the Jaccard distance between the true labels and the predicted labels.
        ...
        """
        y_true_flatten = K.flatten(y_true)
        y_pred_flatten = K.flatten(y_pred)
        return -self.iou(y_true_flatten, y_pred_flatten)