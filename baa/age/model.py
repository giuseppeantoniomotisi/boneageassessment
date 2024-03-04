"""
# Bone Age Assessment Model and Utilities
This Python script encompasses the definition of a Bone Age Assessment model and utility
functions for its preparation, training, evaluation, and visualization. Here's a breakdown
of its key components:

Importing Libraries

The script imports various libraries necessary for deep learning model creation,
data processing, and visualization, including os, numpy, pandas, matplotlib, and keras.

## BoneAgeAssessment Class

The BoneAgeAssessment class provides methods for data preparation, model training, evaluation,
and visualization. Key attributes include paths to data directories, data frames for training,
validation, and test data, image size,batch size, options, and weights directory.
Methods include:
- update_batch_size: Update the batch size for training, validation, and test.
- preparation: Prepare data generators for training, validation, or test.
- r_squared: Calculate R-squared metric.
- compiler: Compile the model with specified learning rate.
- callbacks: Get a list of callbacks for model training.
- loader: Load pre-trained weights into the model.
- fitter: Train the model and return training history.
- get_attention_map_model: Get a sub-model for visualizing attention maps.
- visualize_attention_map: Visualize attention maps for a given image.
- training_evaluation: Evaluate and visualize the training process.
- model_evaluation: Evaluate the model on the test set.

## Model Class

The Model class provides methods for creating different variations of the Bone Age Assessment
model using the VGG16 architecture. Key attributes include the input size of the model and a
flag to show the model summary.
Methods include:
- Model variations such as vgg16regression, vgg16regression_atn, vgg16regression_atn_l1, and
vgg16regression_atn_l2.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.layers import (GlobalAveragePooling2D,
                          Dense,
                          Dropout,
                          Flatten,
                          Input,
                          Conv2D,
                          multiply,
                          LocallyConnected2D,
                          Lambda,
                          BatchNormalization)
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.regularizers import l2,l1
from keras.callbacks import (ModelCheckpoint,
                             LearningRateScheduler,
                             EarlyStopping,
                             ReduceLROnPlateau)
import keras.backend as K

import sys
sys.path.append(sys.path[0].replace('/age',''))
sys.path.append(sys.path[0].replace('age','preprocessing'))
from utils import extract_info
import preprocessing

class BoneAgeAssessment():
    """Class for Bone Age Assessment model.

    This class provides methods for data preparation, model training,
    evaluation, and visualization.

    Attributes:
        main (str): Path to the main directory.
        baa (str): Path to the Bone Age Assessment directory.
        IMAGES (str): Path to the images directory.
        labels (str): Path to the labels directory.
        processed (str): Path to the processed data directory.
        train (str): Path to the training data directory.
        train_df (pd.DataFrame): DataFrame containing training data.
        validation (str): Path to the validation data directory.
        validation_df (pd.DataFrame): DataFrame containing validation data.
        test (str): Path to the test data directory.
        test_df (pd.DataFrame): DataFrame containing test data.
        image_size (tuple): Size of the input images.
        batch_size (tuple): Batch size for training, validation, and test.
        opts (list): List of allowed options.
        weights (str): Path to the directory for storing model weights.

    Methods:
        update_batch_size(new_batch_size: tuple):
            Update the batch size for training, validation, and test.

        preparation(kind: str):
            Prepare data generators for training, validation, or test.

        r_squared(y_true, y_pred):
            Calculate R-squared metric.

        compiler(model, lr: float = 0.0001):
            Compile the model with specified learning rate.

        callbacks() -> list:
            Get a list of callbacks for model training.

        loader(model, weight_name: str) -> keras.model:
            Load pre-trained weights into the model.

        fitter(model, num_epochs: int):
            Train the model and return training history.

        get_attention_map_model(model):
            Get a sub-model for visualizing attention maps.

        visualize_attention_map(img_array, attention_map_model):
            Visualize attention maps for a given image.

        training_evaluation(model, num_epochs: int):
            Evaluate and visualize the training process.

        model_evaluation(weight: str):
            Evaluate the model on the test set.

    """
    def __init__(self):
        """_summary_
        """
        self.main = extract_info('main')
        self.baa = extract_info('baa')
        self.IMAGES = extract_info('IMAGES')
        self.labels = extract_info('labels')
        self.processed = extract_info('processed')
        self.train = extract_info('train')
        self.train_df = pd.read_csv(os.path.join(self.labels, 'train_bal.csv'))
        self.validation = extract_info('validation')
        self.validation_df = pd.read_csv(os.path.join(self.labels, 'validation.csv'))
        self.test = extract_info('test')
        self.test_df = pd.read_csv(os.path.join(self.labels, 'test.csv'))
        self.image_size = (399,399)
        self.batch_size = (32,32,1396) #batch size for training, validation and test
        self.opts = ['train','validation','test']
        self.weights = os.path.join(extract_info('age'),'weights')
    
    def update_batch_size(self,new_batch_size:tuple):
        """Update the batch size for training, validation, and test.

        Args:
            new_batch_size (tuple): New batch size values.
        """
        self.batch_size = new_batch_size
        
    def preparation(self,kind:str):
        """Prepare data generators for training, validation, or test.

        Args:
            kind (str): Type of data preparation.

        Raises:
            KeyError: Raised if the selected key is not allowed.

        Returns:
            _type_: Data generator or test data.
        """
        if kind.isinstance(self.opts):
            raise KeyError(f"the selected key is not allowed. Chooices: {self.opts}")
        if kind == 'train':
            dataframe, directory = self.train_df, self.train
            batch_size = self.batch_size[0]
        elif kind == 'validation':
            dataframe, directory = self.validation_df, self.validation
            batch_size = self.batch_size[1]
        elif kind == 'test':
            dataframe, directory = self.test_df, self.test
            batch_size = self.batch_size[2]

        dataframe_generator = ImageDataGenerator(rescale=1/255.)
        generator = dataframe_generator.flow_from_dataframe(
            dataframe = dataframe,
            directory = directory,
            x_col= 'id',
            y_col= 'boneage',
            batch_size = batch_size,
            seed = 42,
            shuffle = True,
            class_mode= 'other',
            color_mode = 'rgb',
            target_size = self.image_size)
        if kind in ['train','validation']:
            return generator
        else:
            # If we work with test folder, it's better to return test_x and test_y
            return next(generator)

    def r_squared(y_true, y_pred):
        """Calculate R-squared metric.

        Args:
            y_true (tensor): True values.
            y_pred (tensor): Predicted values.

        Returns:
            tensor: R-squared value.
        """
        SS_res = K.sum(K.square(y_true - y_pred)) 
        SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
        return (1 - SS_res / (SS_tot + K.epsilon()))
    
    def compiler(self, model, lr: float = 0.0001):
        """Compile the model with specified optimizer and loss function.

        Args:
        model (keras.Model): The model to be compiled.
        lr (float, optional): Learning rate for the optimizer. Defaults to 0.0001.

        Returns:
        keras.Model: Compiled model.
        """
        optim = Adam(learning_rate=lr)
        return model.compile(optimizer=optim,loss='mse',metrics=['mae', self.r_squared])
    
    def callbacks(self) -> list:
        """Get a list of callbacks for model training.

        Returns:
            list: List of callbacks.
        """
        path = self.weights
        checkpoint = ModelCheckpoint(path,
                                     monitor='val_loss',
                                     verbose=1,
                                     save_best_only=True,
                                     save_weights_only=True,
                                     mode='min')
    
        early = EarlyStopping(monitor="val_loss",
                              mode="min",
                              patience=5)
        return [checkpoint, early]
    
    def loader(self, model, weight_name: str) -> keras.Model:
        """Load pre-trained weights into the model.

        Args:
            model (keras.Model): The model to load weights into.
            weight_name (str): Name of the weight file.

        Returns:
            keras.Model: Model with loaded weights.
        """
        return model.load_weights(os.path.join(self.weights, weight_name))
    
    def fitter(self, model, num_epochs: int):
        """Train the model and return training history.

        Args:
            model (keras.Model): The model to be trained.
            num_epochs (int): Number of epochs for training.

        Returns:
            History: Training history.
        """
        train_generator = self.preparation('training')
        validation_generator = self.preparation('validation')
        history = model.fit(train_generator,
                            steps_per_epoch=len(self.train_df['id']) // self.batch_size[0],
                            batch_size=self.batch_size[0],
                            validation_data=validation_generator,
                            validation_steps=len(self.validation_df['id']) // self.batch_size[1],
                            epochs=num_epochs,
                            callbacks=self.callbacks)
        model.save(os.path.join())
        return history
    
    def get_attention_map_model(self, model):
        """Extract the attention layer from the model.

        Args:
            model (keras.Model): The model.

        Returns:
            keras.Model: Model with the attention layer.
        """
        attn_layer = model.layers[-2]  # Assuming the attention layer is the second last layer
        attention_map_model = Model(inputs=model.input, outputs=attn_layer.output)
        return attention_map_model

    def visualize_attention_map(self, img_array, attention_map_model):
        """Visualize attention maps for a given image.

        Args:
            img_array (np.array): Image array.
            attention_map_model (keras.Model): Model with attention maps.
        """
        img_array = np.expand_dims(img_array, axis=0)
        attention_map = attention_map_model.predict(img_array)
        attention_map = np.squeeze(attention_map, axis=0)

        plt.imshow(attention_map, cmap='jet')
        plt.colorbar()
        plt.show()

    def training_evaluation(self, model, num_epochs: int):
        """Evaluate and visualize the training process.

        Args:
            model (keras.Model): The trained model.
            num_epochs (int): Number of epochs used for training.
        """
        history = self.fitter(model, num_epochs)
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
        axs[0, 0].plot(history['loss'])
        axs[0, 0].plot(history['val_loss'])
        axs[0, 0].set_title('Model loss')
        axs[0, 0].set_ylabel('Loss')
        axs[0, 0].set_xlabel('Epoch')
        axs[0, 0].legend(['train', 'val'], loc='upper left', fontsize=14)

        axs[0, 1].plot(history['mae'])
        axs[0, 1].plot(history['val_mae'])
        axs[0, 1].set_title('Model MAE')
        axs[0, 1].set_ylabel('MAE')
        axs[0, 1].set_xlabel('Epoch')
        axs[0, 1].legend(['train', 'val'], loc='upper left', fontsize=14)

        axs[1, 0].plot(history['r_squared'])
        axs[1, 0].plot(history['val_r_squared'])
        axs[1, 0].set_title('Model $R^2$')
        axs[1, 0].set_ylabel('$R^2$')
        axs[1, 0].set_xlabel('Epoch')
        axs[1, 0].legend(['train', 'val'], loc='upper left', fontsize=14)

        axs[1, 1].axis('off')

        plt.show()

    def model_evaluation(self,weight:str):
        """Evaluate the model on the test set.

        Args:
            weight (str): Name of the pre-trained weight file.
        """
        test_x, test_y = self.preparation('test')
        model = keras.models.load_model(os.path.join(self.weights,weight))
        predictions = (model.predict(test_x, steps = len(test_x))).flatten()

        test_evaluation = pd.DataFrame({'id':self.test_df['id'],
                                'boneage_real':self.test_df['boneage'],
                                'boneage_predict':predictions,
                                'error':(predictions-test_y)})
        test_evaluation.to_csv(os.path.join(self.age,'predicted_age.csv'))

        idx = np.random.randint(0,1396, size=(8,8))
        delta = 50

        images = test_x[delta:delta+64].reshape((8,8,399,399,3))
        real_age = test_y[delta:delta+64].reshape((8,8))
        pred_age = predictions[delta:delta+64].reshape((8,8))

        fig1, axs = plt.subplots(8, 8, figsize=(30,30))
        for i in range(8):
            for j in range(8):
                axs[i][j].imshow(images[i][j], cmap='gray')
                axs[i][j].set_title(f"Real Age (months): {real_age[i][j]}\n" + \
                                    f"Predicted Age (months): {round(pred_age[i][j])}")
                axs[i][j].axis('off')
        plt.show()
        
        fig2, axs = plt.subplots(2,1,figsize=(12,12))
        axs[0].plot(test_y, predictions,'.','Predicted age')
        axs[0].set_title('Prediction results with augmented datatraining')
        axs[0].plot(test_y, test_y, color='tab:red', label='Real age')
        axs[0].set_xlabel('Real age [months]')
        axs[0].set_ylabel('Predicted age [months]')

        h, e, _ = axs[1].hist(test_evaluation['error'], bins=120, range=(-60,60))
        axs[1].set_title('Predicted age error histogram (augmented datatraining)')
        axs[1].set_xlabel('Predicted age error [months]')
        axs[1].set_ylabel('Occurences')

        plt.show()

class Model:
    """Class for Bone Age Assessment model creation.

    This class provides methods for creating different variations of
    the Bone Age Assessment model using VGG16 architecture.

    Attributes:
        input_size (tuple): Input size of the model.
        summ (bool): Flag to show model summary.

    Methods:
        vgg16regression():
            Create a simplified VGG16 regression model.

        vgg16regression_atn():
            Create VGG16 regression model with attention mechanism.

        vgg16regression_atn_l1(reg_factor):
            Create VGG16 regression model with attention mechanism and L1 regularization.

        vgg16regression_atn_l2(reg_factor):
            Create VGG16 regression model with attention mechanism and L2 regularization.

    """
    def __init__(self,input_size:tuple=(399,399,3),summ:bool=True):
        """Initialize the BoneAgeAssessmentModel.

        Args:
            input_size (tuple, optional): The input size of the model. Defaults to (399, 399, 3).
            transfer_learning (bool, optional): Whether to use transfer learning. Default: False.
            summ (bool, optional): Whether to show summary. Defaults to True.

        Raises:
            TypeError: Raised if input_size is not a tuple or transfer_learning is not a bool.
        """
        if not isinstance(input_size,tuple):
            raise TypeError("Input size must be a tuple.")

        self.input_size = input_size
        self.summ = summ

    def vgg16regression(self):
        """Create a simplified VGG16 regression model.

        Returns:
            keras.Model: Simplified VGG16 regression model.
        """
        # Define input layer
        in_layer = Input(self.input_size)
        # Load VGG16 model with pre-trained weights
        base_model = VGG16(weights="imagenet", include_top=False, input_shape=self.input_size)
        base_model.trainable = True

        # Obtain features from VGG16 base
        pt_features = base_model(in_layer)

        # Pooling features
        gap_features = GlobalAveragePooling2D()(pt_features)

        # Adding a dropout layer for some regularization
        gap_dr = Dropout(0.5)(gap_features)

        # Output layer
        out_layer = Dense(1, activation='linear')(gap_dr)

        # Compile the model
        model = Model(inputs=[in_layer], outputs=[out_layer], name='RVGG16')

        # Show summary if summ flag is True
        if self.summ:
            model.summary()

        return model
    
    def vgg16regression_atn(self):
        """Create VGG16 regression model with attention mechanism.

        Returns:
            keras.Model: VGG16 regression model with attention.
        """
        # Define input layer
        in_layer = Input(self.input_size)
        # Load VGG16 model with pre-trained weights
        base_model = VGG16(weights="imagenet",include_top=False,input_shape=self.input_size)
        base_model.trainable = True

        # Obtain features from VGG16 base
        pt_depth = base_model.layers[-1].output_shape[3]
        pt_features = base_model(in_layer)

        # Apply attention mechanism
        bn_features = BatchNormalization()(pt_features)
        attn_layer = Conv2D(64,kernel_size=(1,1),padding='same',activation='relu')(bn_features)
        attn_layer = Conv2D(16,kernel_size=(1,1),padding='same',activation='relu')(attn_layer)
        attn_layer = LocallyConnected2D(1,kernel_size=(1,1),padding='valid',
                                        activation='sigmoid')(attn_layer)
        up_c2_w = np.ones((1,1,1,pt_depth))
        up_c2 = Conv2D(pt_depth,kernel_size=(1,1),padding='same',
                    activation='linear',use_bias=False,weights=[up_c2_w])
        up_c2.trainable = False
        attn_layer = up_c2(attn_layer)
        mask_features = multiply([attn_layer,bn_features])
        gap_features = GlobalAveragePooling2D()(mask_features)
        gap_mask = GlobalAveragePooling2D()(attn_layer)
        gap = Lambda(lambda x: x[0]/x[1],name='RescaleGAP')([gap_features,gap_mask])
        gap_dr = Dropout(0.5)(gap)
        dr_steps = Dropout(0.25)(Dense(1024,activation='elu')(gap_dr))
        out_layer = Dense(1,activation='linear')(dr_steps)

        # Compile the model
        model = Model(inputs=[in_layer],outputs=[out_layer],
                      name='attention_RVGG16')

        # Show summary if summ flag is True
        if self.summ:
            model.summary()

        return model
    
    def vgg16regression_atn_l1(self,reg_factor):
        """Create VGG16 regression model with attention mechanism and L1 regularization.

        Returns:
            keras.Model: VGG16 regression model with attention.
        """
        # Define input layer
        in_layer = Input(self.input_size)
        # Load VGG16 model with pre-trained weights
        base_model = VGG16(weights="imagenet", include_top=False, input_shape=self.input_size)
        base_model.trainable = True

        # Obtain features from VGG16 base
        pt_features = base_model(in_layer)

        # Apply attention mechanism
        bn_features = BatchNormalization()(pt_features)
        attn_layer = Conv2D(64, kernel_size=(1, 1), padding='same', activation='relu')(bn_features)
        attn_layer = Conv2D(16, kernel_size=(1, 1), padding='same', activation='relu')(attn_layer)
        attn_layer = LocallyConnected2D(1, kernel_size=(1, 1), padding='valid', activation='sigmoid')(attn_layer)
        pt_depth = base_model.layers[-1].output_shape[-1]  # Update to use dynamic shape
        up_c2_w = np.ones((1, 1, 1, pt_depth))
        up_c2 = Conv2D(pt_depth, kernel_size=(1, 1), padding='same', activation='linear', use_bias=False, weights=[up_c2_w])
        up_c2.trainable = False
        attn_layer = up_c2(attn_layer)
        mask_features = multiply([attn_layer, bn_features])
        gap_features = GlobalAveragePooling2D()(mask_features)
        gap_mask = GlobalAveragePooling2D()(attn_layer)
        gap = Lambda(lambda x: x[0] / x[1], name='RescaleGAP')([gap_features, gap_mask])
        gap_dr = Dropout(0.5)(gap)
        dr_steps = Dropout(0.25)(Dense(1024, activation='elu', kernel_regularizer=l1(reg_factor))(gap_dr))
        out_layer = Dense(1, activation='linear', kernel_regularizer=l1(reg_factor))(dr_steps)

        # Compile the model
        model = Model(inputs=[in_layer], outputs=[out_layer], name='attention_regularizer_RVGG16')

        # Show summary if summ flag is True
        if self.summ:
            model.summary()

        return model, attn_layer
    
    def vgg16regression_atn_l2(self,reg_factor):
        """Create VGG16 regression model with attention mechanism and L2 regularization.

        Returns:
            keras.Model: VGG16 regression model with attention.
        """
        # Define input layer
        in_layer = Input(self.input_size)
        # Load VGG16 model with pre-trained weights
        base_model = VGG16(weights="imagenet", include_top=False, input_shape=self.input_size)
        base_model.trainable = True

        # Obtain features from VGG16 base
        pt_features = base_model(in_layer)

        # Apply attention mechanism
        bn_features = BatchNormalization()(pt_features)
        attn_layer = Conv2D(64, kernel_size=(1, 1), padding='same', activation='relu')(bn_features)
        attn_layer = Conv2D(16, kernel_size=(1, 1), padding='same', activation='relu')(attn_layer)
        attn_layer = LocallyConnected2D(1, kernel_size=(1, 1), padding='valid', activation='sigmoid')(attn_layer)
        pt_depth = base_model.layers[-1].output_shape[-1]  # Update to use dynamic shape
        up_c2_w = np.ones((1, 1, 1, pt_depth))
        up_c2 = Conv2D(pt_depth, kernel_size=(1, 1), padding='same', activation='linear', use_bias=False, weights=[up_c2_w])
        up_c2.trainable = False
        attn_layer = up_c2(attn_layer)
        mask_features = multiply([attn_layer, bn_features])
        gap_features = GlobalAveragePooling2D()(mask_features)
        gap_mask = GlobalAveragePooling2D()(attn_layer)
        gap = Lambda(lambda x: x[0] / x[1], name='RescaleGAP')([gap_features, gap_mask])
        gap_dr = Dropout(0.5)(gap)
        dr_steps = Dropout(0.25)(Dense(1024, activation='elu', kernel_regularizer=l2(reg_factor))(gap_dr))
        out_layer = Dense(1, activation='linear', kernel_regularizer=l2(reg_factor))(dr_steps)

        # Compile the model
        model = Model(inputs=[in_layer], outputs=[out_layer], name='attention_regularizer_RVGG16')

        # Show summary if summ flag is True
        if self.summ:
            model.summary()

        return model, attn_layer
        

if __name__ == '__main__':
    print(0)
