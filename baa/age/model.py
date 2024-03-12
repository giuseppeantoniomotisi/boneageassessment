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
sys.path.append(sys.path[0].replace('/age','/preprocessing'))
from utils import extract_info
from preprocessing import tools

def mean_absolute_error(y_true,y_pred):
    error = y_true - y_pred
    return np.sum(np.abs(error)) / len(error)

def mean_absolute_deviation(y_true,y_pred):
    error = y_true - y_pred
    return np.sum(np.abs(error - np.mean(error))) / len(error)

def lr_scheduler(epoch, initial_lr=1e-04, decay_rate=0.95):
    """
    Learning rate schedule function with exponential decay.

    Parameters:
    - epoch: Current epoch number
    - initial_lr: Initial learning rate
    - decay_rate: Rate of decay

    Returns:
    - lr: Updated learning rate
    """
    lr = initial_lr * np.power(decay_rate, epoch)
    return lr

@keras.saving.register_keras_serializable()
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
class BoneAgeAssessment():
    """Class for Bone Age Assessment model.

    This class provides methods for data preparation, model training,
    evaluation, and visualization.

    Attributes:
        main (str): Path to the boneageassessment directory.
        baa (str): Path to the baa directory.
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
    def __init__(self, image_size: tuple = (399, 399), batch_size: tuple = (32, 32, 1396), epochs: int = 20, lr: float = 1e-05, balanced: bool = True):
        """Initialize the class with necessary parameters.

        Args:
            image_size (tuple, optional): Size of the images. Defaults to (399,399).
            batch_size (tuple, optional): Batch size for training, validation, and test sets. Defaults to (32, 32, 1396).
            epochs (int, optional): Number of epochs for training. Defaults to 20.
            lr (float, optional): Learning rate. Defaults to 1e-05.
            balanced (bool, optional): Flag indicating whether to use balanced datasets. Defaults to True.
        """
        # Path to directories
        self.main = extract_info('main')
        self.baa = extract_info('baa')
        self.IMAGES = extract_info('IMAGES')
        self.labels = extract_info('labels')
        self.processed = extract_info('processed')
        self.age = extract_info('age')
        self.weights = os.path.join(extract_info('age'), 'weights')
        self.results = os.path.join(extract_info('age'), 'weights')

        # Training, validation and test folders and labels
        self.train = extract_info('train')
        if balanced:
            self.train_df = pd.read_csv(os.path.join(self.labels, 'train_bal.csv'))
        else:
            self.train_df = pd.read_csv(os.path.join(self.labels, 'train.csv'))
        self.validation = extract_info('validation')
        self.validation_df = pd.read_csv(os.path.join(self.labels, 'validation.csv'))
        self.test = extract_info('test')
        self.test_df = pd.read_csv(os.path.join(self.labels, 'test.csv'))

        # Instance BoneAgeAssessment() variables
        self.image_size = image_size
        self.batch_size = batch_size  # batch size for training, validation, and test
        self.lr = lr
        self.EPOCHS = epochs

        # Utils
        self.opts = ['train', 'validation', 'test']  # List of options for data sets
    
    def __update_batch_size__(self,new_batch_size:int,key:str) -> None:
        """Update the batch size for training, validation, and test.

        Args:
            new_batch_size (int ot tuple): New batch size value if key is in
            ['train','validation','test]. New batch size values if key is 'all'.
        """
        new_opts = ['train','validation','test','all']
        if key not in new_opts:
            raise KeyError(f"the selected key is not allowed. Chooices: {new_opts}")
        if key == 'train':
            self.batch_size[0] = new_batch_size
        if key == 'validation':
            self.batch_size[1] = new_batch_size
        if key == 'test':
            self.batch_size[2] = new_batch_size
        elif key == 'all':
            if type(new_batch_size) != type((0,0)):
                raise TypeError('if you select the option all, your input is a tuple')
            else:
                self.batch_size = new_batch_size
        
    def __update_lr__(self,new_lr:float) -> None:
        self.lr = new_lr
    
    def __update_epochs__(self,new_num_epochs:int) -> None:
        self.EPOCHS = new_num_epochs
    
    def __show_info__(self) -> dict:
        return {'image size':self.image_size,
                'batch size':self.batch_size,
                'learning rate':self.lr,
                'number of epochs':self.EPOCHS,
                'weights loc':self.weights}
    
    def __get_dataframe__(self,kind:str) -> pd.DataFrame:
        """Get a dataframe selecting a key.

        Args:
            kind (str): Type of data preparation.

        Raises:
            KeyError: Raised if the selected key is not allowed.

        Returns:
            pandas.DataFrame: Traininig or validation or test dataframe  
        """
        if kind not in self.opts:
            raise KeyError(f"the selected key is not allowed. Chooices: {self.opts}")
        else:
            if kind == 'train':
                return self.train_df
            elif kind == 'validation':
                return self.validation_df
            elif kind == 'test':
                return self.test_df
                
    def __get_generator__(self,kind:str):
        """Get a generator selecting a key. If key is 'test', the function
        returns images and labels.

        Args:
            kind (str): Type of data preparation.

        Raises:
            KeyError: Raised if the selected key is not allowed.

        Returns:
            keras.generator: A DataFrameIterator yielding tuples of (x, y) where x is a
            numpy array containing a batch of images with shape (batch_size, *target_size,
            channels) and y is a numpy array of corresponding labels. If key is 'test', it
            returns test_x and test_y.
        """
        if kind not in self.opts:
            raise KeyError(f"the selected key is not allowed. Chooices: {self.opts}")
        if kind == 'train':
            dataframe, directory = self.train_df, self.train
            batch_size = self.batch_size[0]
            sort = True
        elif kind == 'validation':
            dataframe, directory = self.validation_df, self.validation
            batch_size = self.batch_size[1]
            sort = True
        elif kind == 'test':
            dataframe, directory = self.test_df, self.test
            batch_size = self.batch_size[2]
            sort = False

        dataframe_generator = ImageDataGenerator(rescale=1/255.)
        generator = dataframe_generator.flow_from_dataframe(
            dataframe = dataframe,
            directory = directory,
            x_col= 'id',
            y_col= 'boneage',
            batch_size = batch_size,
            seed = 42,
            shuffle = sort,
            class_mode= 'other',
            color_mode = 'rgb',
            target_size = self.image_size)
        if kind in ['train','validation']:
            return generator
        else:
            # If we work with test folder, it's better to return test_x and test_y
            return next(generator)

    def __change_training__(self,balanced:bool=True):
        if balanced:
            self.train_df = pd.read_csv(os.path.join(self.labels, 'train_bal.csv'))
        else:
            self.train_df = pd.read_csv(os.path.join(self.labels, 'train.csv'))

    def preparatory(self):
        """Prepare data generators for training and validation.
        """
        self.train_generator = self.__get_generator__('train')
        self.validation_generator = self.__get_generator__('validation')

    def compiler(self,model):
        """Compile the model with specified optimizer and loss function.

        Args:
        model (keras.Model): The model to be compiled.
        lr (float, optional): Learning rate for the optimizer. Defaults to 0.0001.

        Returns:
        keras.Model: Compiled model.
        """
        optim = Adam(learning_rate=self.lr)
        return model.compile(optimizer=optim,loss='mse',metrics=['mae', r_squared],run_eagerly=True)

    def loader(self,model,weight_name:str) -> keras.Model:
        """Load pre-trained weights into the model.

        Args:
            model (keras.Model): The model to load weights into.
            weight_name (str): Name of the weight file.

        Returns:
            keras.Model: Model with loaded weights.
        """
        return model.load_weights(os.path.join(self.weights,weight_name))

    def fitter(self, model):
        """Train the model and return training history.

        Args:
            model (keras.Model): The model to be trained.
            num_epochs (int): Number of epochs for training.

        Returns:
            History: Training history.
        """
        path = os.path.join(self.weights,'model.keras')
        checkpoint = ModelCheckpoint(path,
                                     monitor='val_loss',
                                     verbose=1,
                                     save_best_only=True,
                                     save_weights_only=True,
                                     mode='min')

        early = EarlyStopping(monitor="val_loss",
                              mode="min",
                              patience=5)

        self.preparatory()
        histo = model.fit(self.train_generator,
                            steps_per_epoch=len(self.train_df['id']) // self.batch_size[0],
                            batch_size=self.batch_size[0],
                            validation_data=self.validation_generator,
                            validation_steps=len(self.validation_df['id']) // self.batch_size[1],
                            epochs=self.EPOCHS,
                            callbacks=[checkpoint,early])
        model.save(os.path.join())
        return histo.history

    def fitter_dummy(self, model, train_generator, validation_generator):
        """Train the model and return training history without callbacks.

        Args:
            model (keras.Model): The model to be trained.
            num_epochs (int): Number of epochs for training.
            train_generator(keras.generator): Keras training dataset generator.
            validation_generator(keras.generator): Keras validation dataset generator.
        Returns:
            History: Training history.
        """
        histo = model.fit(train_generator,
                            steps_per_epoch=len(self.train_df['id']) // self.batch_size[0],
                            batch_size=self.batch_size[0],
                            validation_data=validation_generator,
                            validation_steps=len(self.validation_df['id']) // self.batch_size[1],
                            epochs=self.EPOCHS)
        return histo.history

    def get_attention_map_model(self, model:keras.models):
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

    def training_evaluation(self, model):
        """Evaluate and visualize the training process.

        Args:
            model (keras.Model): The trained model.
            num_epochs (int): Number of epochs used for training.
        """
        path_ckpnt = os.path.join(self.weights,'ckpnt','checkpoint_epoch_{epoch:02d}_model.keras')
        path_best = os.path.join(self.weights,'best_model.keras')
        checkpoint = ModelCheckpoint(path_ckpnt,
                                     monitor='val_loss',
                                     verbose=0,
                                     save_best_only=True,
                                     save_weights_only=True,
                                     mode='min')
        save_best = ModelCheckpoint(path_best,
                                     monitor='val_loss',
                                     verbose=1,
                                     save_best_only=True,
                                     mode='min')
        # differential_lr = ReduceLROnPlateau(monitor="val_loss",
        #                                     factor=0.1,
        #                                     patience=3,
        #                                     verbose=1,
        #                                     mode="auto",
        #                                     min_delta=0.0001,
        #                                     cooldown=0,
        #                                     min_lr=1e-05)
        schedule = LearningRateScheduler(lr_scheduler)
        early = EarlyStopping(monitor="val_loss",
                              mode="min",
                              patience=10)
        
        self.preparatory()
        history_class = model.fit(self.train_generator,
                            steps_per_epoch=len(self.train_df['id']) // self.batch_size[0],
                            batch_size=self.batch_size[0],
                            validation_data=self.validation_generator,
                            validation_steps=len(self.validation_df['id']) // self.batch_size[1],
                            epochs=self.epochs,
                            callbacks=[checkpoint,early,save_best,schedule])
        model.save(os.path.join(self.age,'last_model.keras'))
        self.model = model

        loss, val_loss = history_class.history['loss'], history_class.history['val_loss']
        mae, val_mae = history_class.history['mae'], history_class.history['val_mae']
        r_squared_ = history_class.history['r_squared']
        val_r_squared = history_class.history['val_r_squared']
        epochs = np.arange(0,len(loss),step=1)+1
        filename = os.path.join(self.results,'history.txt')
        with open(filename,'w+') as fp:
            fp.write("# epochs, loss, val_loss, mae, val_mae, r_squared, val_r_squared\n")
            for i in range(len(epochs)):
                fp.write(f"{epochs[i]}, {loss[i]}, {val_loss[i]}, {mae[i]}, {val_mae[i]}, {r_squared_[i]}, {val_r_squared[i]}\n")
        
        fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True,frameon=True,figsize=(10,6))
        axs[0, 0].plot(epochs,loss)
        axs[0, 0].plot(epochs,val_loss)
        axs[0, 0].set_title('Model loss')
        axs[0, 0].set_ylabel('Loss')
        axs[0, 0].set_xlabel('Epoch')
        axs[0, 0].legend(['train', 'val'], loc='upper left')

        axs[0, 1].plot(epochs,mae)
        axs[0, 1].plot(epochs,val_mae)
        axs[0, 1].set_title('Model MAE')
        axs[0, 1].set_ylabel('MAE')
        axs[0, 1].set_xlabel('Epoch')
        axs[0, 1].legend(['train', 'val'], loc='upper left')

        axs[1, 0].plot(epochs,r_squared_)
        axs[1, 0].plot(epochs,val_r_squared)
        axs[1, 0].set_title('Model $R^2$')
        axs[1, 0].set_ylabel('$R^2$')
        axs[1, 0].set_xlabel('Epoch')
        axs[1, 0].legend(['train', 'val'], loc='upper left')

        axs[1, 1].axis('off')
        save_fig = os.path.join(self.results,'training_evaluation.png')
        plt.savefig(fname=save_fig)
        plt.close()

    def model_evaluation(self,training:bool=True,weight:str=None):
        """Evaluate the model on the test set.

        Args:
            training (bool): if True, load the just trained model. Else
                             load the weight specified by the input path,.
            weight (str, optional): Name of the pre-trained weight file.
        """
        test_x, test_y = self.__get_generator__('test')
        if training:
            model = self.model
        else:
            if not weight is None:
                model = keras.models.load_model(os.path.join(self.weights,weight),safe_mode=False)
            else:
                raise ValueError("you must specify the path to the weights!")

        predictions = (model.predict(test_x, steps = len(test_x))).flatten()

        test_evaluation = pd.DataFrame({'id':self.test_df['id'],
                                'boneage_real':self.test_df['boneage'],
                                'boneage_predict':predictions,
                                'error':(predictions-test_y)})
        test_evaluation.to_csv(os.path.join(self.results,'predicted_age.csv'),index=False)

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
        save_fig1 = os.path.join(self.results,'predictions.png')
        plt.savefig(fname=save_fig1)
        plt.close()

        fig2, axs = plt.subplots(2,1,figsize=(12,12))
        axs[0].plot(test_y, predictions,'.',label='Predicted age')
        axs[0].set_title('Predicted age vs real age')
        axs[0].plot(test_y, test_y, color='tab:red', label='Real age')
        axs[0].set_xlabel('Real age [months]')
        axs[0].set_ylabel('Predicted age [months]')
        axs[0].legend(loc="upper left",fontsize=12)

        h, e, _ = axs[1].hist(test_evaluation['error'], bins=120, range=(-60,60))
        axs[1].set_title('Predicted age error histogram')
        axs[1].set_xlabel('Predicted age error [months]')
        axs[1].set_ylabel('Occurences')
        axs[1].legend(loc="upper left",fontsize=12)
        save_fig2 = os.path.join(self.results,'model_results.png')
        plt.savefig(fname=save_fig2)
        plt.close()

        results = pd.DataFrame({'MAE(months)':mean_absolute_error(test_y,predictions),
                               'MAD(months)':mean_absolute_deviation(test_y,predictions),
                               'Smaller abs error(months)':np.min(np.abs(predictions-test_y)),
                               'Max error(months)':np.max(predictions-test_y),
                               'Min error(months)':np.min(predictions-test_y)},dtype=float,index=[0])
        results.to_csv(os.path.join(self.results,'results.csv'),index=False)
    
    def prediction(self, image:np.ndarray, show:bool=True, save:bool=True, image_id:int=0) -> float:
        """Make prediction on an image and visualize the result.

        Args:
            image (np.ndarray): The input image array.
            show (bool, optional): Whether to display the prediction plot. Defaults to True.
            save (bool, optional): Whether to save the prediction plot. Defaults to True.
            image_id (int, optional): Identifier for the image. Defaults to 0.

        Returns:
            float: Predicted age.
        """
        # First load the best weights for the model
        weights = os.path.join(self.weights,'best_model.keras')
        model = keras.models.load_model(weights, safe_mode=False)
        # Now make prediction
        pred = model.predict(image, steps = 1)
        results = pd.read_csv(os.path.join(self.results,'results.csv'))
        error = results['MAD(months)'] # As error we use MAD in months

        fig, ax = plt.subplots(figsize=(10,10))
        z = ax.imshow(image,cmap='gray')
        ax.set_title(f'Predicted age: {pred}$\pm${error}') # Results is shown as title
        ax.axis('off')
        plt.colorbar(z, ax=ax)
        # Show flag, default if True
        if show:
            plt.show()
        # Save flag, default if True
        if save:
            os.makedirs(os.path.join(os.getcwd(),'predictions'), exist_ok=True)
            save_fig = os.path.join(os.getcwd(),'predictions',f'prediction_{image_id}.png')
            plt.savefig(save_fig)
            plt.close()

        return f'Predicted age is {pred}$\pm${error}.'

class BaaModel:
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
            
        vgg16regression_l2(reg_factor):
            Create VGG16 regression model with L2 regularization.

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

    def vgg16regression(self) -> keras.src.engine.functional.Functional:
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

    def vgg16regression_atn(self) -> keras.src.engine.functional.Functional:
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

    def vgg16regression_atn_l1(self,reg_factor:float) -> keras.src.engine.functional.Functional:
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

    def vgg16regression_atn_l2(self,reg_factor:float) -> keras.src.engine.functional.Functional:
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

    def vgg16regression_l2(self, reg_factor:float) -> keras.src.engine.functional.Functional:
        """Create VGG16 regression model with L2 regularization.

        Returns:
            keras.Model: VGG16 regression model.
        """
        # Define input layer
        in_layer = Input(self.input_size)
        # Load VGG16 model with pre-trained weights
        base_model = VGG16(weights="imagenet", include_top=False, input_shape=self.input_size)
        base_model.trainable = True

        # Obtain features from VGG16 base
        pt_features = base_model(in_layer)

        # Apply global average pooling to the features
        gap_features = GlobalAveragePooling2D()(pt_features)
        gap_dr = Dropout(0.5)(gap_features)
        dr_steps = Dropout(0.25)(Dense(1024, activation='elu', kernel_regularizer=l2(reg_factor))(gap_dr))
        out_layer = Dense(1, activation='linear', kernel_regularizer=l2(reg_factor))(dr_steps)

        # Compile the model
        model = Model(inputs=[in_layer], outputs=[out_layer], name='regularized_VGG16')

        # Show summary if summ flag is True
        if self.summ:
            model.summary()

        return model
