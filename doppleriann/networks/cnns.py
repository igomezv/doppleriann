#!/usr/bin/env python

"""
DopplerIANN Networks Module - CNN Architectures
----------------------------------------------------------
2025
by Isidro Gomez-Vargas (isidro.gomezvargas@unige.ch)
----------------------------------------------------------
1D and 2D Convolutional Neural Networks (CNNs) 
for regression and supervised learning tasks using 
shell-based or spectral representations.
"""

from tensorflow import keras as K
from .base_networks import SupervisedNET
from .net_blocks import MCDropout
from ..utils.logger_config import logger 


class ShellCNN1D(SupervisedNET):
    """
    Flexible 1D Convolutional Neural Network (CNN) for supervised regression tasks.

    This model dynamically constructs 1D convolutional and dense layers for
    processing shell-based or spectral inputs.

    Parameters
    ----------
    input_shape : tuple
        Shape of the 1D input data.
    n_outputs : int
        Number of output values (e.g., RV, FWHM, activity index).
    learning_rate : float, optional
        Learning rate for the optimizer (default: 0.001).
    conv_layers : list of tuple, optional
        List of (filters, kernel_size) for each convolutional layer.
        Default is [(128, 5), (256, 5)].
    dense_layers : list of int, optional
        List specifying neuron counts for fully connected layers.
        Default is [512, 256].
    **kwargs : dict
        Additional arguments for SupervisedNET.
    """

    def __init__(self, input_shape, n_outputs, learning_rate=0.001, conv_layers=None, dense_layers=None, **kwargs):
        super().__init__(deep=dense_layers, **kwargs)
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.n_outputs = n_outputs
        self.conv_layers = conv_layers if conv_layers is not None else [(128, 5), (256, 5)]
        self.dense_layers = dense_layers if dense_layers is not None else [512, 256]

    def model_tf(self):
        """Build and return a 1D CNN model dynamically based on user-defined architecture."""
        logger.info("Building 1D CNN model...")

        input_layer = K.layers.Input(shape=self.input_shape)
        x = input_layer

        for i, (filters, kernel_size) in enumerate(self.conv_layers):
            x = K.layers.Conv1D(filters=filters, kernel_size=kernel_size, activation=self.actfn, kernel_initializer='lecun_normal')(x)

        x = K.layers.Flatten()(x)

        for neurons in self.dense_layers:
            x = K.layers.Dense(neurons, activation=self.actfn, kernel_initializer='lecun_normal')(x)
            x = MCDropout(self.dropout)(x)

        output_layer = K.layers.Dense(self.n_outputs, activation='linear', name='output', kernel_initializer='lecun_normal')(x)
        model = K.models.Model(inputs=input_layer, outputs=output_layer)

        logger.info(model.summary())

        return model


class ShellCNN1DDual(SupervisedNET):
    """
    Dual-input 1D Convolutional Neural Network (CNN).

    This architecture processes two separate 1D shell-based inputs
    through parallel convolutional branches and merges their extracted
    features before regression.

    Parameters
    ----------
    input_shape : tuple
        Shape of each individual 1D input array.
    n_outputs : int
        Number of output regression values.
    learning_rate : float, optional
        Learning rate for the optimizer (default: 0.001).
    conv_layers : list of tuple, optional
        List of (filters, kernel_size) defining Conv1D layers.
        Default is [(128, 5), (256, 5)].
    dense_layers : list of int, optional
        Number of neurons in fully connected layers (default: [512, 256]).
    **kwargs : dict
        Additional keyword arguments for SupervisedNET.
    """
    def __init__(self, input_shape, n_outputs, learning_rate=0.001, conv_layers=None, dense_layers=None, **kwargs):
        super().__init__(deep=dense_layers, **kwargs)
        self.input_shape = input_shape  
        self.learning_rate = learning_rate
        self.n_outputs = n_outputs
        self.conv_layers = conv_layers if conv_layers is not None else [(128, 5), (256, 5)]
        self.dense_layers = dense_layers if dense_layers is not None else [512, 256]

    def model_tf(self):
        """Builds and compiles a 1D CNN model with two input branches."""
        logger.info("Building 1D CNN model with dual inputs...")

        # First input.
        input_1 = K.layers.Input(shape=self.input_shape, name="Input_1")
        x1 = self._conv_branch(input_1)
        # Second input.
        input_2 = K.layers.Input(shape=self.input_shape, name="Input_2")
        x2 = self._conv_branch(input_2)

        # Merge the two processed feature maps
        merged = K.layers.Concatenate()([x1, x2])

        # Fully Connected Dense layers
        x = merged
        for neurons in self.dense_layers:
            x = K.layers.Dense(neurons, activation=self.actfn)(x)
            x = K.layers.Dropout(0.2)(x) 
  
        output = K.layers.Dense(self.n_outputs, activation='linear', name="Output")(x)
        model = K.models.Model(inputs=[input_1, input_2], outputs=output)
        
        logger.info(model.summary())
        return model

    def _conv_branch(self, input_layer):
        """Create a convolutional feature extraction branch for one input."""
        x = input_layer
        for filters, kernel_size in self.conv_layers:
            x = K.layers.Conv1D(filters=filters, kernel_size=kernel_size, activation=self.actfn)(x)
            x = K.layers.BatchNormalization()(x)
        x = K.layers.Flatten()(x)
        return x


class ShellCNN2Channel(SupervisedNET):
    """
    Early-fusion 2D CNN that processes two shell matrices as a single 2-channel input.

    This approach stacks two shell representations (e.g., flux and temperature)
    as separate channels within a single 2D input tensor.

    Parameters
    ----------
    input_shape : tuple
        Input shape including channels (e.g., (9, 9, 2)).
    n_outputs : int
        Number of output regression targets (e.g., RV and DS).
    learning_rate : float, optional
        Optimizer learning rate (default: 0.001).
    conv_layers : list of tuple, optional
        List of (filters, kernel_size) for Conv2D layers.
        Default is [(128, 3), (256, 3)].
    dense_layers : list of int, optional
        Fully connected layer sizes. Default is [512, 256].
    **kwargs : dict
        Additional arguments for SupervisedNET.
    """
    def __init__(self, input_shape, n_outputs, learning_rate=0.001, conv_layers=None, dense_layers=None, **kwargs):
        super().__init__(deep=dense_layers, **kwargs)
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.n_outputs = n_outputs
        self.conv_layers = conv_layers if conv_layers is not None else [(128, 3), (256, 3)]
        self.dense_layers = dense_layers if dense_layers is not None else [512, 256]

    def model_tf(self):
        """Build and return a 2-channel early-fusion Conv2D regression model."""
        logger.info("Building early-fusion 2-channel Conv2D model...")

        inputs = K.layers.Input(shape=self.input_shape, name="ShellInput")  
        x = inputs

        # Convolutional layers
        for filters, kernel_size in self.conv_layers:
            x = K.layers.Conv2D(filters=filters, kernel_size=(kernel_size, kernel_size), padding='same', use_bias=False)(x)
            x = K.layers.BatchNormalization()(x)
            x = K.layers.Activation(self.actfn)(x)

        x = K.layers.Flatten()(x)

        # Dense layers
        for units in self.dense_layers:
            x = K.layers.Dense(units, activation=self.actfn)(x)
            x = K.layers.Dropout(0.4)(x)

        outputs = K.layers.Dense(self.n_outputs, activation='linear', name='Output')(x)
        model = K.models.Model(inputs=inputs, outputs=outputs)

        logger.info(model.summary())
        return model


class ShellCNN2D(SupervisedNET):
    """
    Flexible 2D Convolutional Neural Network (CNN) for regression tasks.

    Parameters
    ----------
    input_shape : tuple
        Shape of the 2D input matrix.
    n_outputs : int
        Number of regression outputs.
    learning_rate : float, optional
        Optimizer learning rate (default: 0.001).
    conv_layers : list of tuple, optional
        List specifying (filters, kernel_size) for Conv2D layers.
        Default is [(128, 5), (256, 5)].
    dense_layers : list of int, optional
        List of neuron counts for dense layers.
        Default is [512, 256].
    **kwargs : dict
        Additional arguments passed to SupervisedNET.
    """

    def __init__(self, input_shape, n_outputs, learning_rate=0.001, conv_layers=None, dense_layers=None, **kwargs):
        super().__init__(**kwargs)
        self.input_shape = input_shape
        self.n_outputs = n_outputs
        self.learning_rate = learning_rate
        self.conv_layers = conv_layers if conv_layers is not None else [(128, 5), (256, 5)]
        self.dense_layers = dense_layers if dense_layers is not None else [512, 256]

    def model_tf(self):
        """Build and return a flexible 2D CNN regression model."""
        logger.info("Building 2D CNN model...")

        model = K.models.Sequential()

        for i, (filters, kernel_size) in enumerate(self.conv_layers):
            if i == 0:  
                model.add(K.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding='same', use_bias=False, input_shape=self.input_shape))
            else:
                model.add(K.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding='same', use_bias=False))
            model.add(K.layers.BatchNormalization())
            model.add(K.layers.Activation(self.actfn))

        model.add(K.layers.Flatten())  

        for neurons in self.dense_layers:
            model.add(K.layers.Dense(neurons, activation=self.actfn))
            model.add(K.layers.Dropout(0.2)) 

        model.add(K.layers.Dense(self.n_outputs, activation='linear'))

        logger.info(model.summary())
        return model
