#!/usr/bin/env python

"""
DopplerIANN Networks Module - Kolmogorov-Arnold Networks (KAN)
----------------------------------------------------------
2025
by Isidro Gomez-Vargas (isidro.gomezvargas@unige.ch)
----------------------------------------------------------
Kolmogorov-Arnold Network (KAN) implementation for shell-based
or spectral inputs, using spline-like nonlinearities for flexible,
interpretable regression modeling.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras as K
# For spline-like activation functions
from scipy.interpolate import UnivariateSpline  
from .base_networks import SupervisedNET
from .net_blocks import MCDropout
from ..utils.logger_config import logger  



class KAN(SupervisedNET):
    """
    Kolmogorov–Arnold Network (KAN) Spectrum Feedforward Neural Network.

    A feedforward neural network that approximates continuous mappings
    using univariate spline-based activation functions, inspired by
    the Kolmogorov–Arnold representation theorem.

    Parameters
    ----------
    n_inputs : int
        Number of input features (e.g., number of wavelength bins).
    n_outputs : int
        Number of outputs (e.g., target regression values).
    deep : list of int, optional
        Number of neurons in each hidden layer. Default is [50, 50].
    actfn : str, optional
        Standard activation function for dense layers (default: 'tanh').
    dropout : float, optional
        Dropout rate between layers. Default is 0.2.
    mcdropout : bool, optional
        Whether to use Monte Carlo Dropout for Bayesian sampling. Default is True.
    **kwargs : dict
        Additional arguments passed to `SupervisedNET`.
    """

    def __init__(self, n_inputs, n_outputs, deep=None, actfn='tanh', dropout=0.2, mcdropout=True, **kwargs):
        super().__init__(dropout=dropout, mcdropout=mcdropout, **kwargs)
        self.deep = deep if deep is not None else [50, 50]  #   Avoids mutable default
        self.actfn = actfn  #   Activation function
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

    def univariate_activation(self, x):
        """
        Custom activation function using cubic UnivariateSpline interpolation.

        Each layer applies a smooth, spline-based nonlinearity over the
        neuron outputs, enabling flexible functional approximations
        with interpretable mappings.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor to apply the spline-based activation.

        Returns
        -------
        tf.Tensor
            Tensor with spline-based nonlinear transformation applied.
        """

        def spline_func(x_np):
            """Internal cubic spline applied to 1D input samples."""
            x_np = x_np.flatten()  # Ensure input is 1D
            x_vals = np.linspace(0, 1, len(x_np))  # Ensure equal length for interpolation

            spline = UnivariateSpline(x_vals, x_np, k=3, s=0)  # Cubic spline
            result = spline(x_vals)

            return np.clip(result, -10, 10).astype(np.float32)  # Prevent extreme values

        # Apply the function and ensure correct shape
        output = tf.numpy_function(func=spline_func, inp=[x], Tout=tf.float32)
        output = tf.reshape(output, tf.shape(x))  # Preserve batch shape

        return output

    def model_tf(self):
        """Builds and compiles the SpecKAN model."""
        logger.info("Building Kolmogorov-Arnold Network (KAN)...")

        # Input layer
        inputs = tf.keras.Input(shape=(self.n_inputs,), name='kan_input')

        # First layer with univariate activation functions
        h = inputs
        for neurons in self.deep:
            h = K.layers.Dense(neurons)(h)
            h = K.layers.Lambda(self.univariate_activation)(h)  # Apply univariate function on connections
            h = MCDropout(self.dropout)(h)

        # Output layer (regression output in this case)
        outputs = K.layers.Dense(1, activation='linear')(h)

        # Define the model
        kan_ffnn = K.Model(inputs, outputs, name='kan_ffnn')
        logger.info(kan_ffnn.summary())

        # Compile the model with an appropriate regression loss function
        # kan_ffnn.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        return kan_ffnn