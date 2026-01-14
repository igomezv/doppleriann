#!/usr/bin/env python

"""
DopplerIANN Networks Module - Custom TensorFlow Layers
-------------------------------------------------------
2025
by Isidro Gomez-Vargas (isidro.gomezvargas@unige.ch)
-------------------------------------------------------
Collection of custom TensorFlow/Keras layers and helper functions
used in DopplerIANN for neural network regularization, uncertainty
quantification (Monte Carlo dropout), and variational autoencoders.
"""

import tensorflow as tf
from tensorflow import keras as K
from ..utils.logger_config import logger  


class MCDropout(K.layers.Layer):
    """
    Monte Carlo Dropout layer.

    Enables dropout to remain active during inference, allowing stochastic
    forward passes for uncertainty estimation and Bayesian approximation.

    Parameters
    ----------
    rate : float
        Dropout rate (probability of dropping a unit).
    is_disabled : bool, optional
        If True, disables dropout completely (default is False).
    noise_shape : tuple, optional
        Shape of the dropout noise mask (default is None).
    name : str, optional
        Name of the layer.
    **kwargs : dict
        Additional arguments passed to the Keras Layer base class.
    """

    def __init__(
            self, rate: float, is_disabled: bool = False,
            noise_shape: tuple = None, name: str = None, **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.rate = rate
        self.is_disabled = is_disabled
        self.noise_shape = noise_shape
        logger.info(f"MCDropout initialized with rate={self.rate}, is_disabled={self.is_disabled}")

    def call(self, inputs: tf.Tensor, training: bool = None) -> tf.Tensor:
        """
        Apply dropout during both training and inference (if not disabled).

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor to apply dropout on.
        training : bool, optional
            Whether the layer is currently in training mode.

        Returns
        -------
        tf.Tensor
            Tensor after dropout or unmodified input if disabled.
        """
        if self.is_disabled:
            return inputs
        return tf.nn.dropout(inputs, rate=self.rate, noise_shape=self.noise_shape)

    def get_config(self) -> dict:
        """Returns the configuration of the layer for serialization."""
        config = super().get_config()
        config.update({
            'rate': self.rate,
            'is_disabled': self.is_disabled,
            'noise_shape': self.noise_shape,
        })
        return config
