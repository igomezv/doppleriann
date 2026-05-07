#!/usr/bin/env python

"""
DopplerIANN Networks Module - Autoencoders
----------------------------------------------------------
2025
by Isidro Gomez-Vargas (isidro.gomezvargas@unige.ch)
----------------------------------------------------------
Variable 1D Convolutional Variational Autoencoder (VAE) and
Autoencoder architectures for shell-based or spectral inputs.
"""

import numpy as np
from tensorflow import keras as K
from .net_blocks import MCDropout, KLDivergenceLayer, ReconstructionLossLayer, Resize1DTensorLayer, L1LatentRegularization, sampling
from .base_networks import SuperVAE, SupervisedNET
from ..utils.logger_config import logger 


class VaeCNN1D(SuperVAE):
    """
    Flexible 1D Convolutional Variational Autoencoder (VAE).

    This model allows dynamic configuration of convolutional and
    dense layers for 1D shell-based or spectral representations.

    Parameters
    ----------
    n_inputs : int
        Length of the 1D input vector (e.g., number of spectral bins).
    conv_layers : list of tuple, optional
        List specifying (filters, kernel_size) for convolutional layers.
        Default is [(128, 5), (256, 5)].
    dense_layers : list of int, optional
        List specifying fully connected layer sizes.
        Default is [512, 256].
    **kwargs : dict
        Additional arguments for the parent SuperVAE class.

    Returns
    -------
    tuple
        (vae, encoder, decoder) TensorFlow Keras models.
    """

    def __init__(self, n_inputs, conv_layers=None, dense_layers=None, **kwargs):
        super().__init__(**kwargs)
        self.n_inputs = n_inputs
        self.conv_layers = conv_layers if conv_layers is not None else [(128, 5), (256, 5)]
        self.dense_layers = dense_layers if dense_layers is not None else [512, 256]

    def model_tf(self):
        """Builds the flexible 1D CNN VAE model dynamically based on user-defined layers."""
        logger.info("Building flexible 1D CNN VAE...")

        # ===== ENCODER =====
        inputs = K.Input(shape=(self.n_inputs,))
        x = K.layers.Reshape((self.n_inputs, 1))(inputs)

        for filters, kernel_size in self.conv_layers:
            x = K.layers.Conv1D(filters=filters, kernel_size=kernel_size, padding='same', activation=self.actfn)(x)
            x = K.layers.AveragePooling1D(pool_size=5)(x)
            x = K.layers.BatchNormalization()(x)
            x = MCDropout(self.dropout)(x)

        shape_after_pooling = K.backend.int_shape(x)[1:]  
        x = K.layers.Flatten()(x)

        for neurons in self.dense_layers:
            x = K.layers.Dense(neurons, activation=self.actfn)(x)
            x = MCDropout(self.dropout)(x)

        z_mean = K.layers.Dense(self.latent_dim)(x)
        z_log_sigma = K.layers.Dense(self.latent_dim)(x)
        
        # Add KL divergence and L1 regularization
        z_mean = KLDivergenceLayer(beta=1.0)([z_mean, z_log_sigma])
        z = K.layers.Lambda(sampling)([z_mean, z_log_sigma])
        z = L1LatentRegularization(l1_lambda=1e-3)(z)  

        encoder = K.Model(inputs, [z_mean, z_log_sigma, z], name='encoder')
        logger.info(encoder.summary())

        # ===== DECODER =====
        latent_inputs = K.Input(shape=(self.latent_dim,), name='z_sampling')
        x = latent_inputs

        for neurons in reversed(self.dense_layers):
            x = K.layers.Dense(neurons, activation=self.actfn)(x)
            x = MCDropout(self.dropout)(x)

        x = K.layers.Dense(np.prod(shape_after_pooling), activation=self.actfn)(x)
        x = K.layers.Reshape(shape_after_pooling)(x)

        # Reverse convolutional layers
        for filters, kernel_size in reversed(self.conv_layers):
            x = K.layers.UpSampling1D(size=5)(x)  # Mirror of AveragePooling1D(pool_size=5)
            x = K.layers.Conv1D(filters=filters, kernel_size=kernel_size, activation=self.actfn, padding='same')(x)
            x = K.layers.BatchNormalization()(x)
            x = MCDropout(self.dropout)(x)

        x = K.layers.Conv1D(1, 3, activation=self.actfn, padding="same")(x)
        x = MCDropout(self.dropout)(x)
    
        target_len = self.n_inputs  
        x = Resize1DTensorLayer(target_len)(x)
        decoder_outputs = K.layers.Reshape((self.n_inputs,), name="decoder_output")(x)

        decoder = K.Model(latent_inputs, decoder_outputs, name='decoder')
        logger.info(decoder.summary())

        # ===== FINAL VAE MODEL =====
        outputs = decoder(encoder(inputs)[0])
        final_outputs = ReconstructionLossLayer()([inputs, outputs])
        vae = K.Model(inputs, final_outputs, name='vae')

        return vae, encoder, decoder


class AeCNN1D(SupervisedNET):
    """
    Flexible 1D Convolutional Autoencoder.

    This model provides a convolutional autoencoder architecture with
    L1 regularization for sparse latent representations.

    Parameters
    ----------
    n_inputs : int
        Length of the flattened input vector.
    conv_layers : list of tuple, optional
        List of (filters, kernel_size) for convolutional layers.
        Default is [(128, 5), (256, 5)].
    dense_layers : list of int, optional
        Sizes of fully connected layers.
        Default is [512, 256].
    **kwargs : dict
        Additional parameters passed to SupervisedNET.

    Returns
    -------
    tuple
        (autoencoder, encoder, decoder) TensorFlow Keras models.
    """
    def __init__(self, n_inputs, conv_layers=None, dense_layers=None, **kwargs):
        super().__init__(**kwargs)
        self.n_inputs = n_inputs
        self.conv_layers = conv_layers if conv_layers is not None else [(128, 5), (256, 5)]
        self.dense_layers = dense_layers if dense_layers is not None else [512, 256]

    def model_tf(self):
        """Build the flexible 1D CNN Autoencoder architecture."""
        logger.info("Building flexible 1D CNN Autoencoder...")

        # ===== ENCODER =====
        inputs = K.Input(shape=(self.n_inputs,), name='ae_input')
        x = K.layers.Reshape((self.n_inputs, 1))(inputs)

        for filters, kernel_size in self.conv_layers:
            x = K.layers.Conv1D(filters=filters, kernel_size=kernel_size, padding='same', activation=self.actfn)(x)
            x = K.layers.AveragePooling1D(pool_size=5)(x)
            x = K.layers.BatchNormalization()(x)
            x = MCDropout(self.dropout)(x)

        shape_after_pooling = K.backend.int_shape(x)[1:]
        x = K.layers.Flatten()(x)

        for neurons in self.dense_layers:
            x = K.layers.Dense(neurons, activation=self.actfn)(x)
            x = MCDropout(self.dropout)(x)

        latent = K.layers.Dense(
            self.dense_layers[-1],
            activation=self.actfn,
            activity_regularizer=K.regularizers.l1(1e-5),
            name='latent_code',
        )(x)

        encoder = K.Model(inputs, latent, name='encoder')
        logger.info(encoder.summary())

        # ===== DECODER =====
        latent_inputs = K.Input(shape=(self.dense_layers[-1],), name='decoder_input')
        x = latent_inputs

        for neurons in reversed(self.dense_layers[:-1]):
            x = K.layers.Dense(neurons, activation=self.actfn)(x)
            x = MCDropout(self.dropout)(x)

        x = K.layers.Dense(np.prod(shape_after_pooling), activation=self.actfn)(x)
        x = K.layers.Reshape(shape_after_pooling)(x)

        for filters, kernel_size in reversed(self.conv_layers):
            x = K.layers.UpSampling1D(size=5)(x)
            x = K.layers.Conv1D(filters=filters, kernel_size=kernel_size, activation=self.actfn, padding='same')(x)
            x = K.layers.BatchNormalization()(x)
            x = MCDropout(self.dropout)(x)

        x = K.layers.Conv1D(1, 3, activation=self.actfn, padding="same")(x)
        x = MCDropout(self.dropout)(x)
        x = Resize1DTensorLayer(self.n_inputs)(x)
        decoder_outputs = K.layers.Reshape((self.n_inputs,), name="decoder_output")(x)

        decoder = K.Model(latent_inputs, decoder_outputs, name='decoder')
        logger.info(decoder.summary())

        # ===== AUTOENCODER =====
        autoencoder_output = decoder(encoder(inputs))
        autoencoder = K.Model(inputs, autoencoder_output, name='autoencoder')

        return autoencoder, encoder, decoder