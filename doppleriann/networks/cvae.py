#!/usr/bin/env python

"""
DopplerIANN Networks Module - Conditional VAEs
----------------------------------------------------------
2025
by Isidro Gomez-Vargas (isidro.gomezvargas@unige.ch)
----------------------------------------------------------
Conditional 1D Convolutional Variational Autoencoder (CVAE)
for shell-based or spectral representations conditioned on
astrophysical or auxiliary parameters.
"""

import numpy as np
from tensorflow import keras as K
from .base_networks import SuperVAE
from .net_blocks import MCDropout, sampling, ReconstructionLossLayer, Resize1DTensorLayer
from ..utils.logger_config import logger


class CondVaeCNN1D(SuperVAE):
    """
    Conditional 1D Convolutional Variational Autoencoder (CVAE).

    This model learns a latent representation of 1D shell-based or spectral
    data while conditioning on auxiliary physical variables (e.g., stellar
    activity indices, RVs, or other metadata).

    Parameters
    ----------
    n_inputs : int
        Number of input features (e.g., number of wavelength bins).
    n_conditions : int
        Number of auxiliary conditioning variables.
    conv_layers : list of tuple, optional
        Convolutional layer configuration, each tuple as (filters, kernel_size).
        Default is [(128, 5), (256, 5)].
    dense_layers : list of int, optional
        Dense layer configuration for encoder and decoder.
        Default is [512, 256].
    **kwargs : dict
        Additional keyword arguments passed to `SuperVAE`.
    """

    def __init__(self, n_inputs, n_conditions, conv_layers=None, dense_layers=None, **kwargs):
        super().__init__(**kwargs)
        self.n_inputs = n_inputs
        self.n_conditions = n_conditions
        self.conv_layers = conv_layers if conv_layers is not None else [(128, 5), (256, 5)]
        self.dense_layers = dense_layers if dense_layers is not None else [512, 256]

    def model_tf(self):
        """Build and return the full Conditional VAE model (encoder, decoder, and CVAE)."""
        logger.info("Building Conditional 1D CNN VAE...")

        # ========== ENCODER ==========
        input_shell = K.Input(shape=(self.n_inputs,), name='shell_input')
        input_cond = K.Input(shape=(self.n_conditions,), name='condition_input')
        x = K.layers.Reshape((self.n_inputs, 1))(input_shell)

        for filters, kernel_size in self.conv_layers:
            x = K.layers.Conv1D(filters, kernel_size, activation=self.actfn, padding='same')(x)
            x = K.layers.AveragePooling1D(pool_size=2)(x)
            x = K.layers.BatchNormalization()(x)
            x = MCDropout(self.dropout)(x)

        shape_before_flattening = K.backend.int_shape(x)[1:]  
        x = K.layers.Flatten()(x)
        x = K.layers.Concatenate()([x, input_cond])

        for units in self.dense_layers:
            x = K.layers.Dense(units, activation=self.actfn)(x)
            x = MCDropout(self.dropout)(x)

        z_mean = K.layers.Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = K.layers.Dense(self.latent_dim, name='z_log_var')(x)
        z = K.layers.Lambda(sampling, name='z')([z_mean, z_log_var])

        encoder = K.Model([input_shell, input_cond], [z_mean, z_log_var, z], name='encoder')
        logger.info(encoder.summary())

        # ========== DECODER ==========
        decoder_input_z = K.Input(shape=(self.latent_dim,), name='z_input')
        decoder_input_c = K.Input(shape=(self.n_conditions,), name='cond_input')
        x = K.layers.Concatenate()([decoder_input_z, decoder_input_c])

        for units in reversed(self.dense_layers):
            x = K.layers.Dense(units, activation=self.actfn)(x)
            x = MCDropout(self.dropout)(x)

        x = K.layers.Dense(np.prod(shape_before_flattening), activation=self.actfn)(x)
        x = K.layers.Reshape(shape_before_flattening)(x)

        for filters, kernel_size in reversed(self.conv_layers):
            x = K.layers.UpSampling1D(size=2)(x)
            x = K.layers.Conv1D(filters, kernel_size, activation=self.actfn, padding='same')(x)
            x = K.layers.BatchNormalization()(x)
            x = MCDropout(self.dropout)(x)

        x = K.layers.Conv1D(1, 3, activation='linear', padding='same')(x)
        x = Resize1DTensorLayer(self.n_inputs)(x)
        decoder_output = K.layers.Reshape((self.n_inputs,), name='decoder_output')(x)

        decoder = K.Model([decoder_input_z, decoder_input_c], decoder_output, name='decoder')
        logger.info(decoder.summary())

        # ========== FULL CVAE ==========
        reconstructed = decoder([z, input_cond])
        cvae_output = ReconstructionLossLayer()([input_shell, reconstructed])
        cvae = K.Model([input_shell, input_cond], cvae_output, name='conditional_vae')

        return cvae, encoder, decoder