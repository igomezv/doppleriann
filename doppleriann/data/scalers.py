"""
----------------------------------------------------------
2025
by Isidro Gomez-Vargas (isidro.gomezvargas@unige.ch)
----------------------------------------------------------

Scaling utilities for DopplerIANN shell data.

Includes Min-Max, Standard, and Masked Standard scalers for
3D datasets (e.g., spectral shells or image-like data cubes).
"""

import numpy as np
import matplotlib.pyplot as plt
from ..utils.logger_config import logger


class MinMaxScaler3D:
    """
    Scales each pixel independently across all shells using min-max normalization.
    """
    def __init__(self, feature_range=(0, 1)):
        self.min = feature_range[0]
        self.max = feature_range[1]
        self.data_min_ = None
        self.data_max_ = None
        self.data_range_ = None

    def fit(self, X):
        """
        Compute min and max per pixel position across all shells.

        Parameters
        ----------
        X : ndarray of shape (N, H, W)
            Input data array.

        Returns
        -------
        self : MinMaxScaler3D
            Fitted scaler instance.
        """
        self.data_min_ = X.min(axis=0, keepdims=True)  # Keep shape (1, H, W)
        self.data_max_ = X.max(axis=0, keepdims=True)
        self.data_range_ = self.data_max_ - self.data_min_
        self.data_range_[self.data_range_ == 0] = 1  # Avoid division by zero
        return self

    def transform(self, X):
        """
        Scale each pixel independently using stored min and max values.

        Parameters
        ----------
        X : ndarray of shape (N, H, W)
            Input data array to scale.

        Returns
        -------
        X_scaled : ndarray
            Scaled data array.
        """
        X_std = (X - self.data_min_) / self.data_range_
        X_scaled = X_std * (self.max - self.min) + self.min
        return X_scaled

    def fit_transform(self, X):
        """Fit and transform the input data in one step."""
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X_scaled):
        """
        Reverse the scaling transformation.

        Parameters
        ----------
        X_scaled : ndarray of shape (N, H, W)
            Scaled data array to revert.

        Returns
        -------
        X_orig : ndarray
            Original (unscaled) data array.
        """
        X_std = (X_scaled - self.min) / (self.max - self.min)
        X_orig = X_std * self.data_range_ + self.data_min_
        return X_orig


class StandardScaler3D:
    """
    Standardizes 3D data by removing the mean and scaling to unit variance.
    Useful for shell datasets and other image-like structures.
    """
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        """
        Compute the mean and standard deviation across all samples.

        Parameters
        ----------
        X : ndarray of shape (N, H, W)
            Input data array.

        Returns
        -------
        self : StandardScaler3D
            Fitted scaler instance.
        """
        self.mean_ = X.mean(axis=(0, 1))  
        self.std_ = X.std(axis=(0, 1))  
        return self

    def transform(self, X):
        """
        Apply standardization to the dataset.

        Parameters
        ----------
        X : ndarray of shape (N, H, W)
            Input data array.

        Returns
        -------
        X_scaled : ndarray
            Standardized data array.
        """
        return (X - self.mean_) / self.std_
    
    def fit_transform(self, X):
        """Fit and transform the input data in one step."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled):
        """
        Reverse the standardization process.

        Parameters
        ----------
        X_scaled : ndarray of shape (N, H, W)
            Standardized data array.

        Returns
        -------
        X_orig : ndarray
            Original data array restored from standardized form.
        """
        return X_scaled * self.std_ + self.mean_
        
    

class MaskedStandardScaler3D:
    """
    Standardizes 3D arrays while preserving zeros (masked normalization).
    Zeros are treated as missing data and excluded from mean/std computation.
    """
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        """
        Compute mean and standard deviation while ignoring zero values.

        Parameters
        ----------
        X : ndarray of shape (N, H, W)
            Input data array with possible zero-masked elements.

        Returns
        -------
        self : MaskedStandardScaler3D
            Fitted scaler instance.
        """

        X_masked = np.where(X == 0, np.nan, X)

        # Compute mean and std, ignoring NaNs.
        self.mean_ = np.nanmean(X_masked, axis=(0, 1, 2))
        self.std_ = np.nanstd(X_masked, axis=(0, 1, 2))
        return self

    def transform(self, X):
        """
        Apply standardization while preserving zero-masked elements.

        Parameters
        ----------
        X : ndarray of shape (N, H, W)
            Input data array with zeros representing masked elements.

        Returns
        -------
        X_norm : ndarray
            Standardized data array with zeros preserved.
        """
        # Mask zeros
        X_masked = np.where(X == 0, np.nan, X)

        # Standardize
        X_norm = (X_masked - self.mean_) / self.std_

        # Replace NaNs back to zeros
        X_norm = np.nan_to_num(X_norm, nan=0.0)

        return X_norm

    def fit_transform(self, X):
        """
        Fit the scaler and apply the transformation in one step.

        Parameters
        ----------
        X : ndarray of shape (N, H, W)
            Input data array.

        Returns
        -------
        X_norm : ndarray
            Standardized data array.
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X_scaled):
        """
        Reverse the standardization, restoring original scale and preserving zeros.

        Parameters
        ----------
        X_scaled : ndarray of shape (N, H, W)
            Standardized data array.

        Returns
        -------
        X_orig : ndarray
            Original data array with zero-mask preserved.
        """
        X_masked = np.where(X_scaled == 0, np.nan, X_scaled)
        X_orig = X_masked * self.std_ + self.mean_
        return np.nan_to_num(X_orig, nan=0.0)