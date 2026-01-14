"""
DopplerIANN Neural Architectures
----------------------------------------------------------
2025
by Isidro Gomez-Vargas
----------------------------------------------------------
This package contains all neural network architectures, base classes,
and custom layers used for spectral and shell-based data modeling.
"""

# --- Base Classes ---
from .base_networks import SuperVAE, SupervisedNET

# --- Custom Layers and Tensor Utilities ---
from .net_blocks import (
    MCDropout,
    KLDivergenceLayer,
    ReconstructionLossLayer,
    Resize1DTensorLayer,
    L1LatentRegularization,
    sampling,
)

# --- Model Architectures ---
from .cnns import ShellCNN1D, ShellCNN1DDual, ShellCNN2D, ShellCNN2Channel
from .mlp import MLP

__all__ = [
    # Base Classes
    "SuperVAE",
    "SupervisedNET",

    # Custom Layers and Utilities
    "MCDropout",
    "KLDivergenceLayer",
    "ReconstructionLossLayer",
    "Resize1DTensorLayer",
    "L1LatentRegularization",
    "sampling",

    # Model Architectures

    "ShellCNN1D",
    "ShellCNN1DDual",
    "ShellCNN2D",
    "ShellCNN2Channel",
    "MLP",
]
