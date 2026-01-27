"""
Custom medical image data loaders.
Add your custom data loaders here.
"""

# Import your custom loaders here
from .msd_loader import MSDLoader

__all__ = [
    "MSDLoader",
]
