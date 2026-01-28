"""
Custom medical image dataset loaders.
Add your custom dataset loaders here.
"""

# Import your custom loaders here
from .kits_loader import KiTSLoader

__all__ = [
    "KiTSLoader",
]