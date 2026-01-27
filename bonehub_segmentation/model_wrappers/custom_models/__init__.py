"""
Custom model wrappers.
Add your custom model wrappers here.
"""

# Import your custom wrappers here
from .nnunet_wrapper import CustomNNUNetWrapper
from .moose_wrapper import MOOSEWrapper
from .totalsegmentator_wrapper import TotalSegmentatorWrapper

__all__ = [
    "CustomNNUNetWrapper",
    "MOOSEWrapper",
    "TotalSegmentatorWrapper",
]
