"""
BoneHub Data Schema - Standard data structures and constants for BoneHub projects
"""

from .labelmap import BoneLabelMap, bonehub_to_snomed
from .subject_info import SubjectInfo
from .dataset_info import DatasetInfo
from .bonehub_dataset_io import BoneHubDatasetIO


__all__ = [
    "BoneLabelMap",
    "SubjectInfo",
    "DatasetInfo",
    "bonehub_to_snomed",
    "BoneHubDatasetIO",
]

__version__ = "0.1.0"
__author__ = "Hamid Alavi"
