"""
BoneHub Data Schema - Standard data structures and constants for BoneHub projects
"""

from ._version import __version__, is_compatible_schema_version
from .labelmap import BoneLabelMap, bonehub_to_snomed
from .label_status import LabelStatus, Origin, Review
from .subject_info import SubjectInfo
from .dataset_info import DatasetInfo
from .segmentation_file import SEGMENTATION_SUFFIX, write_segmentation, read_segmentation_labels
from .bonehub_dataset_io import BoneHubDatasetIO


__all__ = [
    "BoneLabelMap",
    "LabelStatus",
    "Origin",
    "Review",
    "SubjectInfo",
    "DatasetInfo",
    "bonehub_to_snomed",
    "BoneHubDatasetIO",
    "SEGMENTATION_SUFFIX",
    "write_segmentation",
    "read_segmentation_labels",
    "is_compatible_schema_version",
]

__author__ = "Hamid Alavi"
