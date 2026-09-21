"""
BoneHub Data Schema - Standard data structures and constants for BoneHub projects
"""

from ._version import __version__, is_compatible_schema_version
from .labelmap import BoneLabelMap, bonehub_to_snomed
from .label_status import VALID_LABEL_VALUES
from .subject_info import SubjectInfo
from .dataset_info import DatasetInfo
from .segmentation_file import (
    SEGMENTATION_SUFFIX,
    segment_number_dtype,
    write_segmentation,
    write_indexed_segmentation,
    read_segmentation,
    read_segmentation_labels,
)
from .bonehub_dataset_io import BoneHubDatasetIO


__all__ = [
    "BoneLabelMap",
    "VALID_LABEL_VALUES",
    "SubjectInfo",
    "DatasetInfo",
    "bonehub_to_snomed",
    "BoneHubDatasetIO",
    "SEGMENTATION_SUFFIX",
    "segment_number_dtype",
    "write_segmentation",
    "write_indexed_segmentation",
    "read_segmentation",
    "read_segmentation_labels",
    "is_compatible_schema_version",
]

__author__ = "Hamid Alavi"
