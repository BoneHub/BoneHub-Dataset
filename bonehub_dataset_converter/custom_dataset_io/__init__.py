"""
input/output operations for custom medical image datasets.
Add your custom dataset here.
"""

import os

if "MAX_SUBJECTS_FOR_TESTING" not in os.environ:
    os.environ["MAX_SUBJECTS_FOR_TESTING"] = ""  # Default to '' (process all subjects) if not set in environment variables

try:
    MAX_SUBJECTS_FOR_TESTING = int(os.environ["MAX_SUBJECTS_FOR_TESTING"])
    print(f"WARNING: MAX_SUBJECTS_FOR_TESTING is set to {MAX_SUBJECTS_FOR_TESTING}. Not all subjects will be processed.")
except ValueError:
    print(
        f"Invalid value for MAX_SUBJECTS_FOR_TESTING: '{os.environ['MAX_SUBJECTS_FOR_TESTING']}'. Defaulting to processing all subjects."
    )
    MAX_SUBJECTS_FOR_TESTING = None

from .kits2023 import KiTS2023
from .spine_mets_ct_seg import SpineMetsCTSeg
from .acrin_6664 import ACRIN6664
from .bonedat import BoneDat
from .vsd_reconstruction import VSDReconstruction

__all__ = [
    "KiTS2023",
    "SpineMetsCTSeg",
    "ACRIN6664",
    "BoneDat",
    "VSDReconstruction",
]
