"""
input/output operations for custom medical image datasets.
Add your custom dataset here.
"""

import os

if "MAX_SUBJECTS_FOR_TESTING" not in os.environ:
    os.environ["MAX_SUBJECTS_FOR_TESTING"] = "-1"

MAX_SUBJECTS_FOR_TESTING = int(os.environ["MAX_SUBJECTS_FOR_TESTING"])

if MAX_SUBJECTS_FOR_TESTING != -1:
    print(f"WARNING: MAX_SUBJECTS_FOR_TESTING is set to {MAX_SUBJECTS_FOR_TESTING}. Not all subjects will be processed.")


from .kits2023 import KiTS2023
from .spine_mets_ct_seg import SpineMetsCTSeg
from .acrin_6664 import ACRIN6664
from .bonedat import BoneDat

__all__ = [
    "KiTS2023",
    "SpineMetsCTSeg",
    "ACRIN6664",
    "BoneDat",
]
