"""
Enhance-PET-1.6k Dataset Converter
url: https://github.com/ENHANCE-PET/MOOSE/blob/main/DATA_CARD.md
"""

import os
from pathlib import Path
import shutil
import pandas as pd
from bonehub_data_schema import SubjectInfo, DatasetInfo, BoneLabelMap as BLM

from .. import BaseDatasetIO, DataSource
from ..utils import export_nii_segmentation

from . import MAX_SUBJECTS_FOR_TESTING

label_mapping_peripheral_bones = {
    1: BLM.CARPALS_LEFT.value,
    2: BLM.CARPALS_RIGHT.value,
    3: BLM.CLAVICLE_LEFT.value,
    4: BLM.CLAVICLE_RIGHT.value,
    5: BLM.FEMUR_LEFT.value,
    6: BLM.FEMUR_RIGHT.value,
    7: BLM.FIBULA_LEFT.value,
    8: BLM.FIBULA_RIGHT.value,
    9: BLM.PHALANGES_HAND_LEFT.value,
    10: BLM.PHALANGES_HAND_RIGHT.value,
    11: BLM.HUMERUS_LEFT.value,
    12: BLM.HUMERUS_RIGHT.value,
    13: BLM.METACARPALS_LEFT.value,
    14: BLM.METACARPALS_RIGHT.value,
    15: BLM.METATARSALS_LEFT.value,
    16: BLM.METATARSALS_RIGHT.value,
    17: BLM.PATELLA_LEFT.value,
    18: BLM.PATELLA_RIGHT.value,
    19: BLM.RADIUS_LEFT.value,
    20: BLM.RADIUS_RIGHT.value,
    21: BLM.SCAPULA_LEFT.value,
    22: BLM.SCAPULA_RIGHT.value,
    23: BLM.SKULL.value,
    24: BLM.TARSALS_LEFT.value,
    25: BLM.TARSALS_RIGHT.value,
    26: BLM.TIBIA_LEFT.value,
    27: BLM.TIBIA_RIGHT.value,
    28: BLM.PHALANGES_FOOT_LEFT.value,
    29: BLM.PHALANGES_FOOT_RIGHT.value,
    30: BLM.ULNA_LEFT.value,
    31: BLM.ULNA_RIGHT.value,
}

label_mapping_vertebrae = {
    1: BLM.VERTEBRA_C1.value,
    2: BLM.VERTEBRA_C2.value,
    3: BLM.VERTEBRA_C3.value,
    4: BLM.VERTEBRA_C4.value,
    5: BLM.VERTEBRA_C5.value,
    6: BLM.VERTEBRA_C6.value,
    7: BLM.VERTEBRA_C7.value,
    8: BLM.VERTEBRA_T1.value,
    9: BLM.VERTEBRA_T2.value,
    10: BLM.VERTEBRA_T3.value,
    11: BLM.VERTEBRA_T4.value,
    12: BLM.VERTEBRA_T5.value,
    13: BLM.VERTEBRA_T6.value,
    14: BLM.VERTEBRA_T7.value,
    15: BLM.VERTEBRA_T8.value,
    16: BLM.VERTEBRA_T9.value,
    17: BLM.VERTEBRA_T10.value,
    18: BLM.VERTEBRA_T11.value,
    19: BLM.VERTEBRA_T12.value,
    20: BLM.VERTEBRA_L1.value,
    21: BLM.VERTEBRA_L2.value,
    22: BLM.VERTEBRA_L3.value,
    23: BLM.VERTEBRA_L4.value,
    24: BLM.VERTEBRA_L5.value,
    25: BLM.VERTEBRA_L6.value,
    26: BLM.HIP_LEFT.value,
    27: BLM.HIP_RIGHT.value,
    28: BLM.SACRUM.value,
}

label_mapping_ribs = {
    1: BLM.RIB_1_LEFT.value,
    2: BLM.RIB_2_LEFT.value,
    3: BLM.RIB_3_LEFT.value,
    4: BLM.RIB_4_LEFT.value,
    5: BLM.RIB_5_LEFT.value,
    6: BLM.RIB_6_LEFT.value,
    7: BLM.RIB_7_LEFT.value,
    8: BLM.RIB_8_LEFT.value,
    9: BLM.RIB_9_LEFT.value,
    10: BLM.RIB_10_LEFT.value,
    11: BLM.RIB_11_LEFT.value,
    12: BLM.RIB_12_LEFT.value,
    13: BLM.RIB_13_LEFT.value,
    14: BLM.RIB_1_RIGHT.value,
    15: BLM.RIB_2_RIGHT.value,
    16: BLM.RIB_3_RIGHT.value,
    17: BLM.RIB_4_RIGHT.value,
    18: BLM.RIB_5_RIGHT.value,
    19: BLM.RIB_6_RIGHT.value,
    20: BLM.RIB_7_RIGHT.value,
    21: BLM.RIB_8_RIGHT.value,
    22: BLM.RIB_9_RIGHT.value,
    23: BLM.RIB_10_RIGHT.value,
    24: BLM.RIB_11_RIGHT.value,
    25: BLM.RIB_12_RIGHT.value,
    26: BLM.RIB_13_RIGHT.value,
    27: BLM.STERNUM.value,
}


class EnhancePET(BaseDatasetIO):
    """
    Data reader for Enhance-PET-1.6k dataset.
    Expexted structure:
    root_directory/
    ├── CT-details.xlsx
    ├── PT-details.xlsx
    └── imaging-data/
        ├── images/
        │   ├── CT/
        │   │   ├── 0001.nii.gz
        │   │   ├── 0002.nii.gz
        │   │   └── ...
        │   └── PT/
        │       ├── 0001.nii.gz
        │       ├── 0002.nii.gz
        │       └── ...
        └── ground-truth/
            ├── Peripheral-bones/
            │   ├── 0001.nii.gz
            │   ├── 0002.nii.gz
            │   └── ...
            ├── Vertebrae/
            │   ├── 0001.nii.gz
            │   ├── 0002.nii.gz
            │   └── ...
            ├── Ribs/
            │   ├── 0001.nii.gz
            │   ├── 0002.nii.gz
            │   └── ...
            ├─ Body-Composition/
            ├─ Cardiac/
            ├─ Muscles/
            └─ Organs/
    """

    def __init__(self, dataset_root: Path):
        dataset_info = DatasetInfo(
            name="Enhance-PET-1.6k",
            description="Enhance-PET-1.6k dataset",
            url="https://github.com/ENHANCE-PET/MOOSE/blob/main/DATA_CARD.md",
            modality="CT",
        )
        super().__init__(dataset_root, dataset_info)
        self.custom_data_handlers.read_dataset = read_dataset
        self.custom_data_handlers.export_image = export_image
        self.custom_data_handlers.export_segmentation = export_segmentation


def read_dataset(dataset_root: Path) -> list[DataSource]:
    subjects = [d for d in (dataset_root / "imaging-data" / "images" / "CT").iterdir()][:MAX_SUBJECTS_FOR_TESTING]
    subjects.sort()
    datalist = []
    metadata = pd.read_excel(dataset_root / "PT-details.xlsx")

    for sub in subjects:
        sub_id = sub.name.replace(".nii.gz", "")
        subject_metadata = metadata.loc[metadata["Patient"] == int(sub_id)]
        age = int(subject_metadata["Age"].iloc[0])
        gender = subject_metadata["Sex"].iloc[0]
        if pd.isna(gender):
            gender = None
        weight = float(subject_metadata["Weight [kg]"].iloc[0])
        height = subject_metadata["Height [m]"].iloc[0]
        if not pd.isna(height):
            height = float(height) * 100  # Convert height from meters to centimeters
        else:
            height = None
        peripheral_bones_path = dataset_root / "imaging-data" / "ground-truth" / "Peripheral-bones" / f"{sub_id}.nii.gz"
        vertebrae_path = dataset_root / "imaging-data" / "ground-truth" / "Vertebrae" / f"{sub_id}.nii.gz"
        ribs_path = dataset_root / "imaging-data" / "ground-truth" / "Ribs" / f"{sub_id}.nii.gz"
        data = DataSource(
            img_path=sub,
            segmentation_path=[peripheral_bones_path, vertebrae_path, ribs_path],
            subject_info=SubjectInfo(
                source_subject_path=sub_id,
                age=age,
                gender=gender,
                height=height,
                weight=weight,
                imaging_modality="CT",
                image=True,
            ),
        )
        datalist.append(data)

    return datalist


def export_image(data: DataSource, output_file_path: Path):
    shutil.copy(str(data.img_path), str(output_file_path))


def export_segmentation(data: DataSource, output_file_path: Path):
    export_nii_segmentation(
        data.segmentation_path,
        output_file_path,
        [label_mapping_peripheral_bones, label_mapping_vertebrae, label_mapping_ribs],
    )
