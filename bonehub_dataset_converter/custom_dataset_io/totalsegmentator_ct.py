"""
TotalSegmentator CT Dataset Converter
url: https://doi.org/10.5281/zenodo.6802613

Note: TotalSegmentator separates S1 from the rest of the sacrum, but BoneHub does not. Therefore, during export, we combine the S1 and sacrum labels into a single sacrum label to ensure compatibility with the BoneHub data schema.
"""

import os
import shutil
from pathlib import Path
import pandas as pd
from bonehub_data_schema import SubjectInfo, DatasetInfo, BoneLabelMap as BLM
import SimpleITK as sitk

from .. import BaseDatasetIO, DataSource
from ..utils import export_nii_segmentation, export_image_monai

from . import MAX_SUBJECTS_FOR_TESTING

label_mapping = {
    "clavicula_left": BLM.CLAVICLE_LEFT.value,
    "clavicula_right": BLM.CLAVICLE_RIGHT.value,
    "femur_left": BLM.FEMUR_LEFT.value,
    "femur_right": BLM.FEMUR_RIGHT.value,
    "hip_left": BLM.HIP_LEFT.value,
    "hip_right": BLM.HIP_RIGHT.value,
    "humerus_left": BLM.HUMERUS_LEFT.value,
    "humerus_right": BLM.HUMERUS_RIGHT.value,
    "rib_left_1": BLM.RIB_1_LEFT.value,
    "rib_left_2": BLM.RIB_2_LEFT.value,
    "rib_left_3": BLM.RIB_3_LEFT.value,
    "rib_left_4": BLM.RIB_4_LEFT.value,
    "rib_left_5": BLM.RIB_5_LEFT.value,
    "rib_left_6": BLM.RIB_6_LEFT.value,
    "rib_left_7": BLM.RIB_7_LEFT.value,
    "rib_left_8": BLM.RIB_8_LEFT.value,
    "rib_left_9": BLM.RIB_9_LEFT.value,
    "rib_left_10": BLM.RIB_10_LEFT.value,
    "rib_left_11": BLM.RIB_11_LEFT.value,
    "rib_left_12": BLM.RIB_12_LEFT.value,
    "rib_right_1": BLM.RIB_1_RIGHT.value,
    "rib_right_2": BLM.RIB_2_RIGHT.value,
    "rib_right_3": BLM.RIB_3_RIGHT.value,
    "rib_right_4": BLM.RIB_4_RIGHT.value,
    "rib_right_5": BLM.RIB_5_RIGHT.value,
    "rib_right_6": BLM.RIB_6_RIGHT.value,
    "rib_right_7": BLM.RIB_7_RIGHT.value,
    "rib_right_8": BLM.RIB_8_RIGHT.value,
    "rib_right_9": BLM.RIB_9_RIGHT.value,
    "rib_right_10": BLM.RIB_10_RIGHT.value,
    "rib_right_11": BLM.RIB_11_RIGHT.value,
    "rib_right_12": BLM.RIB_12_RIGHT.value,
    "sacrum": BLM.SACRUM.value,  # TotalSegmentator separates S1 from the rest of the sacrum, but we will assign the same label to S1 and the rest of the sacrum
    "scapula_left": BLM.SCAPULA_LEFT.value,
    "scapula_right": BLM.SCAPULA_RIGHT.value,
    "skull": BLM.SKULL.value,
    "sternum": BLM.STERNUM.value,
    "vertebrae_C1": BLM.VERTEBRA_C1.value,
    "vertebrae_C2": BLM.VERTEBRA_C2.value,
    "vertebrae_C3": BLM.VERTEBRA_C3.value,
    "vertebrae_C4": BLM.VERTEBRA_C4.value,
    "vertebrae_C5": BLM.VERTEBRA_C5.value,
    "vertebrae_C6": BLM.VERTEBRA_C6.value,
    "vertebrae_C7": BLM.VERTEBRA_C7.value,
    "vertebrae_L1": BLM.VERTEBRA_L1.value,
    "vertebrae_L2": BLM.VERTEBRA_L2.value,
    "vertebrae_L3": BLM.VERTEBRA_L3.value,
    "vertebrae_L4": BLM.VERTEBRA_L4.value,
    "vertebrae_L5": BLM.VERTEBRA_L5.value,
    "vertebrae_S1": BLM.SACRUM.value,  # TotalSegmentator separates S1 from the rest of the sacrum, but we will assign the same label to S1 and the rest of the sacrum
    "vertebrae_T1": BLM.VERTEBRA_T1.value,
    "vertebrae_T2": BLM.VERTEBRA_T2.value,
    "vertebrae_T3": BLM.VERTEBRA_T3.value,
    "vertebrae_T4": BLM.VERTEBRA_T4.value,
    "vertebrae_T5": BLM.VERTEBRA_T5.value,
    "vertebrae_T6": BLM.VERTEBRA_T6.value,
    "vertebrae_T7": BLM.VERTEBRA_T7.value,
    "vertebrae_T8": BLM.VERTEBRA_T8.value,
    "vertebrae_T9": BLM.VERTEBRA_T9.value,
    "vertebrae_T10": BLM.VERTEBRA_T10.value,
    "vertebrae_T11": BLM.VERTEBRA_T11.value,
    "vertebrae_T12": BLM.VERTEBRA_T12.value,
}


class TotalSegmentatorCT(BaseDatasetIO):
    """
    Data reader for TotalSegmentator CT dataset.
    Expexted structure:
    root_directory/
    ├── meta.csv
    ├── s0000/
    │   ├── ct.nii.gz
    │   ├── segmentations/
    │   │   ├── femur_left.nii.gz
    │   │   ├── femur_right.nii.gz
    │   │   └── ...
    ├── s0001/
    │   └── ...
    └── ...
    """

    def __init__(self, dataset_root: Path):
        dataset_info = DatasetInfo(
            name="TotalSegmentator CT",
            description="TotalSegmentator CT dataset version 2.0.1",
            url="https://doi.org/10.5281/zenodo.6802613",
            modality="CT",
        )
        super().__init__(dataset_root, dataset_info)
        self.custom_data_handlers.read_dataset = read_dataset
        self.custom_data_handlers.export_image = export_image
        self.custom_data_handlers.export_segmentation = export_segmentation


def read_dataset(dataset_root: Path) -> list[DataSource]:
    subjects = [d for d in dataset_root.iterdir() if d.is_dir()][:MAX_SUBJECTS_FOR_TESTING]
    subjects.sort()
    datalist = []
    metadata = pd.read_csv(dataset_root / "meta.csv", delimiter=";")

    for sub in subjects:
        subject_metadata = metadata.loc[metadata["image_id"] == sub.name]
        age = None if pd.isna(v := subject_metadata["age"].iloc[0]) else int(v)
        gender = None if pd.isna(v := subject_metadata["gender"].iloc[0]) else v.upper()
        segmentations = [sub / "segmentations" / f"{bone_label}.nii.gz" for bone_label in label_mapping]
        data = DataSource(
            img_path=sub / "ct.nii.gz",
            segmentation_path=segmentations,
            subject_info=SubjectInfo(
                source_subject_path=sub.name,
                age=age,
                gender=gender,
                imaging_modality="CT",
                image=True,
            ),
        )
        datalist.append(data)

    return datalist


def export_image(data: DataSource, output_file_path: Path):
    """
    In TotalSegmentator dataset, eventhough the images are already in NIfTI format, some of them cannot be read correctly even with 3DSlicer.
    Therefore, instead of directly copying the files, we use the export_image_monai function to read and write the images,
    which ensures that the output images are in a consistent format that can be read by all tools.
    """
    export_image_monai(data.img_path, output_file_path)


def export_segmentation(data: DataSource, output_file_path: Path):
    """
    In TotalSegmentator dataset, there are two issues regarding the segmentation files:
    1) The segmentations are provided as separate files for each bone
    2) Even though the segmentations are in NIfTI format, some of them cannot be read correctly by 3DSlicer, likely due to some inconsistencies in the file formatting.
    To address these issues, we first read and write each segmentation file using the export_image_monai function to ensure they are in a consistent format,
    and then we combine them into a single segmentation file with the correct BoneHub labels using the export_nii_segmentation function.
    """

    temp_folder = output_file_path.parent / f"temp_{output_file_path.name}"
    os.makedirs(temp_folder, exist_ok=True)
    temp_seg_paths = []
    for seg_path in data.segmentation_path:
        temp_output_path = temp_folder / seg_path.name
        export_image_monai(seg_path, temp_output_path)
        temp_seg_paths.append(temp_output_path)
    export_nii_segmentation(
        temp_seg_paths,
        output_file_path,
        [{1: label_mapping[bone_label.name.replace(".nii.gz", "")]} for bone_label in data.segmentation_path],
    )
    shutil.rmtree(temp_folder, ignore_errors=True)
