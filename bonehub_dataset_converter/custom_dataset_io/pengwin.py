"""
PENGWIN Dataset Converter
url: https://doi.org/10.5281/zenodo.10927452

description:
This dataset comprises CT scans from 150 patients scheduled for pelvic reduction surgery.
Each bone anatomy (sacrum, left hipbone, right hipbone) has up to 10 fragments.
Bone that does not present any fracuture has only one fragment, which is itself.
Label assignment: 0 = background, 1-10 = sacrum fragment, 11-20 = left hipbone fragment, 21-30 = right hipbone fragment.
"""

from pathlib import Path
import numpy as np
import SimpleITK as sitk
from bonehub_data_schema import SubjectInfo, DatasetInfo, BoneLabelMap as BLM

from .. import BaseDatasetIO, DataSource
from ..utils import export_nii_segmentation, export_image_monai

from . import MAX_SUBJECTS_FOR_TESTING

label_mapping = {
    0: BLM.BACKGROUND.value,
    1: BLM.SACRUM.value,
    2: BLM.SACRUM.value,
    3: BLM.SACRUM.value,
    4: BLM.SACRUM.value,
    5: BLM.SACRUM.value,
    6: BLM.SACRUM.value,
    7: BLM.SACRUM.value,
    8: BLM.SACRUM.value,
    9: BLM.SACRUM.value,
    10: BLM.SACRUM.value,
    11: BLM.HIP_LEFT.value,
    12: BLM.HIP_LEFT.value,
    13: BLM.HIP_LEFT.value,
    14: BLM.HIP_LEFT.value,
    15: BLM.HIP_LEFT.value,
    16: BLM.HIP_LEFT.value,
    17: BLM.HIP_LEFT.value,
    18: BLM.HIP_LEFT.value,
    19: BLM.HIP_LEFT.value,
    20: BLM.HIP_LEFT.value,
    21: BLM.HIP_RIGHT.value,
    22: BLM.HIP_RIGHT.value,
    23: BLM.HIP_RIGHT.value,
    24: BLM.HIP_RIGHT.value,
    25: BLM.HIP_RIGHT.value,
    26: BLM.HIP_RIGHT.value,
    27: BLM.HIP_RIGHT.value,
    28: BLM.HIP_RIGHT.value,
    29: BLM.HIP_RIGHT.value,
    30: BLM.HIP_RIGHT.value,
}


class PENGWIN(BaseDatasetIO):
    """
    Data reader for PENGWIN dataset.
    Expexted structure:
    root_directory/
    ├── PENGWIN_CT_train_images_part1/
    │   ├── 001.mha
    │   ├── 002.mha
    │   ├── ...
    │   └── 050.mha
    ├── PENGWIN_CT_train_images_part2/
    │   ├── 051.mha
    │   ├── 052.mha
    │   ├── ...
    │   └── 100.mha
    └── PENGWIN_CT_train_labels/
        ├── 001.mha
        ├── 002.mha
        ├── ...
        └── 100.mha
    """

    def __init__(self, dataset_root: Path):
        dataset_info = DatasetInfo(
            name="PENGWIN",
            description="PENGWIN dataset",
            url="https://doi.org/10.5281/zenodo.10927452",
            modality="CT",
        )
        super().__init__(dataset_root, dataset_info)
        self.custom_data_handlers.read_dataset = read_dataset
        self.custom_data_handlers.export_image = export_image
        self.custom_data_handlers.export_segmentation = export_segmentation


def read_dataset(dataset_root: Path) -> list[DataSource]:
    subjects = [d for d in (dataset_root / "PENGWIN_CT_train_images_part1").iterdir()] + [
        d for d in (dataset_root / "PENGWIN_CT_train_images_part2").iterdir()
    ]
    subjects = subjects[:MAX_SUBJECTS_FOR_TESTING]
    subjects.sort()
    datalist = []

    for sub in subjects:
        image_path = sub
        seg_path = dataset_root / "PENGWIN_CT_train_labels" / sub.name
        # check fractures
        sitk_image = sitk.ReadImage(str(seg_path))
        seg_array = sitk.GetArrayFromImage(sitk_image)
        unique_labels = np.unique(seg_array)
        fracture_counts = [
            (BLM.SACRUM.name, np.sum((unique_labels >= 1) & (unique_labels <= 10)) - 1),
            (BLM.HIP_LEFT.name, np.sum((unique_labels >= 11) & (unique_labels <= 20)) - 1),
            (BLM.HIP_RIGHT.name, np.sum((unique_labels >= 21) & (unique_labels <= 30)) - 1),
        ]
        remarks = (
            "; ".join(f"{count} fractures at {bone_name}" for bone_name, count in fracture_counts if count > 0) or None
        )
        data = DataSource(
            img_path=image_path,
            segmentation_path=[seg_path],
            subject_info=SubjectInfo(
                source_subject_path=sub.stem,
                imaging_modality="CT",
                image=True,
                remarks=remarks,
            ),
        )
        datalist.append(data)

    return datalist


def export_image(data: DataSource, output_file_path: Path):
    export_image_monai(data.img_path, output_file_path)


def export_segmentation(data: DataSource, output_file_path: Path):
    export_nii_segmentation(
        input_label_paths=data.segmentation_path,
        output_label_path=output_file_path,
        label_mappings=[label_mapping],
    )
