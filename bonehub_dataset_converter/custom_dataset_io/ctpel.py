"""
CTPEL dataset.
Dataset link: https://doi.org/10.23698/aida/ctpel
"""

from pathlib import Path

from bonehub_data_schema import SubjectInfo, DatasetInfo
from bonehub_data_schema import BoneLabelMap as BLM

from .. import BaseDatasetIO, DataSource
from ..utils import get_dicom_subject_metadata, export_image_monai, export_dicom_segmentation

from . import MAX_SUBJECTS_FOR_TESTING

label_mapping = {
    "Right Femur": BLM.FEMUR_PROXIMAL_RIGHT.value,
    "Left Femur": BLM.FEMUR_PROXIMAL_LEFT.value,
    "Right Hip": BLM.HIP_RIGHT.value,
    "Left Hip": BLM.HIP_LEFT.value,
    "Sacrum": BLM.SACRUM.value,
}


class CTPEL(BaseDatasetIO):
    """Data reader for CTPEL dataset.

    Expected structure:
    root_directory/
    ├── 0A44743795D421F7/
    │   ├──im_1/
    │   │   ├── i0000,0000b.dcm
    │   │   ├── i0001,0000b.dcm
    │   │   └── ...
    │   ├── im_2/
    │   │   └── x0000.dcm
    │   ├── im_3/
    │   │   └── x0000.dcm
    │   ├── im_4/
    │   │   └── x0000.dcm
    │   ├── im_5/
    │   │   └── x0000.dcm
    │   └── im_6/
    │       └── x0000.dcm
    ├── 0AD7CE889B4FB16F/
    │   └── ...
    └── ...
    """

    def __init__(self, dataset_root: Path):
        dataset_info = DatasetInfo(
            name="CTPEL",
            description="CTPEL dataset",
            url="https://doi.org/10.23698/aida/ctpel",
            modality="CT",
        )
        super().__init__(dataset_root, dataset_info)
        self.custom_data_handlers.read_dataset = read_dataset
        self.custom_data_handlers.export_image = export_image
        self.custom_data_handlers.export_segmentation = export_segmentation


def read_dataset(dataset_root: Path) -> list[DataSource]:
    case_ids = sorted([d.name for d in dataset_root.iterdir() if d.is_dir()])[:MAX_SUBJECTS_FOR_TESTING]
    datalist = []

    for case_id in case_ids:
        case_dir = dataset_root / case_id
        dicom_image_dir = case_dir / "im_1"
        dicom_label_file = case_dir / "im_3" / "x0000.dcm"
        subject_metadata = get_dicom_subject_metadata(str(dicom_image_dir))
        data = DataSource(
            img_path=dicom_image_dir,
            segmentation_path=[dicom_label_file],
            subject_info=SubjectInfo(
                source_subject_path=case_id,
                age=subject_metadata["age"],
                gender=subject_metadata["gender"],
                imaging_modality=subject_metadata["modality"],
                image=True,
            ),
        )

        datalist.append(data)

    if not datalist:
        raise ValueError(f"No valid cases found in {dataset_root}")

    return datalist


def export_image(data: DataSource, output_file_path: Path):
    export_image_monai(data.img_path, output_file_path)


def export_segmentation(data: DataSource, output_file_path: Path):
    export_dicom_segmentation(
        input_image_path=data.img_path,
        input_label_path=data.segmentation_path[0],
        output_label_path=output_file_path,
        label_mapping=label_mapping,
        dicom_segment_key="SegmentDescription",
    )
