"""
TCIA Spine-Mets-CT-SEG dataset.
Dataset link: https://www.cancerimagingarchive.net/collection/spine-mets-ct-seg/
"""

from pathlib import Path

from bonehub_data_schema import SubjectInfo, DatasetInfo
from bonehub_data_schema import BoneLabelMap as BLM

from .. import BaseDatasetIO, DataSource
from ..utils import get_dicom_subject_metadata, export_image, export_dicom_segmentation

from . import MAX_SUBJECTS_FOR_TESTING


class SpineMetsCTSeg(BaseDatasetIO):
    """Data reader for TCIA Spine-Mets-CT-SEG dataset.

    Expected structure:
        root_directory/
            ├── metadata.csv
            └── Spine-Mets-CT-SEG/
                ├───10250
                │   └───04-04-2009-NA-SpineSPINEBONESBRT Adult-04098
                │       ├───300.000000-Spine Segmentation-86171
                │       │       1-1.dcm
                │       │
                │       └───5.000000-SKINTOSKINSIM0.5MM10250a iMAR-27242
                │               1-001.dcm
                │               1-002.dcm
                │               1-003.dcm
    """

    label_map = {
        "C1 vertebra": BLM.VERTEBRA_C1.value,
        "C2 vertebra": BLM.VERTEBRA_C2.value,
        "C3 vertebra": BLM.VERTEBRA_C3.value,
        "C4 vertebra": BLM.VERTEBRA_C4.value,
        "C5 vertebra": BLM.VERTEBRA_C5.value,
        "C6 vertebra": BLM.VERTEBRA_C6.value,
        "C7 vertebra": BLM.VERTEBRA_C7.value,
        "T1 vertebra": BLM.VERTEBRA_T1.value,
        "T2 vertebra": BLM.VERTEBRA_T2.value,
        "T3 vertebra": BLM.VERTEBRA_T3.value,
        "T4 vertebra": BLM.VERTEBRA_T4.value,
        "T5 vertebra": BLM.VERTEBRA_T5.value,
        "T6 vertebra": BLM.VERTEBRA_T6.value,
        "T7 vertebra": BLM.VERTEBRA_T7.value,
        "T8 vertebra": BLM.VERTEBRA_T8.value,
        "T9 vertebra": BLM.VERTEBRA_T9.value,
        "T10 vertebra": BLM.VERTEBRA_T10.value,
        "T11 vertebra": BLM.VERTEBRA_T11.value,
        "T12 vertebra": BLM.VERTEBRA_T12.value,
        "L1 vertebra": BLM.VERTEBRA_L1.value,
        "L2 vertebra": BLM.VERTEBRA_L2.value,
        "L3 vertebra": BLM.VERTEBRA_L3.value,
        "L4 vertebra": BLM.VERTEBRA_L4.value,
        "L5 vertebra": BLM.VERTEBRA_L5.value,
    }

    def __init__(self, dataset_root: Path):
        dataset_info = DatasetInfo(
            name="Spine-Mets-CT-SEG",
            description="TCIA Spine-Mets-CT-SEG dataset",
            url="https://www.cancerimagingarchive.net/collection/spine-mets-ct-seg/",
            modality="CT",
        )
        super().__init__(dataset_root, dataset_info)
        self.register_data_handler("read_dataset", read_dataset)
        self.register_data_handler("export_image", _export_image)
        self.register_data_handler("export_segmentation", export_segmentation)


def read_dataset(dataset_root: Path) -> list[DataSource]:
    case_ids = sorted([d.name for d in (dataset_root / "Spine-Mets-CT-SEG").iterdir() if d.is_dir()])[:MAX_SUBJECTS_FOR_TESTING]
    datalist = []

    for case_id in case_ids:
        case_dir = dataset_root / "Spine-Mets-CT-SEG" / case_id
        dicom_image_dir = None
        dicom_label_file = None
        study_dir = next(case_dir.iterdir())
        for subdir in study_dir.iterdir():
            if "segmentation" in subdir.name.lower():
                dicom_label_file = next(subdir.glob("*.dcm"))
            else:
                dicom_image_dir = subdir

        subject_metadata = get_dicom_subject_metadata(str(dicom_image_dir))
        data = DataSource(
            img_path=str(dicom_image_dir),
            segmentation_path=str(dicom_label_file),
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


def _export_image(data: DataSource, output_file_path: Path):
    export_image(data.img_path, output_file_path)


def export_segmentation(data: DataSource, output_file_path: Path):
    export_dicom_segmentation(data.img_path, data.segmentation_path, output_file_path, SpineMetsCTSeg.label_map)
