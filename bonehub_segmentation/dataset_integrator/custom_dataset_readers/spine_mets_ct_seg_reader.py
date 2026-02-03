"""
TCIA Spine-Mets-CT-SEG dataset reader.
Dataset link: https://www.cancerimagingarchive.net/collection/spine-mets-ct-seg/
"""

from typing import List

from ..base import BaseDatasetReader, get_dicom_subject_metadata
from .. import SubjectData
from .. import BoneHubLabelMap as BHL


class SpineMetsCTSegReader(BaseDatasetReader):
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

    def run_data_reader(self) -> List[SubjectData]:
        case_ids = sorted([d.name for d in (self.dataset_root / "Spine-Mets-CT-SEG").iterdir() if d.is_dir()])
        data = []

        for case_id in case_ids:
            case_dir = self.dataset_root / "Spine-Mets-CT-SEG" / case_id
            dicom_image_dir = None
            dicom_label_file = None
            study_dir = next(case_dir.iterdir())
            for subdir in study_dir.iterdir():
                if "segmentation" in subdir.name.lower():
                    dicom_label_file = next(subdir.glob("*.dcm"))
                else:
                    dicom_image_dir = subdir

            subject_metadata = get_dicom_subject_metadata(str(dicom_image_dir))
            subject_data = SubjectData(
                image=str(dicom_image_dir),
                label=str(dicom_label_file),
                dataset_name="Spine-Mets-CT-SEG",
                case_id=case_id,
                age=subject_metadata["age"],
                gender=subject_metadata["gender"],
            )

            data.append(subject_data)

        if not data:
            raise ValueError(f"No valid cases found in {self.dataset_root}")

        return data

    def get_label_mapping(self):
        return {
            "C1 vertebra": BHL.VERTEBRA_C1.value,
            "C2 vertebra": BHL.VERTEBRA_C2.value,
            "C3 vertebra": BHL.VERTEBRA_C3.value,
            "C4 vertebra": BHL.VERTEBRA_C4.value,
            "C5 vertebra": BHL.VERTEBRA_C5.value,
            "C6 vertebra": BHL.VERTEBRA_C6.value,
            "C7 vertebra": BHL.VERTEBRA_C7.value,
            "T1 vertebra": BHL.VERTEBRA_T1.value,
            "T2 vertebra": BHL.VERTEBRA_T2.value,
            "T3 vertebra": BHL.VERTEBRA_T3.value,
            "T4 vertebra": BHL.VERTEBRA_T4.value,
            "T5 vertebra": BHL.VERTEBRA_T5.value,
            "T6 vertebra": BHL.VERTEBRA_T6.value,
            "T7 vertebra": BHL.VERTEBRA_T7.value,
            "T8 vertebra": BHL.VERTEBRA_T8.value,
            "T9 vertebra": BHL.VERTEBRA_T9.value,
            "T10 vertebra": BHL.VERTEBRA_T10.value,
            "T11 vertebra": BHL.VERTEBRA_T11.value,
            "T12 vertebra": BHL.VERTEBRA_T12.value,
            "L1 vertebra": BHL.VERTEBRA_L1.value,
            "L2 vertebra": BHL.VERTEBRA_L2.value,
            "L3 vertebra": BHL.VERTEBRA_L3.value,
            "L4 vertebra": BHL.VERTEBRA_L4.value,
            "L5 vertebra": BHL.VERTEBRA_L5.value,
        }
