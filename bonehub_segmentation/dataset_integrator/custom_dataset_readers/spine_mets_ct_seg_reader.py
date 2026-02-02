"""
TCIA Spine-Mets-CT-SEG dataset reader.
Dataset link: https://www.cancerimagingarchive.net/collection/spine-mets-ct-seg/
"""

from typing import List

from ..base import BaseDatasetReader, get_dicom_subject_metadata
from .. import SubjectData
from .. import BoneHubLabels as BHL


class SpineMetsCTSegReader(BaseDatasetReader):
    """Data reader for TCIA Spine-Mets-CT-SEG dataset.

    Expected structure:
        +---10250
        |   \---04-04-2009-NA-SpineSPINEBONESBRT Adult-04098
        |       +---300.000000-Spine Segmentation-86171
        |       |       1-1.dcm
        |       |
        |       \---5.000000-SKINTOSKINSIM0.5MM10250a iMAR-27242
        |               1-001.dcm
        |               1-002.dcm
        |               1-003.dcm
        |               1-004.dcm
    """

    def run_data_reader(self) -> List[SubjectData]:
        case_ids = sorted([d.name for d in self.dataset_root.iterdir() if d.is_dir()])
        data = []

        for case_id in case_ids:
            case_dir = self.dataset_root / case_id
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
                age=subject_metadata["age"],
                gender=subject_metadata["gender"],
            )

            data.append(subject_data)

        if not data:
            raise ValueError(f"No valid cases found in {self.dataset_root}")

        return data

    def get_label_mapping(self):
        return {
            0: BHL.BACKGROUND.value,
            1: BHL.VERTEBRA_T1.value,
            2: BHL.VERTEBRA_T2.value,
            3: BHL.VERTEBRA_T3.value,
            4: BHL.VERTEBRA_T4.value,
            5: BHL.VERTEBRA_T5.value,
            6: BHL.VERTEBRA_T6.value,
            7: BHL.VERTEBRA_T7.value,
            8: BHL.VERTEBRA_T8.value,
            9: BHL.VERTEBRA_T9.value,
            10: BHL.VERTEBRA_T10.value,
            11: BHL.VERTEBRA_T11.value,
            12: BHL.VERTEBRA_T12.value,
            13: BHL.VERTEBRA_L1.value,
            14: BHL.VERTEBRA_L2.value,
        }
