"""
TCIA CT Colonography ARCIN 6664 dataset reader.
Dataset link: https://doi.org/10.7937/K9/TCIA.2015.NWTESAY1
"""

from typing import List

from ..base import BaseDatasetReader
from ..utils import get_dicom_subject_metadata
from ...constants import SubjectData


class ACRIN6664Reader(BaseDatasetReader):
    """Data reader for TCIA CT Colonography with ARCIN 6664 dataset.

    Expected structure:
        root_directory/
            ├── metadata.csv
            ├── TCIA-CTC-6-to-9-mm-polyps.xls
            ├── TCIA-CTC-large-10-mm-polyps.xls
            ├── TCIA-CTC-no-polyp-found.xls
            ├── CT COLONOGRAPHY
                ├── 1.3.6.1.4.1.9328.50.4.0001
                    ├── 01-01-2000-1-Abdomen24ACRINColoIRB2415-04 Adult-0.4.1
                        ├── 1.000000-Topo supine  0.6  T20s-.1222
                            ├── 1-1.dcm
                            ├── 1-2.dcm

                        ├──2.000000-Topo prone  0.6  T20s-.1224
                            ├──1-1.dcm

                        ├── 3.000000-Colosupine  1.0  B30f-4.563
                            ├── 1-001.dcm
                            ├── 1-002.dcm
                            ├── 1-003.dcm

    """

    def run_data_reader(self) -> List[SubjectData]:
        case_ids = sorted([d.name for d in (self.dataset_root / "CT COLONOGRAPHY").iterdir() if d.is_dir()])
        data = []

        for case_id in case_ids:
            case_dir = self.dataset_root / "CT COLONOGRAPHY" / case_id
            case_dir = next(case_dir.iterdir())
            for subdir in case_dir.iterdir():
                if "Colosupine" in subdir.name:
                    dicom_image_dir = subdir
                    subject_orientation = "supine"
                elif "Coloprone" in subdir.name:
                    dicom_image_dir = subdir
                    subject_orientation = "prone"
                else:
                    continue
                subject_metadata = get_dicom_subject_metadata(str(dicom_image_dir))
                subject_data = SubjectData(
                    image=str(dicom_image_dir),
                    dataset_name="CT Colonography ARCIN 6664",
                    case_id=case_id,
                    age=subject_metadata["age"],
                    gender=subject_metadata["gender"],
                    subject_orientation=subject_orientation,
                )

                data.append(subject_data)

        if not data:
            raise ValueError(f"No valid cases found in {self.dataset_root}")

        return data

    def get_label_mapping(self):
        # No specific labels defined for this dataset
        return {}
