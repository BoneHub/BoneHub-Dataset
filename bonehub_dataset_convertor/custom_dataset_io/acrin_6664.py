"""
TCIA CT Colonography ARCIN 6664 dataset.
Dataset link: https://doi.org/10.7937/K9/TCIA.2015.NWTESAY1
"""

from typing import List
from pathlib import Path

from .. import BaseDatasetIO, DataSource
from ..utils import get_dicom_subject_metadata, export_image
from bonehub_data_schema import SubjectInfo, DatasetInfo


class ACRIN6664(BaseDatasetIO):
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

    def __init__(self, dataset_root: Path):
        dataset_info = DatasetInfo(
            name="CT Colonography ARCIN 6664",
            description="TCIA CT Colonography ARCIN 6664 dataset.",
            url="https://doi.org/10.7937/K9/TCIA.2015.NWTESAY1",
        )
        super().__init__(dataset_root, dataset_info)
        self.register_data_handler("read_dataset", read_dataset)
        self.register_data_handler("export_image", _export_image)


def read_dataset(dataset_root: Path) -> List[DataSource]:
    case_ids = sorted([d.name for d in (dataset_root / "CT COLONOGRAPHY").iterdir() if d.is_dir()])
    datalist = []

    for case_id in case_ids:
        case_dir = dataset_root / "CT COLONOGRAPHY" / case_id
        case_dir = next(case_dir.iterdir())
        for subdir in case_dir.iterdir():
            file_count = len(list(subdir.glob("*.dcm")))
            if file_count <= 50:
                continue
            subject_orientation = ""
            if "supin" in subdir.name.lower():
                subject_orientation = "supine"
            if "pron" in subdir.name.lower():
                subject_orientation = "prone"

            subject_metadata = get_dicom_subject_metadata(str(subdir))
            gender = subject_metadata["gender"].lower()
            age = subject_metadata["age"]
            if age:
                age = "".join([c for c in age if c.isdigit()])
                age = int(age)
            data = DataSource(
                img_path=str(subdir),
                subject_info=SubjectInfo(
                    subject_id_source=str(subdir.relative_to(dataset_root / "CT COLONOGRAPHY")),
                    gender=gender,
                    age=age,
                    subject_orientation=subject_orientation,
                ),
            )

            datalist.append(data)

    if not datalist:
        raise ValueError(f"No valid cases found in {dataset_root}")

    return datalist


def _export_image(data: DataSource, output_file_path: Path):
    export_image(data.img_path, output_file_path)
