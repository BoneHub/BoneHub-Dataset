"""
BoneDat dataset reader.
Dataset link: https://doi.org/10.5281/zenodo.15189760
"""

from pathlib import Path
import shutil
from bonehub_data_schema import SubjectInfo, DatasetInfo, BoneLabelMap as BLM
import pandas as pd

from .. import BaseDatasetIO, DataSource
from ..utils import export_nii_nrrd_segmentation


_gender_mapping = {"f": "female", "m": "male"}


class BoneDat(BaseDatasetIO):
    """Data reader for BoneDat dataset.

    Expected structure:
        data_root/
        ├── raw/
        │   ├── subjectid/
        │   │   ├── original.nii.gz
        │   │   ├── calibration_constants.csv
        │   │   └── metadata.csv
        derived/
        │   ├── geometry/
        │   ├── registration/
        │   ├── segmentation/
        │   │   ├── subjectid/
        │   │   │  ├── mask.nii.gz
        │   │   │  └── masked.nii.gz
    """

    label_mapping = {
        0: BLM.BACKGROUND.value,
        1: BLM.HIP_RIGHT.value,
        2: BLM.HIP_LEFT.value,
        3: BLM.SACRUM.value,
        4: BLM.VERTEBRA_L5.value,
        5: BLM.VERTEBRA_L4.value,
    }

    def __init__(self, dataset_root: Path):
        dataset_info = DatasetInfo(
            name="BoneDat",
            description="BoneDat dataset",
        )
        super().__init__(dataset_root, dataset_info)
        self.register_data_handler("read_dataset", read_dataset)
        self.register_data_handler("export_image", export_image)
        self.register_data_handler("export_segmentation", export_segmentation)


def read_dataset(dataset_root: Path) -> list[DataSource]:
    subject_dirs = sorted([d for d in (dataset_root / "raw").iterdir() if d.is_dir()])
    datalist = []

    for subject_dir in subject_dirs:
        image_path = subject_dir / "original.nii.gz"
        segmentation_path = dataset_root / "derived" / "segmentation" / subject_dir.name / "mask.nii.gz"

        if not image_path.exists():
            raise ValueError(f"Missing image file for {subject_dir.name}")

        if not segmentation_path.exists():
            raise ValueError(f"Missing segmentation file for {subject_dir.name}")

        with open(subject_dir / "metadata.xlsx", "r") as f:
            metadata = pd.read_excel(subject_dir / "metadata.xlsx")

        data = DataSource(
            img_path=image_path,
            segmentation_path=segmentation_path,
            subject_info=SubjectInfo(
                subject_id_source=subject_dir.name,
                age=int(metadata["CT date"].iloc[0]) - int(metadata["born"].iloc[0]),
                gender=_gender_mapping[metadata["sex"].iloc[0].lower()],
            ),
        )

        datalist.append(data)

    if not datalist:
        raise ValueError(f"No valid subjects found in {dataset_root}")

    return datalist


def export_image(data: DataSource, output_file_path: Path):
    shutil.copyfile(data.img_path, output_file_path)


def export_segmentation(data: DataSource, output_file_path: Path):
    export_nii_nrrd_segmentation(data.segmentation_path, output_file_path, label_mapping=BoneDat.label_mapping)
