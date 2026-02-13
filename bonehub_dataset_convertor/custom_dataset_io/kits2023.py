"""
KiTS 2023 challenge dataset.
Dataset link: https://github.com/neheller/kits23
"""

from pathlib import Path
from typing import List
import json
import shutil

from .. import BaseDatasetIO, DataSource

from bonehub_data_schema import SubjectInfo, DatasetInfo


class KiTS2023(BaseDatasetIO):
    """Data reader for KiTS 2023 challenge dataset.

    Expected structure:
        data_root/
        ├── case_00000/
        │   ├── imaging.nii.gz
        │   ├── segmentation.nii.gz
        │   └── instances/
        │       ├── kidney_instance-1_annotation-1.nii.gz
        │       └── ...
        ├── case_00001/
        │   └── ...
    """

    def __init__(self, dataset_root: Path):
        dataset_info = DatasetInfo(
            name="KiTS 2023",
            description="Kidney Tumor Segmentation Challenge 2023 dataset",
        )
        super().__init__(dataset_root, dataset_info)
        self.register_data_handler("read_dataset", read_dataset)
        self.register_data_handler("export_image", export_image)


def read_dataset(dataset_root: Path) -> List[DataSource]:
    with open(dataset_root / "kits23.json") as f:
        metadata = json.load(f)
    metadata = {case["case_id"]: case for case in metadata}

    case_ids = sorted([d.name for d in dataset_root.iterdir() if d.is_dir() and d.name.startswith("case_")])
    datalist = []

    for case_id in case_ids:
        case_dir = dataset_root / case_id
        image_path = case_dir / "imaging.nii.gz"

        if not image_path.exists():
            raise ValueError(f"Missing imaging file for {case_id}")

        data = DataSource(
            img_path=str(image_path),
            subject_info=SubjectInfo(
                source_subject_path=case_id,
                # TODO: confirm if this is the correct age field to use
                # age=int(metadata[case_id]["age_at_nephrectomy"]),
                gender=metadata[case_id]["gender"],
                bmi=metadata[case_id]["bmi"],
            ),
        )

        datalist.append(data)

    if not datalist:
        raise ValueError(f"No valid cases found in {dataset_root}")

    return datalist


def export_image(data: DataSource, output_file_path: Path):
    shutil.copyfile(data.img_path, output_file_path)
