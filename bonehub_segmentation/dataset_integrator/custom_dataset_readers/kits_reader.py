"""
KiTS 2023 challenge dataset reader.
Dataset link: https://github.com/neheller/kits23
"""

from typing import List
import json

from ..base import BaseDatasetReader
from .. import SubjectData


class KiTSReader(BaseDatasetReader):
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

    def run_data_reader(self) -> List[SubjectData]:
        with open(self.dataset_root / "kits23.json") as f:
            metadata = json.load(f)
        metadata = {case["case_id"]: case for case in metadata}

        case_ids = sorted([d.name for d in self.dataset_root.iterdir() if d.is_dir() and d.name.startswith("case_")])
        data = []

        for case_id in case_ids:
            case_dir = self.dataset_root / case_id
            image_path = case_dir / "imaging.nii.gz"

            if not image_path.exists():
                raise ValueError(f"Missing imaging file for {case_id}")

            subject_data = SubjectData(
                image=str(image_path),
                age=metadata[case_id]["age_at_nephrectomy"],
                gender=metadata[case_id]["gender"],
            )

            data.append(subject_data)

        if not data:
            raise ValueError(f"No valid cases found in {self.dataset_root}")

        return data

    def get_label_mapping(self):
        # KiTS does not provide segmentation labels for bones
        return {}
