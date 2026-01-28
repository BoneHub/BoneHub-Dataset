"""KiTS 2023 challenge dataset loader."""

from pathlib import Path
from typing import Dict, List, Optional, Union

from ..base import BaseDataLoader, SubjectData


class KiTSLoader(BaseDataLoader):
    """Data loader for KiTS (Kidney Tumor Segmentation) dataset.

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

    def run_data_reader(self) -> List[Dict]:
        case_ids = sorted([d.name for d in self.data_root.iterdir() if d.is_dir() and d.name.startswith("case_")])
        data = []

        for case_id in case_ids:
            case_dir = self.data_root / case_id

            image_path = case_dir / "imaging.nii.gz"
            label_path = case_dir / "segmentation.nii.gz"

            if not image_path.exists():
                raise ValueError(f"Missing imaging file for {case_id}")

            if not label_path.exists():
                raise ValueError(f"Missing segmentation file for {case_id}")

            subject_data = SubjectData(
                image=str(image_path),
                label=str(label_path),
            )

            data.append(subject_data)

        if not data:
            raise ValueError(f"No valid cases found in {self.data_root}")

        return data
