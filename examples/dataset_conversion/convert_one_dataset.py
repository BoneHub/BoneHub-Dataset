from pathlib import Path

bonehub_dataset_root = Path("Z:/BoneHub/BoneHub_Dataset")

from bonehub_dataset_converter.custom_dataset_io import CTPEL

data_root = Path("Z:/BoneHub/Public_Datasets/138 ctpel")
CTPEL(dataset_root=data_root).export_to_bonehub_format(
    bonehub_dataset_root,
    output_dataset_id=9,
    overwrite=True,
    num_workers=8,
    skip_existing_subjects=True,
)
