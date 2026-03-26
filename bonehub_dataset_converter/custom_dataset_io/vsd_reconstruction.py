"""
VSD Full Body Reconstruction Dataset.
url: https://doi.org/10.5281/zenodo.8302449

Note: This scripts works on a processed version of the dataset where:
    - NRRD files for lower extremity are combined for each subject.
    - Segmentation masks are obtained from the high quality meshes provided in this link https://github.com/MCM-Fischer/VSDFullBodyBoneModels
"""

import json
import os
from pathlib import Path
import shutil
from bonehub_data_schema import SubjectInfo, DatasetInfo, BoneLabelMap as BLM

from .. import BaseDatasetIO, DataSource
from ..utils import export_nii_segmentation

from . import MAX_SUBJECTS_FOR_TESTING

label_mapping = {
    1: BLM.SACRUM.value,
    2: BLM.HIP_LEFT.value,
    3: BLM.HIP_RIGHT.value,
    4: BLM.FEMUR_LEFT.value,
    5: BLM.FEMUR_RIGHT.value,
    6: BLM.PATELLA_LEFT.value,
    7: BLM.PATELLA_RIGHT.value,
    8: BLM.TIBIA_LEFT.value,
    9: BLM.TIBIA_RIGHT.value,
    10: BLM.FIBULA_LEFT.value,
    11: BLM.FIBULA_RIGHT.value,
    12: BLM.TALUS_LEFT.value,
    13: BLM.TALUS_RIGHT.value,
    14: BLM.CALCANEUS_LEFT.value,
    15: BLM.CALCANEUS_RIGHT.value,
    16: BLM.NAVICULAR_LEFT.value,
    17: BLM.NAVICULAR_RIGHT.value,
    18: BLM.CUBOID_LEFT.value,
    19: BLM.CUBOID_RIGHT.value,
    20: BLM.MEDIAL_CUNEIFORM_LEFT.value,
    21: BLM.MEDIAL_CUNEIFORM_RIGHT.value,
    22: BLM.INTERMEDIATE_CUNEIFORM_LEFT.value,
    23: BLM.INTERMEDIATE_CUNEIFORM_RIGHT.value,
    24: BLM.LATERAL_CUNEIFORM_LEFT.value,
    25: BLM.LATERAL_CUNEIFORM_RIGHT.value,
    26: BLM.METATARSAL_1_LEFT.value,
    27: BLM.METATARSAL_1_RIGHT.value,
    28: BLM.METATARSAL_2_LEFT.value,
    29: BLM.METATARSAL_2_RIGHT.value,
    30: BLM.METATARSAL_3_LEFT.value,
    31: BLM.METATARSAL_3_RIGHT.value,
    32: BLM.METATARSAL_4_LEFT.value,
    33: BLM.METATARSAL_4_RIGHT.value,
    34: BLM.METATARSAL_5_LEFT.value,
    35: BLM.METATARSAL_5_RIGHT.value,
    36: BLM.PHALANGE_FOOT_1_1_LEFT.value,
    37: BLM.PHALANGE_FOOT_1_1_RIGHT.value,
    38: BLM.PHALANGE_FOOT_1_2_LEFT.value,
    39: BLM.PHALANGE_FOOT_1_2_RIGHT.value,
    40: BLM.PHALANGE_FOOT_2_1_LEFT.value,
    41: BLM.PHALANGE_FOOT_2_1_RIGHT.value,
    42: BLM.PHALANGE_FOOT_2_2_LEFT.value,
    43: BLM.PHALANGE_FOOT_2_2_RIGHT.value,
    44: BLM.PHALANGE_FOOT_2_3_LEFT.value,
    45: BLM.PHALANGE_FOOT_2_3_RIGHT.value,
    46: BLM.PHALANGE_FOOT_3_1_LEFT.value,
    47: BLM.PHALANGE_FOOT_3_1_RIGHT.value,
    48: BLM.PHALANGE_FOOT_3_2_LEFT.value,
    49: BLM.PHALANGE_FOOT_3_2_RIGHT.value,
    50: BLM.PHALANGE_FOOT_3_3_LEFT.value,
    51: BLM.PHALANGE_FOOT_3_3_RIGHT.value,
    52: BLM.PHALANGE_FOOT_4_1_LEFT.value,
    53: BLM.PHALANGE_FOOT_4_1_RIGHT.value,
    54: BLM.PHALANGE_FOOT_4_2_LEFT.value,
    55: BLM.PHALANGE_FOOT_4_2_RIGHT.value,
    56: BLM.PHALANGE_FOOT_4_3_LEFT.value,
    57: BLM.PHALANGE_FOOT_4_3_RIGHT.value,
    58: BLM.PHALANGE_FOOT_5_1_LEFT.value,
    59: BLM.PHALANGE_FOOT_5_1_RIGHT.value,
    60: BLM.PHALANGE_FOOT_5_2_LEFT.value,
    61: BLM.PHALANGE_FOOT_5_2_RIGHT.value,
    62: BLM.PHALANGE_FOOT_5_3_LEFT.value,
    63: BLM.PHALANGE_FOOT_5_3_RIGHT.value,
}

mesh_label_mapping = {
    "Sacrum": BLM.SACRUM.name,
    "Hip_L": BLM.HIP_LEFT.name,
    "Hip_R": BLM.HIP_RIGHT.name,
    "Femur_L": BLM.FEMUR_LEFT.name,
    "Femur_R": BLM.FEMUR_RIGHT.name,
    "Patella_L": BLM.PATELLA_LEFT.name,
    "Patella_R": BLM.PATELLA_RIGHT.name,
    "Tibia_L": BLM.TIBIA_LEFT.name,
    "Tibia_R": BLM.TIBIA_RIGHT.name,
    "Fibula_L": BLM.FIBULA_LEFT.name,
    "Fibula_R": BLM.FIBULA_RIGHT.name,
    "Talus_L": BLM.TALUS_LEFT.name,
    "Talus_R": BLM.TALUS_RIGHT.name,
    "Calcaneus_L": BLM.CALCANEUS_LEFT.name,
    "Calcaneus_R": BLM.CALCANEUS_RIGHT.name,
    "Navicular_L": BLM.NAVICULAR_LEFT.name,
    "Navicular_R": BLM.NAVICULAR_RIGHT.name,
    "Cuboid_L": BLM.CUBOID_LEFT.name,
    "Cuboid_R": BLM.CUBOID_RIGHT.name,
    "Medial_Cuneiform_L": BLM.MEDIAL_CUNEIFORM_LEFT.name,
    "Medial_Cuneiform_R": BLM.MEDIAL_CUNEIFORM_RIGHT.name,
    "Intermediate_Cuneiform_L": BLM.INTERMEDIATE_CUNEIFORM_LEFT.name,
    "Intermediate_Cuneiform_R": BLM.INTERMEDIATE_CUNEIFORM_RIGHT.name,
    "Lateral_Cuneiform_L": BLM.LATERAL_CUNEIFORM_LEFT.name,
    "Lateral_Cuneiform_R": BLM.LATERAL_CUNEIFORM_RIGHT.name,
    "Metatarsal_1_L": BLM.METATARSAL_1_LEFT.name,
    "Metatarsal_1_R": BLM.METATARSAL_1_RIGHT.name,
    "Metatarsal_2_L": BLM.METATARSAL_2_LEFT.name,
    "Metatarsal_2_R": BLM.METATARSAL_2_RIGHT.name,
    "Metatarsal_3_L": BLM.METATARSAL_3_LEFT.name,
    "Metatarsal_3_R": BLM.METATARSAL_3_RIGHT.name,
    "Metatarsal_4_L": BLM.METATARSAL_4_LEFT.name,
    "Metatarsal_4_R": BLM.METATARSAL_4_RIGHT.name,
    "Metatarsal_5_L": BLM.METATARSAL_5_LEFT.name,
    "Metatarsal_5_R": BLM.METATARSAL_5_RIGHT.name,
    "Phalange_Foot_1_1_L": BLM.PHALANGE_FOOT_1_1_LEFT.name,
    "Phalange_Foot_1_1_R": BLM.PHALANGE_FOOT_1_1_RIGHT.name,
    "Phalange_Foot_1_2_L": BLM.PHALANGE_FOOT_1_2_LEFT.name,
    "Phalange_Foot_1_2_R": BLM.PHALANGE_FOOT_1_2_RIGHT.name,
    "Phalange_Foot_2_1_L": BLM.PHALANGE_FOOT_2_1_LEFT.name,
    "Phalange_Foot_2_1_R": BLM.PHALANGE_FOOT_2_1_RIGHT.name,
    "Phalange_Foot_2_2_L": BLM.PHALANGE_FOOT_2_2_LEFT.name,
    "Phalange_Foot_2_2_R": BLM.PHALANGE_FOOT_2_2_RIGHT.name,
    "Phalange_Foot_2_3_L": BLM.PHALANGE_FOOT_2_3_LEFT.name,
    "Phalange_Foot_2_3_R": BLM.PHALANGE_FOOT_2_3_RIGHT.name,
    "Phalange_Foot_3_1_L": BLM.PHALANGE_FOOT_3_1_LEFT.name,
    "Phalange_Foot_3_1_R": BLM.PHALANGE_FOOT_3_1_RIGHT.name,
    "Phalange_Foot_3_2_L": BLM.PHALANGE_FOOT_3_2_LEFT.name,
    "Phalange_Foot_3_2_R": BLM.PHALANGE_FOOT_3_2_RIGHT.name,
    "Phalange_Foot_3_3_L": BLM.PHALANGE_FOOT_3_3_LEFT.name,
    "Phalange_Foot_3_3_R": BLM.PHALANGE_FOOT_3_3_RIGHT.name,
    "Phalange_Foot_4_1_L": BLM.PHALANGE_FOOT_4_1_LEFT.name,
    "Phalange_Foot_4_1_R": BLM.PHALANGE_FOOT_4_1_RIGHT.name,
    "Phalange_Foot_4_2_L": BLM.PHALANGE_FOOT_4_2_LEFT.name,
    "Phalange_Foot_4_2_R": BLM.PHALANGE_FOOT_4_2_RIGHT.name,
    "Phalange_Foot_4_3_L": BLM.PHALANGE_FOOT_4_3_LEFT.name,
    "Phalange_Foot_4_3_R": BLM.PHALANGE_FOOT_4_3_RIGHT.name,
    "Phalange_Foot_5_1_L": BLM.PHALANGE_FOOT_5_1_LEFT.name,
    "Phalange_Foot_5_1_R": BLM.PHALANGE_FOOT_5_1_RIGHT.name,
    "Phalange_Foot_5_2_L": BLM.PHALANGE_FOOT_5_2_LEFT.name,
    "Phalange_Foot_5_2_R": BLM.PHALANGE_FOOT_5_2_RIGHT.name,
    "Phalange_Foot_5_3_L": BLM.PHALANGE_FOOT_5_3_LEFT.name,
    "Phalange_Foot_5_3_R": BLM.PHALANGE_FOOT_5_3_RIGHT.name,
}


class VSDReconstruction(BaseDatasetIO):
    """
    Data reader for VSD Full Body Reconstruction dataset.
    Expexted structure:
    root_directory/
    ├── imagesTr/
    │   ├── 001_0000.nii.gz
    │   ├── 002_0000.nii.gz
    │   ├── ...
    │   └── 030_0000.nii.gz
    ├── labelsTr/
    │   ├── 001.nii.gz
    │   ├── 002.nii.gz
    │   ├── ...
    │   └── 030.nii.gz
    ├── meshes/
    │   ├── 001/
    │   │   ├── Sacrum.stl
    │   │   ├── Hip_L.stl
    │   │   └── ...
    │   ├── 002/
    │   │   ├── Sacrum.stl
    │   │   ├── Hip_L.stl
    │   │   └── ...
    │   └── ...
    ├── metadata/
    │   ├── 001.json
    │   ├── 002.json
    │   ├── ...
    │   └── 030.json
    └── dataset.json
    """

    def __init__(self, dataset_root: Path):
        dataset_info = DatasetInfo(
            name="VSD Full Body Reconstruction",
            description="VSD Full Body Reconstruction dataset",
            url="https://doi.org/10.5281/zenodo.8302449; https://github.com/MCM-Fischer/VSDFullBodyBoneModels",
            modality="CT",
        )
        super().__init__(dataset_root, dataset_info)
        self.custom_data_handlers.read_dataset = read_dataset
        self.custom_data_handlers.export_image = export_image
        self.custom_data_handlers.export_segmentation = export_segmentation
        self.custom_data_handlers.export_mesh = export_mesh


def read_dataset(dataset_root: Path) -> list[DataSource]:
    subjects = [d for d in (dataset_root / "imagesTr").iterdir()][:MAX_SUBJECTS_FOR_TESTING]
    datalist = []

    for sub in subjects:
        # read metadata from json file to get subject info
        with open(dataset_root / "metadata" / (sub.name.replace("_0000.nii.gz", "") + ".json")) as f:
            metadata = json.load(f)
        if height := metadata["subjectSnapshot"]["heightInMeters"]:
            height = float(height) * 100.0
        if weight := metadata["subjectSnapshot"]["weightInKilograms"]:
            weight = float(weight)

        data = DataSource(
            img_path=sub,
            segmentation_path=dataset_root / "labelsTr" / sub.name.replace("_0000", ""),
            mesh_path=list((dataset_root / "meshes" / sub.name.replace("_0000.nii.gz", "")).glob("*.stl")),
            subject_info=SubjectInfo(
                source_subject_path=metadata["source_subject"],
                age=int(round(int(metadata["subjectSnapshot"]["ageInDays"]) / 365.25)),
                gender=metadata["subjectSnapshot"]["gender"]["name"],
                height=height,
                weight=weight,
                imaging_modality="CT",
                image=True,
            ),
        )
        datalist.append(data)

    return datalist


def export_image(data: DataSource, output_file_path: Path):
    shutil.copyfile(str(data.img_path), str(output_file_path))


def export_segmentation(data: DataSource, output_file_path: Path):
    export_nii_segmentation(data.segmentation_path, output_file_path, label_mapping)


def export_mesh(data: DataSource, output_dir_path: Path):
    for mesh_file in data.mesh_path:
        os.makedirs(output_dir_path, exist_ok=True)
        mesh_name = mesh_file.stem
        try:
            label_name = mesh_label_mapping[mesh_name]
            output_mesh_path = output_dir_path / f"{output_dir_path.name}_{label_name}.stl"
            shutil.copyfile(str(mesh_file), str(output_mesh_path))
        except KeyError:
            print(f"Warning: Mesh file '{mesh_file}' does not match any known label and will be skipped.")
