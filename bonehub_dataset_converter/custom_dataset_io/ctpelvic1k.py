"""
CTPelvic1K dataset.
Dataset link: https://doi.org/10.5281/zenodo.4588402

Remarks:
    - CT images are from seven sources, two internal and five external, internals are included in the dataset but the externals must be obtained from their sources
    - segmentation masks are available for all internal and external datasets
    - During the export process, the images from the external datasets are copied to the output directory.
    - includes 75 CTs with metal artifacts

external sources:
    - Beyond the Cranial Vault: https://doi.org/10.7303/syn3193805
    - KiTS 23: https://github.com/neheller/kits23
    - Medical Segmentation Decathlon: http://medicaldecathlon.com/
    - TCIA Colonography (ACRIN 6664): https://doi.org/10.7937/K9/TCIA.2015.NWTESAY1
"""

from pathlib import Path
import shutil

import json
from bonehub_data_schema import SubjectInfo, DatasetInfo, BoneLabelMap as BLM

from .. import BaseDatasetIO, DataSource
from ..utils import export_nii_segmentation, export_image_monai, get_dicom_subject_metadata

from . import MAX_SUBJECTS_FOR_TESTING

BTCV_DATASET_PATH = Path(
    "Z:/BoneHub/Public_Datasets/066 Multi-Atlas Labeling Beyond the Cranial Vault - Workshop and Challenge"
)  # Update this path to the actual location of the BTCV dataset
KIT23_DATASET_PATH = Path(
    "Z:/BoneHub/Public_Datasets/073 kits23"
)  # Update this path to the actual location of the KiTS23 dataset
MSD_DATASET_PATH = Path(
    "Z:/BoneHub/Public_Datasets/041 MSD/raw_data"
)  # Update this path to the actual location of the MSD dataset
ACRIN6664_DATASET_PATH = Path(
    "Z:/BoneHub/Public_Datasets/070 TCIA CT COLONOGRAPHY ACRIN 6664/CT COLONOGRAPHY"
)  # Update this path to the actual location of the TCIA Colonography dataset (ACRIN 6664 dataset)

label_mapping = {
    0: BLM.BACKGROUND.value,
    1: BLM.SACRUM.value,
    2: BLM.HIP_LEFT.value,
    3: BLM.HIP_RIGHT.value,
    4: BLM.VERTEBRAE_LUMBAR.value,
}


class CTPelvic1K(BaseDatasetIO):
    """Data reader for CTPelvic1K dataset.

    Expected structure:
    root_directory/
    ├── CTPelvic1K_dataset1_mask_mappingback (Multi-Atlas Abdomen)/
    │  ├── dataset1_img0001_mask_4label.nii.gz
    │  ├── dataset1_img0002_mask_4label.nii.gz
    │  └── ...
    ├── CTPelvic1K_dataset2_mask_mappingback (TCIA Colonog)/
    │  └── CTPelvic1K_dataset2_mask_mappingback/
    │      ├── dataset2_1.3.6.1.4.1.9328.50.4.0001_3_325_mask_4label.nii.gz
    │      ├── dataset2_1.3.6.1.4.1.9328.50.4.0002_4_325_mask_4label.nii.gz
    │      └── ...
    ├── CTPelvic1K_dataset3_mask_mappingback (MSD T10 colon)/
    │  └── CTPelvic1K_dataset3_mask_mappingback/
    │      ├── dataset3_colon_001_mask_4label.nii.gz
    │      ├── dataset3_colon_003_mask_4label.nii.gz
    │      └── ...
    ├── CTPelvic1K_dataset4_mask_mappingback (KiTS 19)/
    │  └── CTPelvic1K_dataset4_mask_mappingback/
    │      ├── dataset4_case_00014_mask_4label.nii.gz
    │      ├── dataset4_case_00016_mask_4label.nii.gz
    │      └── ...
    ├── CTPelvic1K_dataset5_mask_mappingback (Multi-Atlas  Cervix)/
    │  ├── dataset5_0507688_Image_mask_4label.nii
    │  ├── dataset5_0759564_Image_mask_4label.nii.gz
    │  └── ...
    ├── CTPelvic1K_dataset6_Anonymized_mask/
    │  └── ipcai2021_dataset6_Anonymized/
    │      ├── dataset6_CLINIC_0001_mask_4label.nii.gz
    │      ├── dataset6_CLINIC_0002_mask_4label.nii.gz
    │      └── ...
    ├── CTPelvic1K_dataset6_data (CLINIC)/
    │  └── CTPelvic1K_dataset6_data/
    │      ├── dataset6_CLINIC_0001_data.nii.gz
    │      ├── dataset6_CLINIC_0002_data.nii.gz
    │      └── ...
    ├── CTPelvic1K_dataset7_data (CLINIC-metal)/
    │  └── CTPelvic1K_dataset7_data/
    │      ├── dataset7_CLINIC_metal_0000_data.nii.gz
    │      ├── dataset7_CLINIC_metal_0001_data.nii.gz
    │      └── ...
    └── CTPelvic1K_dataset7_mask/
           ├── dataset7_CLINIC_metal_0000_mask_4label.nii.gz
           ├── dataset7_CLINIC_metal_0001_mask_4label.nii.gz
           └── ...
    """

    def __init__(self, dataset_root: Path):
        dataset_info = DatasetInfo(
            name="CTPelvic1K",
            description="CTPelvic1K dataset",
            url="https://doi.org/10.5281/zenodo.4588402",
            modality="CT",
        )
        super().__init__(dataset_root, dataset_info)
        self.custom_data_handlers.read_dataset = read_dataset
        self.custom_data_handlers.export_image = export_image
        self.custom_data_handlers.export_segmentation = export_segmentation


def read_dataset1(dataset_root: Path) -> list[DataSource]:
    datalist = []
    seg_files = list((dataset_root / "CTPelvic1K_dataset1_mask_mappingback (Multi-Atlas Abdomen)").glob("*.nii.gz"))
    img_files = list((BTCV_DATASET_PATH / "Abdomen" / "rawdata" / "RawData" / "Training" / "img").glob("*.nii.gz")) + list(
        (BTCV_DATASET_PATH / "Abdomen" / "rawdata" / "RawData" / "Testing" / "img").glob("*.nii.gz")
    )
    for seg_file in seg_files:
        sub_id = seg_file.name.split("_")[1]
        img_file = f"{sub_id}.nii.gz"
        img_file = next((f for f in img_files if f.name == img_file), None)
        if img_file is None:
            raise ValueError(f"Image file for subject {sub_id} not found in BTCV dataset")
        data = DataSource(
            img_path=img_file,
            segmentation_path=[seg_file],
            subject_info=SubjectInfo(
                source_subject_path=img_file.as_posix(),
                imaging_modality="CT",
                image=True,
            ),
        )
        datalist.append(data)
    return datalist


def read_dataset2(dataset_root: Path) -> list[DataSource]:
    datalist = []
    seg_files = list(
        (dataset_root / "CTPelvic1K_dataset2_mask_mappingback (TCIA Colonog)" / "CTPelvic1K_dataset2_mask_mappingback").glob(
            "*.nii.gz"
        )
    )
    for seg_file in seg_files:
        sub_id = seg_file.name.split("_")[1]
        sub_series = seg_file.name.split("_")[2]
        dicom_dir = None
        for subdir in (ACRIN6664_DATASET_PATH / sub_id).iterdir():
            for subsubdir in subdir.iterdir():
                if subsubdir.name.split(".")[0] == sub_series:
                    dicom_dir = subsubdir
                if dicom_dir:
                    break
            if dicom_dir:
                break
        if not dicom_dir:
            raise ValueError(f"DICOM directory for subject {sub_id} with series {sub_series} not found in ACRIN6664 dataset")

        metadata = get_dicom_subject_metadata(dicom_dir)
        data = DataSource(
            img_path=dicom_dir,
            segmentation_path=[seg_file],
            subject_info=SubjectInfo(
                source_subject_path=dicom_dir.as_posix(),
                age=metadata["age"],
                gender=metadata["gender"],
                imaging_modality="CT",
                image=True,
            ),
        )
        datalist.append(data)
    return datalist


def read_dataset3(dataset_root: Path) -> list[DataSource]:
    datalist = []
    seg_files = list(
        (dataset_root / "CTPelvic1K_dataset3_mask_mappingback (MSD T10 colon)" / "CTPelvic1K_dataset3_mask_mappingback").glob(
            "*.nii.gz"
        )
    )
    img_files = list((MSD_DATASET_PATH / "Task10_Colon" / "imagesTr").glob("*.nii.gz")) + list(
        (MSD_DATASET_PATH / "Task10_Colon" / "imagesTs").glob("*.nii.gz")
    )
    for seg_file in seg_files:
        sub_id = "_".join(seg_file.name.split("_")[1:3])
        img_file = f"{sub_id}.nii.gz"
        img_file = next((f for f in img_files if f.name == img_file), None)
        if img_file is None:
            raise ValueError(f"Image file for subject {sub_id} not found in MSD dataset")
        data = DataSource(
            img_path=img_file,
            segmentation_path=[seg_file],
            subject_info=SubjectInfo(
                source_subject_path=img_file.as_posix(),
                imaging_modality="CT",
                image=True,
            ),
        )
        datalist.append(data)
    return datalist


def read_dataset4(dataset_root: Path) -> list[DataSource]:
    datalist = []
    seg_files = list(
        (dataset_root / "CTPelvic1K_dataset4_mask_mappingback (KiTS 19)" / "CTPelvic1K_dataset4_mask_mappingback").glob(
            "*.nii.gz"
        )
    )

    with open(KIT23_DATASET_PATH / "kits23.json") as f:
        metadata = json.load(f)
    metadata = {case["case_id"]: case for case in metadata}

    for seg_file in seg_files:
        sub_id = "_".join(seg_file.name.split("_")[1:3])
        img_file = KIT23_DATASET_PATH / sub_id / "imaging.nii.gz"
        if not img_file.exists():
            raise ValueError(f"Image file for subject {sub_id} not found in KiTS23 dataset")

        data = DataSource(
            img_path=img_file,
            segmentation_path=[seg_file],
            subject_info=SubjectInfo(
                source_subject_path=img_file.as_posix(),
                # TODO: confirm if this is the correct age field to use
                # age=int(metadata[sub_id]["age_at_nephrectomy"]),
                gender=process_gender(metadata[sub_id]["gender"]),
                bmi=metadata[sub_id]["bmi"],
                imaging_modality="CT",
                image=True,
            ),
        )
        datalist.append(data)
    return datalist


def read_dataset5(dataset_root: Path) -> list[DataSource]:
    datalist = []
    seg_files = list((dataset_root / "CTPelvic1K_dataset5_mask_mappingback (Multi-Atlas  Cervix)").glob("*.nii.gz"))
    img_files = list((BTCV_DATASET_PATH / "Cervix" / "cervixrawdata" / "RawData" / "Training" / "img").glob("*.nii.gz")) + list(
        (BTCV_DATASET_PATH / "Cervix" / "cervixrawdata" / "RawData" / "Testing" / "img").glob("*.nii.gz")
    )

    for seg_file in seg_files:
        sub_id = seg_file.name.split("_")[1]
        img_file = f"{sub_id}-Image.nii.gz"
        img_file = next((f for f in img_files if f.name == img_file), None)
        if img_file is None:
            raise ValueError(f"Image file for subject {sub_id} not found in BTCV dataset")
        data = DataSource(
            img_path=img_file,
            segmentation_path=[seg_file],
            subject_info=SubjectInfo(
                source_subject_path=img_file.as_posix(),
                imaging_modality="CT",
                image=True,
            ),
        )
        datalist.append(data)
    return datalist


def read_dataset6(dataset_root: Path) -> list[DataSource]:
    datalist = []
    img_files = list((dataset_root / "CTPelvic1K_dataset6_data (CLINIC)" / "CTPelvic1K_dataset6_data").glob("*.nii.gz"))
    for img_file in img_files:
        seg_file = (
            dataset_root
            / "CTPelvic1K_dataset6_Anonymized_mask"
            / "ipcai2021_dataset6_Anonymized"
            / img_file.name.replace("_data", "_mask_4label")
        )
        data = DataSource(
            img_path=img_file,
            segmentation_path=[seg_file],
            subject_info=SubjectInfo(
                source_subject_path=img_file.name,
                imaging_modality="CT",
                image=True,
            ),
        )
        datalist.append(data)
    return datalist


def read_dataset7(dataset_root: Path) -> list[DataSource]:
    datalist = []
    img_files = list((dataset_root / "CTPelvic1K_dataset7_data (CLINIC-metal)" / "CTPelvic1K_dataset7_data").glob("*.nii.gz"))
    for img_file in img_files:
        seg_file = (
            dataset_root / "CTPelvic1K_dataset7_mask" / img_file.name.replace("dataset7_", "").replace("_data", "_mask_4label")
        )
        if not seg_file.exists():
            # print(f"Warning: Segmentation file {seg_file} not found for image {img_file}.")
            seg_file = None
        else:
            seg_file = [seg_file]
        data = DataSource(
            img_path=img_file,
            segmentation_path=seg_file,
            subject_info=SubjectInfo(
                source_subject_path=img_file.name,
                imaging_modality="CT",
                image=True,
                remarks="Metal artifact present",
            ),
        )
        datalist.append(data)
    return datalist


def read_dataset(dataset_root: Path) -> list[DataSource]:
    datalist = []
    datalist.extend(read_dataset1(dataset_root))
    datalist.extend(read_dataset2(dataset_root))
    datalist.extend(read_dataset3(dataset_root))
    datalist.extend(read_dataset4(dataset_root))
    datalist.extend(read_dataset5(dataset_root))
    datalist.extend(read_dataset6(dataset_root))
    datalist.extend(read_dataset7(dataset_root))
    if not datalist:
        raise ValueError(f"No valid subjects found in {dataset_root}")
    return datalist[:MAX_SUBJECTS_FOR_TESTING]


def export_image(data: DataSource, output_file_path: Path):
    if data.img_path.is_dir():
        # DICOM directory
        export_image_monai(data.img_path, output_file_path)
    else:
        # NIfTI file
        shutil.copyfile(data.img_path, output_file_path)


def export_segmentation(data: DataSource, output_file_path: Path):
    export_nii_segmentation(data.segmentation_path, output_file_path, [label_mapping])


def process_gender(gender_str: str) -> str:
    gender_str = gender_str.strip().upper()
    if not gender_str:
        return None
    if gender_str in {"M", "MALE"}:
        return "M"
    elif gender_str in {"F", "FEMALE"}:
        return "F"
    else:
        return "O"  # Other/Unknown
