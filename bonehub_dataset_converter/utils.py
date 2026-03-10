import os
import shutil
from pathlib import Path
import torch
import numpy as np
from monai.transforms import LoadImaged, SaveImaged, Compose, MapLabelValued, Lambdad
import pydicom
import pydicom_seg
import SimpleITK as sitk


def export_image(input_image_path: Path, output_image_path: Path):
    """
    Converts original images to NIfTI format (nii.gz) and saves the result.
    input_image_path: Path to the original image file or folder (in case of DICOM series).
    output_image_path: Path to save the converted image file ending with `.nii.gz`.
    """
    # Create and apply transform pipeline
    transform = Compose(
        [
            LoadImaged(
                keys=["image"],
                image_only=False,
            ),
            SaveImaged(
                keys=["image"],
                output_dir=str(output_image_path.parent),
                output_postfix="",
                output_ext=".nii.gz",
                resample=False,
                separate_folder=False,
                print_log=False,
            ),
        ]
    )

    # Save and rename to desired output path
    transform({"image": str(input_image_path)})
    saved_file = Path(output_image_path.parent) / (Path(input_image_path).name)
    if not str(saved_file).endswith(".nii.gz"):
        saved_file = saved_file.with_suffix(".nii.gz")
    shutil.move(saved_file, output_image_path)


def export_nii_nrrd_segmentation(input_label_path: Path, output_label_path: Path, label_mapping: dict):
    """
    Converts original labels to BoneHub standardized labels and saves the result.
    input_label_path: Path to the original label file in NIfTI or NRRD format.
    output_label_path: Path to save the converted label file ending with `.nii.gz`.
    label_mapping: Dictionary mapping original labels to BoneHub labels.
    """
    # Create and apply transform pipeline
    transform = Compose(
        [
            LoadImaged(
                keys=["label"],
                image_only=False,
                reader="ITKReader",
            ),
            Lambdad(
                keys=["label"],
                func=lambda x: torch.where(torch.isin(x, torch.tensor(list(label_mapping.keys()), dtype=torch.int)), x, 0),
            ),
            MapLabelValued(
                keys=["label"],
                orig_labels=list(label_mapping.keys()),
                target_labels=list(label_mapping.values()),
                dtype=torch.int,
            ),
            SaveImaged(
                keys=["label"],
                output_dir=str(output_label_path.parent),
                output_postfix="",
                output_ext=".nii.gz",
                resample=False,
                separate_folder=False,
                print_log=False,
                output_dtype=torch.uint16,
            ),
        ]
    )

    # Save and rename to desired output path
    transform({"label": str(input_label_path)})
    saved_file = Path(output_label_path.parent) / (Path(input_label_path).name)
    if not str(saved_file).endswith(".nii.gz"):
        saved_file = saved_file.with_suffix(".nii.gz")
    shutil.move(saved_file, output_label_path)


def export_dicom_segmentation(input_image_path: Path, input_label_path: Path, output_label_path: Path, label_mapping: dict):
    """
    Converts original DICOM labels to BoneHub standardized labels and saves the result.
    input_image_path: Path to the original DICOM image folder.
    input_label_path: Path to the original DICOM label file.
    output_label_path: Path to save the converted label file ending with `.nii.gz`.
    label_mapping: Dictionary mapping original labels to BoneHub labels.
    """
    seg_dcm = pydicom.dcmread(input_label_path)
    seg_reader = pydicom_seg.MultiClassReader()
    seg_result = seg_reader.read(seg_dcm)
    seg_image = seg_result.image
    seg_array = sitk.GetArrayFromImage(seg_image)

    # Map original labels to BoneHub labels
    seg_array_mapped = np.zeros(shape=seg_array.shape, dtype=np.uint16)
    for orig_label in seg_result.segment_infos.keys():
        bonehub_label = label_mapping[seg_result.segment_infos[orig_label].SegmentLabel]
        seg_array_mapped[seg_array == orig_label] = bonehub_label

    seg_image_mapped = sitk.GetImageFromArray(seg_array_mapped)
    seg_image_mapped.CopyInformation(seg_image)

    # Read the reference image series with SimpleITK (preserves LPS orientation)
    series_reader = sitk.ImageSeriesReader()
    series_files = series_reader.GetGDCMSeriesFileNames(input_image_path)
    series_reader.SetFileNames(series_files)
    ref_image = series_reader.Execute()

    # Resample segmentation directly in physical space to match the reference image
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref_image)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    seg_resampled = resampler.Execute(seg_image_mapped)

    # Save as NIfTI
    sitk.WriteImage(seg_resampled, str(output_label_path) + ".nii.gz")


def get_dicom_subject_metadata(dicom_folder: str) -> dict:
    first_file = next(f for f in os.listdir(dicom_folder) if f.endswith(".dcm") or f.endswith(".dicom"))
    ds = pydicom.dcmread(os.path.join(dicom_folder, first_file), stop_before_pixels=True)
    age = getattr(ds, "PatientAge", None)
    if age:
        age = "".join(filter(str.isdigit, age))
        age = int(age)

    gender = getattr(ds, "PatientSex", None)
    modality = getattr(ds, "Modality", None)

    return {
        "age": age,
        "gender": gender,
        "modality": modality,
    }
