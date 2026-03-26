import os
import shutil
from pathlib import Path
import numpy as np
from monai.transforms import LoadImaged, SaveImaged, Compose
import pydicom
import pydicom_seg
import SimpleITK as sitk


def export_image_monai(input_image_path: Path, output_image_path: Path):
    """
    Converts original images to NIfTI format (nii.gz) and saves the result.
    input_image_path: Path to the original image file or folder (in case of DICOM series).
    output_image_path: Path to save the converted image file ending with `.nii.gz`.
    """
    temp_folder = output_image_path.parent / f"temp_{output_image_path.name}"
    os.makedirs(temp_folder, exist_ok=True)
    transform = Compose(
        [
            LoadImaged(
                keys=["image"],
                image_only=False,
                reader="ITKReader",
            ),
            SaveImaged(
                keys=["image"],
                output_dir=str(temp_folder),
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
    saved_file = temp_folder / (Path(input_image_path).name)
    if not str(saved_file).endswith(".nii.gz"):
        saved_file = saved_file.with_suffix(".nii.gz")
    shutil.copyfile(saved_file, output_image_path)
    shutil.rmtree(temp_folder, ignore_errors=True)


def export_nii_segmentation(
    input_label_paths: Path | list[Path],
    output_label_path: Path,
    label_mappings: dict | list[dict],
):
    """
    Converts one or more NIfTI segmentation files to BoneHub standardized labels and saves the result.
    input_label_paths: Path (or list of Paths) to the original label file(s) in NIfTI format.
    output_label_path: Path to save the combined label file ending with `.nii.gz`.
    label_mappings: Dictionary (or list of dictionaries) mapping original labels to BoneHub labels.
                    When multiple files are given, later files take priority over earlier ones in case of overlapping voxels.
    """
    if isinstance(input_label_paths, Path):
        input_label_paths = [input_label_paths]
    if isinstance(label_mappings, dict):
        label_mappings = [label_mappings]

    if len(input_label_paths) != len(label_mappings):
        raise ValueError("The number of input label paths must match the number of label mappings.")

    combined_array = None
    ref_image = None

    for input_label_path, label_mapping in zip(input_label_paths, label_mappings):
        image = sitk.ReadImage(str(input_label_path))

        if ref_image is None:
            ref_image = image
        elif image.GetSize() != ref_image.GetSize() or image.GetOrigin() != ref_image.GetOrigin():
            # Resample to match the reference image grid
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(ref_image)
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            resampler.SetDefaultPixelValue(0)
            image = resampler.Execute(image)

        array = sitk.GetArrayFromImage(image)

        if combined_array is None:
            combined_array = np.zeros(array.shape, dtype=np.uint16)

        mapped_array = np.zeros(array.shape, dtype=np.uint16)
        for orig_label, bonehub_label in label_mapping.items():
            mapped_array[array == orig_label] = bonehub_label

        # Non-zero voxels from this file overwrite the combined array
        combined_array[mapped_array != 0] = mapped_array[mapped_array != 0]

    combined_image = sitk.GetImageFromArray(combined_array)
    combined_image.CopyInformation(ref_image)
    sitk.WriteImage(combined_image, str(output_label_path))


def export_dicom_segmentation(
    input_image_path: Path,
    input_label_path: Path,
    output_label_path: Path,
    label_mapping: dict,
):
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
