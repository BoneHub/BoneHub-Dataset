import os
import shutil
from pathlib import Path
import numpy as np
from monai.transforms import LoadImaged, SaveImaged, Compose
import pydicom
import pydicom_seg
import SimpleITK as sitk

from bonehub_data_schema import LabelStatus, Origin, segment_number_dtype, write_indexed_segmentation

FROM_SOURCE = LabelStatus.of(Origin.SOURCE)


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
    shutil.move(saved_file, output_image_path)
    shutil.rmtree(temp_folder, ignore_errors=True)


def export_nii_segmentation(
    input_label_paths: list[Path],
    output_label_path: Path,
    label_mappings: list[dict],
    status: LabelStatus | dict[int, LabelStatus] = FROM_SOURCE,
):
    """
    Converts one or more NIfTI segmentation files to BoneHub standardized labels and saves the result.
    input_label_paths: list of Paths to the original label file(s) in NIfTI format.
    output_label_path: Path to save the combined label file; written as `.seg.nrrd`.
    label_mappings: list of dictionaries mapping original labels to BoneHub labels.
                    When multiple files are given, later files take priority over earlier ones in case of overlapping voxels.
    status: how these segmentations were produced and reviewed, recorded per segment in the file header.
    """
    if len(input_label_paths) != len(label_mappings):
        raise ValueError("The number of input label paths must match the number of label mappings.")

    ref_image = None
    # (image, label array, mapping) on the reference grid; the image is kept alive because
    # the array is a view of its memory.
    sources = []
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
        sources.append((image, _label_array(image, input_label_path), label_mapping))

    # Number the BoneHub labels present 1..N and map each source label straight to its number.
    present = set()
    for _, array, label_mapping in sources:
        present.update(label_mapping[v] for v in _unique_labels(array) if label_mapping.get(v))
    bonehub_values = sorted(present)
    number_of = {value: n for n, value in enumerate(bonehub_values, start=1)}
    dtype = segment_number_dtype(len(bonehub_values))

    numbers = None
    for _, array, label_mapping in sources:
        lookup = np.zeros(max(int(array.max()), max(label_mapping)) + 1, dtype=dtype)
        for orig_label, bonehub_label in label_mapping.items():
            if bonehub_label in number_of:
                lookup[orig_label] = number_of[bonehub_label]
        mapped = lookup[array]
        # Non-zero voxels from later files overwrite earlier ones
        if numbers is None:
            numbers = mapped
        else:
            np.copyto(numbers, mapped, where=mapped != 0)
        del mapped

    return write_indexed_segmentation(numbers, bonehub_values, ref_image, output_label_path, status)


def _label_array(image: sitk.Image, path: Path) -> np.ndarray:
    """The image's labels as a non-negative integer array (a view when already integer)."""
    array = sitk.GetArrayViewFromImage(image)
    if not np.issubdtype(array.dtype, np.integer):  # some sources store masks as floats
        array = np.rint(array).astype(np.int32)
    if array.min() < 0:
        raise ValueError(f"Negative label values in '{path}'.")
    return array


def _unique_labels(array: np.ndarray, slices: int = 32) -> set[int]:
    """Distinct values in a large array, computed a block of slices at a time."""
    return {int(v) for z in range(0, array.shape[0], slices) for v in np.unique(array[z : z + slices])}


def export_dicom_segmentation(
    input_image_path: Path,
    input_label_path: Path,
    output_label_path: Path,
    label_mapping: dict,
    dicom_segment_key: str = "SegmentLabel",
    status: LabelStatus | dict[int, LabelStatus] = FROM_SOURCE,
):
    """
    Converts original DICOM labels to BoneHub standardized labels and saves the result.
    input_image_path: Path to the original DICOM image folder.
    input_label_path: Path to the original DICOM label file.
    output_label_path: Path to save the converted label file; written as `.seg.nrrd`.
    label_mapping: Dictionary mapping original labels to BoneHub labels.
    dicom_segment_key: Key to access the segment label in the DICOM segmentation file. Options: "SegmentLabel" (default) or "SegmentDescription", depending on how the original labels are stored in the DICOM file.
    status: how these segmentations were produced and reviewed, recorded per segment in the file header.
    """
    seg_dcm = pydicom.dcmread(input_label_path)
    seg_reader = pydicom_seg.MultiClassReader()
    seg_result = seg_reader.read(seg_dcm)
    seg_image = seg_result.image
    seg_array = _label_array(seg_image, input_label_path)

    # Map each DICOM segment to a BoneHub segment number 1..N
    bonehub_of = {}
    for orig_label, info in seg_result.segment_infos.items():
        seg_label_name = getattr(info, dicom_segment_key, None)
        if seg_label_name is None:
            raise ValueError(f"Segment info for label {orig_label} does not contain key '{dicom_segment_key}'.")
        bonehub_of[orig_label] = label_mapping[seg_label_name]
    bonehub_values = sorted(set(bonehub_of.values()))
    number_of = {value: n for n, value in enumerate(bonehub_values, start=1)}
    lookup = np.zeros(max(int(seg_array.max()), max(bonehub_of)) + 1, dtype=segment_number_dtype(len(bonehub_values)))
    for orig_label, bonehub_label in bonehub_of.items():
        lookup[orig_label] = number_of[bonehub_label]

    seg_image_mapped = sitk.GetImageFromArray(lookup[seg_array])
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

    return write_indexed_segmentation(
        sitk.GetArrayViewFromImage(seg_resampled), bonehub_values, seg_resampled, output_label_path, status
    )


def get_dicom_subject_metadata(dicom_folder: str) -> dict:
    first_file = next(f for f in os.listdir(dicom_folder) if f.endswith(".dcm") or f.endswith(".dicom"))
    ds = pydicom.dcmread(os.path.join(dicom_folder, first_file), stop_before_pixels=True)
    age = getattr(ds, "PatientAge", None)
    if age:
        age = "".join(filter(str.isdigit, age))
        age = int(age)
    else:
        age = None
    gender = getattr(ds, "PatientSex", None)
    if gender:
        gender = gender.upper()
        if gender not in ["M", "F", "O"]:
            print(
                f"Warning: Unrecognized gender value '{gender}' in DICOM data folder '{dicom_folder}'. Setting gender to None."
            )
            gender = None

    modality = getattr(ds, "Modality", None)

    return {
        "age": age,
        "gender": gender,
        "modality": modality,
    }
