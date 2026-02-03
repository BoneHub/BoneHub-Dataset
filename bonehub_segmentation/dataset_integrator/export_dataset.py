import os
import shutil
from pathlib import Path
import json
import torch
import numpy as np
from monai.transforms import LoadImaged, SaveImaged, Compose, MapLabelValued, Lambdad
import pydicom
import pydicom_seg
import SimpleITK as sitk

from . import SubjectData


def export_dataset_to_bonehub_format(dataset: list[SubjectData], export_path: str, label_mapping: dict = None) -> None:
    """
    Exports the given dataset to BoneHub standardized format.

    Args:
        dataset: The dataset to be exported.
        export_path (str): The path where the dataset will be exported.
        label_mapping (dict, optional): A dictionary mapping original labels to BoneHub standardized labels.
    """

    if not os.path.exists(export_path):
        os.makedirs(export_path)

    images_dir = os.path.join(export_path, "images")
    labels_dir = os.path.join(export_path, "org_seg")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    metadata = []
    for idx, item in enumerate(dataset, start=1):
        # show a live progress bar in the console
        print(f"Exporting subject {idx}/{len(dataset)} ...", end="\r")
        # Export image
        img_src = Path(item["image"])
        img_dst = Path(images_dir) / Path(f"img_{idx:06d}")
        if img_src.name.endswith("nii.gz"):
            shutil.copy(img_src, img_dst.with_suffix(".nii.gz"))
        else:
            export_org_images_to_nii(img_src, img_dst)

        # Export label
        label_src = item["label"]
        if label_src:
            if not label_mapping:
                raise ValueError("Label mapping is required to export labels.")
            label_dst = Path(labels_dir) / Path(f"org_seg_{idx:06d}")
            label_src = Path(label_src)
            if label_src.name.endswith("nii.gz") or label_src.name.endswith(".nii") or label_src.name.endswith(".nrrd"):
                export_org_nii_nrrd_labels_to_bonehub_labels(label_src, label_dst, label_mapping)
            elif label_src.name.endswith(".dcm") or label_src.name.endswith(".dicom"):
                export_org_dicom_labels_to_bonehub_labels(img_src, label_src, label_dst, label_mapping)
            else:
                raise ValueError(f"Unsupported label file format: {label_src.suffix}")

        # metadata
        metadata.append(
            {
                "id": f"{idx:06d}",
                "src_image": str(img_src),
                "src_org_label": str(label_src),
                "metadata": item["metadata"],
            }
        )

    # Save metadata
    metadata_path = Path(export_path) / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Dataset exported to {export_path}")


def export_org_images_to_nii(input_image_path: Path, output_image_path: Path):
    """
    Converts original images to NIfTI format (nii.gz) and saves the result.
    input_image_path: Path to the original image file.
    output_image_path: Path to save the converted image file.
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
    shutil.move(str(saved_file), str(output_image_path) + ".nii.gz")


def export_org_nii_nrrd_labels_to_bonehub_labels(input_label_path: Path, output_label_path: Path, label_mapping: dict):
    """
    Converts original labels to BoneHub standardized labels and saves the result.
    input_label_path: Path to the original label file.
    output_label_path: Path to save the converted label file.
    label_mapping: Dictionary mapping original labels to BoneHub labels.
    """
    # Create and apply transform pipeline
    transform = Compose(
        [
            LoadImaged(
                keys=["label"],
                image_only=False,
            ),
            Lambdad(
                keys=["label"],
                func=lambda x: torch.where(torch.isin(x, torch.tensor(list(label_mapping.keys()))), x, 0),
            ),
            MapLabelValued(
                keys=["label"],
                orig_labels=list(label_mapping.keys()),
                target_labels=list(label_mapping.values()),
            ),
            SaveImaged(
                keys=["label"],
                output_dir=str(output_label_path.parent),
                output_postfix="",
                output_ext=".nii.gz",
                resample=False,
                separate_folder=False,
                print_log=False,
                output_dtype=torch.uint8,
            ),
        ]
    )

    # Save and rename to desired output path
    transform({"label": str(input_label_path)})
    saved_file = Path(output_label_path.parent) / (Path(input_label_path).name)
    if not str(saved_file).endswith(".nii.gz"):
        saved_file = saved_file.with_suffix(".nii.gz")
    shutil.move(str(saved_file), str(output_label_path) + ".nii.gz")


def export_org_dicom_labels_to_bonehub_labels(
    input_image_path: Path, input_label_path: Path, output_label_path: Path, label_mapping: dict
):
    """
    Converts original DICOM labels to BoneHub standardized labels and saves the result.
    input_image_path: Path to the original DICOM image folder.
    input_label_path: Path to the original DICOM label file.
    output_label_path: Path to save the converted label file.
    label_mapping: Dictionary mapping original labels to BoneHub labels.
    """
    seg_dcm = pydicom.dcmread(input_label_path)
    seg_reader = pydicom_seg.MultiClassReader()
    seg_result = seg_reader.read(seg_dcm)
    seg_image = seg_result.image
    seg_array = sitk.GetArrayFromImage(seg_image)

    # Map original labels to BoneHub labels
    seg_array_mapped = np.zeros(shape=seg_array.shape, dtype=np.uint8)
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
