"""
VSD Full Body Reconstruction Dataset.
url: https://doi.org/10.5281/zenodo.8302449; https://github.com/MCM-Fischer/VSDFullBodyBoneModels
"""

import json
from pathlib import Path
import shutil
import SimpleITK as sitk
import numpy as np
from bonehub_data_schema import SubjectInfo, DatasetInfo, BoneLabelMap as BLM
import pandas as pd

from .. import BaseDatasetIO, DataSource
from ..utils import export_nii_nrrd_segmentation

from . import MAX_SUBJECTS_FOR_TESTING

# subjects_metadata = [
#     ["002", 78, "F", 75.0, 162.0],
#     ["006", 51, "F", 90.0, 177.0],
#     ["010", 45, "F", 54.0, 165.0],
#     ["014", 30, "F", 65.0, 165.0],
#     ["015", 81, "M", 78.0, 175.0],
#     ["016", 95, "F", 60.0, 152.0],
#     ["017", 19, "F", 59.0, 170.0],
#     ["019", 56, "M", 68.0, 170.0],
#     ["023", 74, "M", 86.0, 182.0],
#     ["z001", 76, "M", 87.0, 180.0],
#     ["z004", 65, "M", 82.3, 177.0],
#     ["z009", 25, "M", 74.0, 175.0],
#     ["z019", 58, "M", 71.3, 181.0],
#     ["z023", 47, "F", 61.0, 166.0],
#     ["z027", 37, "F", 51.5, 169.0],
#     ["z035", 30, "F", 50.45, 168.0],
#     ["z042", 61, "F", 53.4, 169.0],
#     ["z046", 38, "M", 72.0, 180.0],
#     ["z056", 26, "M", 81.8, 187.0],
#     ["z062", 43, "M", 76.95, 177.0],
#     ["z013", 41, "F", 56.3, 165.0],
#     ["z036", 62, "M", None, None],
#     ["z049", 34, "M", 87.0, 179.0],
#     ["z050", 84, "M", 73.4, 167.0],
#     ["z055", 73, "M", 73.0, 173.0],
#     ["z057", 75, "M", None, None],
#     ["z061", 39, "F", 37.4, 180.0],
#     ["z063", 72, "F", 80.2, 172.0],
#     ["z064", 69, "M", None, None],
#     ["z066", 48, "M", None, None],
# ]

label_mapping = {
    "Sacrum": BLM.SACRUM.value,
    "Hip_R": BLM.HIP_RIGHT.value,
    "Hip_L": BLM.HIP_LEFT.value,
    "Femur_R": BLM.FEMUR_RIGHT.value,
    "Femur_L": BLM.FEMUR_LEFT.value,
    "Tibia_R": BLM.TIBIA_RIGHT.value,
    "Tibia_L": BLM.TIBIA_LEFT.value,
    "Patella_R": BLM.PATELLA_RIGHT.value,
    "Patella_L": BLM.PATELLA_LEFT.value,
    "Fibula_R": BLM.FIBULA_RIGHT.value,
    "Fibula_L": BLM.FIBULA_LEFT.value,
    "Talus_R": BLM.TALUS_RIGHT.value,
    "Talus_L": BLM.TALUS_LEFT.value,
    "Calcaneus_R": BLM.CALCANEUS_RIGHT.value,
    "Calcaneus_L": BLM.CALCANEUS_LEFT.value,
    "Tarsals_R": BLM.TARSALS_RIGHT.value,
    "Tarsals_L": BLM.TARSALS_LEFT.value,
    "Metatarsals_R": BLM.METATARSALS_RIGHT.value,
    "Metatarsals_L": BLM.METATARSALS_LEFT.value,
    "Phalanges_R": BLM.PHALANGES_FOOT_RIGHT.value,
    "Phalanges_L": BLM.PHALANGES_FOOT_LEFT.value,
}


class VSDReconstruction(BaseDatasetIO):
    """Data reader for VSD Full Body Reconstruction dataset."""

    def __init__(self, dataset_root: Path):
        dataset_info = DatasetInfo(
            name="VSD Full Body Reconstruction",
            description="VSD Full Body Reconstruction dataset",
            url="https://doi.org/10.5281/zenodo.8302449; https://github.com/MCM-Fischer/VSDFullBodyBoneModels",
            modality="CT",
        )
        super().__init__(dataset_root, dataset_info)
        self.register_data_handler("read_dataset", read_dataset)
        self.register_data_handler("export_image", export_image)
        self.register_data_handler("export_segmentation", export_segmentation)
        self.register_data_handler("export_mesh", export_mesh)


def read_dataset(dataset_root: Path) -> list[DataSource]:
    subject_dirs = [d for d in (dataset_root / "raw_data (zenodo)").iterdir() if d.is_dir()][:MAX_SUBJECTS_FOR_TESTING]
    datalist = []

    for subject_dir in subject_dirs:
        # Find folders containing "smir" in their names
        relevant_dirs = [d for d in subject_dir.iterdir() if d.is_dir() and ("smir" in d.name.lower())]
        for rdir in relevant_dirs:
            # first find how many images and segmentations are in the folder
            volume_paths = list(rdir.glob("*.nii")) + list(rdir.glob("*.nrrd"))
            img_paths = []
            seg_paths = []
            for volpath in volume_paths:
                if "iliac" in volpath.name.lower():
                    print(
                        f"Warning: Found image file that contains 'iliac' in its name: {volpath.name}. This file is likely a partial scan of the pelvis region and may not be suitable for export. We will skip this file."
                    )
                    continue
                if volpath.name.lower().endswith("_reconstruction.seg.nrrd"):
                    if "pelvis-thighs" in volpath.name.lower() or "shanks-feet" in volpath.name.lower():
                        seg_paths.append(volpath)
                    else:
                        print(
                            f"Warning: Found segmentation file that does not match expected naming convention: {volpath.name}. We will skip this file."
                        )
                elif "seg.nrrd" not in volpath.name.lower():
                    img_paths.append(volpath)
            del volume_paths

            # read metadata from json file to get subject info
            with open(rdir / (rdir.name + ".json")) as f:
                metadata = json.load(f)
            if height := metadata["subjectSnapshot"]["heightInMeters"]:
                height = float(height) * 100.0
            if weight := metadata["subjectSnapshot"]["weightInKilograms"]:
                weight = float(weight)

            # check if upside-down transform is available for this subject
            transform_path = rdir / "Transform-Upside-Down.h5"
            transform = None
            if transform_path.exists():
                transform = sitk.ReadTransform(str(transform_path))

            # check if number of image files is always equal or more than number of segmentation files
            assert len(img_paths) >= len(
                seg_paths
            ), f"Expected at least as many image files as segmentation files in {rdir}, but found {len(img_paths)} img files and {len(seg_paths)} seg files."

            # create a DataSource for each image file, associating the same segmentation files to each of them (if any)
            for impath in img_paths:
                segpath = None
                for _segpath in seg_paths:
                    if _segpath.name.replace("_Reconstruction.seg.nrrd", "") == impath.stem:
                        segpath = _segpath
                        break
                if not segpath and len(img_paths) and len(seg_paths) == 1:
                    segpath = seg_paths[0]

                data = DataSource(
                    img_path=impath,
                    img_transform=transform,
                    segmentation_path=[segpath] if segpath else None,
                    # mesh_path=[str(f) for f in (rdir / "meshes").glob("*.stl")],
                    subject_info=SubjectInfo(
                        source_subject_path=str(rdir.relative_to(dataset_root)),
                        age=int(round(int(metadata["subjectSnapshot"]["ageInDays"]) / 365.25)),
                        gender=metadata["subjectSnapshot"]["gender"]["name"],
                        height=height,
                        weight=weight,
                    ),
                )
                datalist.append(data)

    return datalist


def export_image(data: DataSource, output_file_path: Path):
    # return
    image = sitk.ReadImage(str(data.img_path))
    sitk.WriteImage(image, str(output_file_path))


def export_segmentation(data: DataSource, output_file_path: Path):
    if not data.segmentation_path:
        return

    merged_seg_array = None
    valid_bonehub_labels = set(label_mapping.values())
    reference_seg_image = None

    for seg_file in data.segmentation_path:
        seg_file_path = Path(seg_file)
        if not seg_file_path.exists():
            print(f"Warning: Segmentation file not found: {seg_file_path}")
            continue

        seg_image = sitk.ReadImage(str(seg_file_path))
        seg_array = sitk.GetArrayFromImage(seg_image)

        if merged_seg_array is None:
            target_shape = seg_array.shape[-3:] if seg_array.ndim == 4 else seg_array.shape
            merged_seg_array = np.zeros(target_shape, dtype=np.uint16)
            reference_seg_image = seg_image
        else:
            current_shape = seg_array.shape[-3:] if seg_array.ndim == 4 else seg_array.shape
            if current_shape != merged_seg_array.shape:
                print(
                    f"Warning: Segmentation shape mismatch in {seg_file_path}. Expected {merged_seg_array.shape}, got {current_shape}. Skipping this file."
                )
                continue

            current_spacing = seg_image.GetSpacing()
            current_origin = seg_image.GetOrigin()
            current_direction = seg_image.GetDirection()
            if (
                current_spacing != reference_seg_image.GetSpacing()
                or current_origin != reference_seg_image.GetOrigin()
                or current_direction != reference_seg_image.GetDirection()
            ):
                print(
                    f"Warning: Geometry mismatch in {seg_file_path}. Using geometry from the first segmentation file for output."
                )

        if seg_array.ndim == 4:
            segment_meta = {}
            for key in seg_image.GetMetaDataKeys():
                if not key.startswith("Segment") or "_" not in key:
                    continue
                head, tail = key.split("_", 1)
                try:
                    seg_idx = int(head.replace("Segment", ""))
                except ValueError:
                    continue
                if seg_idx not in segment_meta:
                    segment_meta[seg_idx] = {}
                segment_meta[seg_idx][tail] = seg_image.GetMetaData(key)

            for seg_idx, meta in segment_meta.items():
                seg_name = meta.get("Name")
                if seg_name not in label_mapping:
                    continue

                try:
                    layer = int(meta.get("Layer", 0))
                    label_value = int(meta.get("LabelValue", 1))
                except ValueError:
                    continue

                if layer < 0 or layer >= seg_array.shape[0]:
                    continue

                mask = seg_array[layer] == label_value
                if np.any(mask):
                    merged_seg_array[mask] = label_mapping[seg_name]

        elif seg_array.ndim == 3:
            segment_meta = {}
            for key in seg_image.GetMetaDataKeys():
                if not key.startswith("Segment") or "_" not in key:
                    continue
                head, tail = key.split("_", 1)
                try:
                    seg_idx = int(head.replace("Segment", ""))
                except ValueError:
                    continue
                if seg_idx not in segment_meta:
                    segment_meta[seg_idx] = {}
                segment_meta[seg_idx][tail] = seg_image.GetMetaData(key)

            if segment_meta:
                for meta in segment_meta.values():
                    seg_name = meta.get("Name")
                    if seg_name not in label_mapping:
                        continue
                    try:
                        label_value = int(meta.get("LabelValue", 1))
                    except ValueError:
                        continue
                    mask = seg_array == label_value
                    if np.any(mask):
                        merged_seg_array[mask] = label_mapping[seg_name]
            else:
                for orig_label in np.unique(seg_array):
                    if orig_label == 0:
                        continue
                    if int(orig_label) in valid_bonehub_labels:
                        merged_seg_array[seg_array == orig_label] = int(orig_label)
        else:
            print(f"Warning: Unsupported segmentation dimensionality ({seg_array.ndim}) in {seg_file_path}")

    if merged_seg_array is None:
        return

    output_image = sitk.GetImageFromArray(merged_seg_array)
    output_image = sitk.Cast(output_image, sitk.sitkUInt16)
    if reference_seg_image is not None:
        output_image.CopyInformation(reference_seg_image)

    output_file_path = Path(output_file_path)
    if output_file_path.suffixes[-2:] != [".nii", ".gz"]:
        output_file_path = output_file_path.with_suffix("") if output_file_path.suffix else output_file_path
        output_file_path = output_file_path.with_suffix(".nii.gz")

    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(output_image, str(output_file_path))


def export_segmentation_old(data: DataSource, output_file_path: Path):
    # return
    """Export and merge multiple segmentation files to match CT geometry."""
    # Read reference CT image for geometry
    ref_image = sitk.ReadImage(str(data.img_path))
    ref_size = ref_image.GetSize()
    ref_spacing = ref_image.GetSpacing()
    ref_origin = ref_image.GetOrigin()
    ref_direction = ref_image.GetDirection()

    # Create empty merged segmentation with reference geometry
    merged_seg_array = np.zeros(ref_size[::-1], dtype=np.uint16)  # sitk uses xyz, numpy uses zyx

    print(f"Reference image size: {ref_size}, spacing: {ref_spacing}")
    print(f"Processing {len(data.segmentation_path)} segmentation files")

    # Process each segmentation file
    for seg_file in data.segmentation_path:
        seg_file_path = Path(seg_file)
        if not seg_file_path.exists():
            print(f"Warning: Segmentation file not found: {seg_file_path}")
            continue

        print(f"\nProcessing: {seg_file_path.name}")

        # Read segmentation file
        seg_image = sitk.ReadImage(str(seg_file_path))

        # Convert to numpy array first to check dimensions
        seg_array = sitk.GetArrayFromImage(seg_image)
        print(f"  Segmentation array shape: {seg_array.shape}, dtype: {seg_array.dtype}")
        print(f"  Unique values in array: {np.unique(seg_array)}")

        # Get metadata to find segment names
        seg_metadata = {key: seg_image.GetMetaData(key) for key in seg_image.GetMetaDataKeys()}

        # Parse segment information from metadata
        segment_info = {}
        for key, value in seg_metadata.items():
            if key.startswith("Segment") and "_Name" in key:
                seg_idx = int(key.split("_")[0].replace("Segment", ""))
                segment_info[seg_idx] = value

        print(f"  Found {len(segment_info)} segments: {segment_info}")

        # Handle 4D segmentation (segments × z × y × x)
        if seg_array.ndim == 4:
            num_segments = seg_array.shape[0]
            print(f"  Processing 4D segmentation with {num_segments} layers")

            for seg_idx in range(num_segments):
                # Get the specific segment layer
                segment_layer = seg_array[seg_idx]

                # Check if this segment has any data
                if segment_layer.max() == 0:
                    continue

                # Get segment name from metadata (index might be 0-based or 1-based)
                seg_name = segment_info.get(seg_idx) or segment_info.get(seg_idx + 1)

                if seg_name:
                    print(f"    Segment {seg_idx}: '{seg_name}' (max value: {segment_layer.max()})")

                    # Map segment name to BoneHub label
                    if seg_name in label_mapping:
                        bonehub_label = label_mapping[seg_name]
                        print(f"      Mapped to BoneHub label: {bonehub_label}")

                        # Create temporary 3D image from the segment layer
                        temp_seg = sitk.GetImageFromArray(segment_layer.astype(np.uint8))

                        # Set 3D geometry (extract from 4D by removing the first dimension info)
                        # For Slicer seg.nrrd files, spacing/origin/direction are for the 3D volume
                        temp_seg.SetSpacing(ref_spacing)
                        temp_seg.SetOrigin(ref_origin)
                        temp_seg.SetDirection(ref_direction)

                        # Resample to match reference image geometry
                        resampler = sitk.ResampleImageFilter()
                        resampler.SetReferenceImage(ref_image)
                        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
                        resampler.SetDefaultPixelValue(0)
                        resampled_seg = resampler.Execute(temp_seg)

                        # Get resampled array and merge
                        resampled_array = sitk.GetArrayFromImage(resampled_seg)
                        mask = resampled_array > 0
                        voxel_count = np.sum(mask)
                        print(f"      Added {voxel_count} voxels")
                        merged_seg_array[mask] = bonehub_label
                    else:
                        print(f"      Warning: '{seg_name}' not in label_mapping")
                        print(f"      Available mappings: {list(label_mapping.keys())}")

        # Handle 3D segmentation
        elif seg_array.ndim == 3:
            print(f"  Processing 3D segmentation")
            # Resample to match reference image
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(ref_image)
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            resampler.SetDefaultPixelValue(0)
            resampled_seg = resampler.Execute(seg_image)

            # Get resampled array
            resampled_array = sitk.GetArrayFromImage(resampled_seg)

            # Apply label mapping to unique values
            unique_labels = np.unique(resampled_array)
            for orig_label in unique_labels:
                if orig_label == 0:
                    continue
                mask = resampled_array == orig_label
                if orig_label in label_mapping.values():
                    merged_seg_array[mask] = orig_label

    # Create output image
    output_image = sitk.GetImageFromArray(merged_seg_array)
    output_image.SetSpacing(ref_spacing)
    output_image.SetOrigin(ref_origin)
    output_image.SetDirection(ref_direction)

    # Ensure output directory exists
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to file
    print(f"\nFinal merged segmentation stats:")
    print(f"  Shape: {merged_seg_array.shape}")
    print(f"  Unique labels: {np.unique(merged_seg_array)}")
    print(f"  Total non-zero voxels: {np.sum(merged_seg_array > 0)}")
    sitk.WriteImage(output_image, str(output_file_path))
    print(f"  Saved to: {output_file_path}")


def export_mesh(data: DataSource, output_dir_path: Path):
    pass


def transform_image(nrrd_file: Path, output_path: Path):
    transform = sitk.ReadTransform(Path(nrrd_file).parent / "Transform-Upside-Down.h5")
    # Apply transform: need to compute new output geometry that will contain the transformed image
    # For resampling, we need the inverse transform (output->input mapping)
    # But we also need to define an output space that will contain the transformed image

    # Get original image parameters
    original_size = ct_img.GetSize()
    original_spacing = ct_img.GetSpacing()
    original_direction = ct_img.GetDirection()

    # Compute corners of the original image in physical space
    corners = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                point = ct_img.TransformIndexToPhysicalPoint(
                    (int(i * (original_size[0] - 1)), int(j * (original_size[1] - 1)), int(k * (original_size[2] - 1)))
                )
                # Transform the corner point to the new space
                transformed_point = transform.TransformPoint(point)
                corners.append(transformed_point)

    # Find bounding box of transformed corners
    corners_array = np.array(corners)
    min_coords = corners_array.min(axis=0)
    max_coords = corners_array.max(axis=0)

    # Compute new size (keep same spacing)
    new_size = [int(np.ceil((max_coords[i] - min_coords[i]) / original_spacing[i])) + 1 for i in range(3)]

    # Set new origin to min corner
    new_origin = tuple(min_coords)

    # Create output image with new geometry
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(original_spacing)
    resampler.SetOutputOrigin(new_origin)
    resampler.SetOutputDirection(original_direction)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(transform.GetInverse())

    ct_img = resampler.Execute(ct_img)
