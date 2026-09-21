<div align="center">

# ✨[**Browse BoneHub Dataset Here**](https://bonehub.github.io/BoneHub-Dataset)✨

### 👆 **Click above to explore** 👆

</div>

> 🚀 **What's New**
>
> [2026-??] TODO: Make data public <br>
> [2026-??] TODO: Generate mesh and NURBS models <br>
> [2026-04] Reaching 5K subjects  <br>
> [2026-03] Project started


# BoneHub Dataset

A Python package for generating the BoneHub Dataset, including data schema, dataset conversion, segmentation, mesh and NURBS generation.

## Install

The base install contains **only `bonehub_data_schema`** and its single dependency
(`pydantic`). Use it when all you need is to read or write BoneHub metadata — the
label map, `SubjectInfo`, `DatasetInfo` and `BoneHubDatasetIO`:

```bash
pip install -e .                        # from a local clone
pip install "bonehub-dataset @ git+https://github.com/BoneHub/BoneHub-Dataset.git"
```

The heavier tooling is opt-in, so nothing pulls in torch, monai or ITK unless asked:

```bash
pip install -e ".[io]"              # + reading/writing segmentation files (numpy, SimpleITK)
pip install -e ".[converter]"       # + dataset conversion (monai, pydicom, ITK, pandas, ...)
pip install -e ".[segmentation]"    # + segmentation models (torch, torchvision, monai, ...)
pip install -e ".[all]"             # everything
```

## Related repositories

| Repository | Purpose |
| --- | --- |
| [bonehub_dataset_quality_check_server](https://github.com/BoneHub/bonehub_dataset_quality_check_server) | Dockerised server that distributes subjects to reviewers and writes confirmed segmentations back into a dataset folder |
| [bonehub_dataset_quality_check_3dslicer_extension](https://github.com/BoneHub/bonehub_dataset_quality_check_3dslicer_extension) | 3D Slicer extension reviewers use to fetch, correct and submit segmentations |


## Dataset Structure

```
BoneHub Dataset/
├── Dataset_001/
│   ├── Dataset_info_001.json
│   ├── Subject_info_001.json
│   ├── Image/
│   │   ├── 001_000001.nii.gz
│   │   ├── 001_000002.nii.gz
│   │   └── ...
│   ├── Segmentation/
│   │   ├── 001_000001.seg.nrrd
│   │   ├── 001_000002.seg.nrrd
│   │   └── ...
│   ├── Mesh/
│   │   ├── 001_000001/
│   │   │   ├── 001_000001_FEMUR_LEFT.stl
│   │   │   ├── 001_000001_FEMUR_RIGHT.stl
│   │   │   └── ...
│   │   ├── 001_000002/
│   │   │   ├── 001_000002_FEMUR_LEFT.stl
│   │   │   ├── 001_000002_FEMUR_RIGHT.stl
│   │   │   └── ...
│   │   └── ...
│   ├── NURBS/
│   │   ├── 001_000001/
│   │   │   ├── 001_000001_FEMUR_LEFT.iges
│   │   │   ├── 001_000001_FEMUR_RIGHT.iges
│   │   │   └── ...
│   │   ├── 001_000002/
│   │   │   ├── 001_000002_FEMUR_LEFT.iges
│   │   │   ├── 001_000002_FEMUR_RIGHT.iges
│   │   │   └── ...
│   │   └── ...
│   └── Landmark/
│       ├── 001_000001.csv
│       ├── 001_000002.csv
│       └── ...
├── Dataset_002/
│   └── ...
└── ...
```

### Segmentation files

Masks are stored as `.seg.nrrd`, 3D Slicer's segmentation format. Voxels hold per-file
segment numbers (1, 2, 3 ...), and the header maps each number to its `BoneLabelMap` label
and records its colour and bounding box. Open them in 3D Slicer as a
**Segmentation**; in Python, `read_segmentation` returns the mask as `BoneLabelMap` values:

```python
import SimpleITK as sitk, numpy as np
from bonehub_data_schema import BoneLabelMap, read_segmentation  # needs the [io] extra

mask = read_segmentation("001_000001.seg.nrrd")  # SimpleITK image of BoneLabelMap values
arr = sitk.GetArrayViewFromImage(mask)
print([BoneLabelMap(v).name for v in np.unique(arr) if v])
```

The values are int32; do not cast them to float32, which cannot hold 9-digit values exactly.

The label value is built from four independent fields, so a bone, a sub-part, a tissue
compartment and a side each occupy their own digits:

```
value = structure * 100000 + part * 1000 + tissue * 10 + side

FEMUR_PROXIMAL_CORTICAL_LEFT = 710001011  =  7100 | 01 | 01 | 1
                                              |      |    |    +-- side       1 = left
                                              |      |    +------- tissue    01 = cortical
                                              |      +------------ part      01 = proximal
                                              +------------------- structure 7100 = femur
```

Every value has 9 digits (4 + 2 + 2 + 1), except `BACKGROUND = 0`.

Use `structure_of`, `part_of`, `tissue_of` and `side_of` from `bonehub_data_schema.labelmap`
to group labels — e.g. every left-femur label is `structure_of(v) == 7100 and side_of(v) == 1`.

`read_segmentation_labels` lists the labels in a mask from its header alone:

```python
from bonehub_data_schema import read_segmentation_labels  # needs the [io] extra
read_segmentation_labels("001_000001.seg.nrrd")
# ['FEMUR_LEFT', 'PHALANX_HAND_2_DISTAL_LEFT']
```

### Label status values

Every label in a subject's `segmentation`, `mesh` and `nurbs` entries carries a status:

| status | meaning |
|---|---|
| 0 | not available |
| 1 | available, not reviewed or corrected |
| 2 | available, reviewed and corrected (if necessary) |

A label missing from an entry is the same as status `0`. For example:

```json
{
    "segmentation": {"SKULL": 1, "VERTEBRA_C1": 2},
    "mesh": {"SKULL": 0, "VERTEBRA_C1": 1}
}
```

`subject.available_labels("segmentation")` returns the labels with status `1` or `2`.

## Custom Dataset Conversion Guide

### Use pre-made conversion scripts

We have prepared conversion scripts for various datasets.
Here is an example of converting TCIA Spine-Mets-CT-SEG dataset ([link](https://www.cancerimagingarchive.net/collection/spine-mets-ct-seg/)) into BoneHub's data-structure:

```python
from pathlib import Path
from bonehub_dataset_converter.custom_dataset_io import SpineMetsCTSeg

data_root = Path("path/to/dataset/root/folder")
output_root = Path("path/to/output/root/folder")

# The guard is required: subjects are converted in worker processes that re-import this script.
if __name__ == "__main__":
    dataset = SpineMetsCTSeg(dataset_root=data_root)
    dataset.export_to_bonehub_format(output_root, output_dataset_id=1, overwrite=False)
```

### Schema versions

Each `Dataset_info_XXX.json` records the `schema_version` it was written with.
`BoneHubDatasetIO` refuses a dataset whose major or minor version differs from the installed
schema, because its label values and label statuses may mean something else; regenerate it instead.

### Regenerating a dataset without re-converting its images

When only segmentations, meshes or subject info need regenerating (for example after a label
map change), keep the images that are already exported:

```python
dataset.export_to_bonehub_format(output_root, output_dataset_id=1, overwrite=True, skip_existing_images=True)
```

An image is reused only if the previous `Subject_info_XXX.json` shows it was made from the
same source subject; otherwise it is converted again. For all datasets at once:

```bash
python examples/dataset_conversion/convert_all_datasets_parallel.py --output-root <root> --skip-existing-images
```

## License
MIT
