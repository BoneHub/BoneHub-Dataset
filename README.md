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
│   │   ├── 001_000001.nii.gz
│   │   ├── 001_000002.nii.gz
│   │   └── ...
│   ├── Mesh/
│   │   ├── 001_000001/
│   │   │   ├── 001_000001_femur_left.stl
│   │   │   ├── 001_000001_femur_right.stl
│   │   │   └── ...
│   │   ├── 001_000002/
│   │   │   ├── 001_000002_femur_left.stl
│   │   │   ├── 001_000002_femur_right.stl
│   │   │   └── ...
│   │   └── ...
│   ├── NURBS/
│   │   ├── 001_000001/
│   │   │   ├── 001_000001_femur_left.iges
│   │   │   ├── 001_000001_femur_right.iges
│   │   │   └── ...
│   │   ├── 001_000002/
│   │   │   ├── 001_000002_femur_left.iges
│   │   │   ├── 001_000002_femur_right.iges
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

## Custom Dataset Conversion Guide

### Use pre-made conversion scripts

We have prepared conversion scripts for various datasets.
Here is an example of converting TCIA Spine-Mets-CT-SEG dataset ([link](https://www.cancerimagingarchive.net/collection/spine-mets-ct-seg/)) into BoneHub's data-structure:

```python
from bonehub_dataset_converter.custom_dataset_io import SpineMetsCTSeg

data_root = Path("path/to/dataset/root/folder")
output_root = Path("path/to/output/root/folder")
dataset = SpineMetsCTSeg(dataset_root=data_root)
dataset.export_to_bonehub_format(output_root, output_dataset_id=1, overwrite=False)
```

## License
MIT
