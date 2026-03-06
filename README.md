# BoneHub Dataset

A Python package for generating the BoneHub Dataset, including data schema, dataset conversion, segmentation, mesh and NURBS generation.

## Install from source

```bash
pip install -e .
```

## BoneHub Dataset's Data-Structure

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
