# BoneHub Segmentation

Medical image segmentation framework for the lower-extremity bones built on MONAI. Supports training custom nnUNet models and integration with pretrained models (TotalSegmentator, MOOSE).

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Train Custom Model

```python
from bonehub_segmentation.data_loaders import StandardSegmentationLoader, create_train_val_split
from bonehub_segmentation.model_wrappers import CustomNNUNetWrapper
from bonehub_segmentation import SegmentationTrainer

# Prepare data
train_imgs, train_labels, val_imgs, val_labels = create_train_val_split(
    "data/images", "data/labels", val_split=0.2
)

# Create loaders
train_loader = StandardSegmentationLoader(train_imgs, train_labels, mode="train").get_dataloader(batch_size=2)
val_loader = StandardSegmentationLoader(val_imgs, val_labels, mode="val").get_dataloader(batch_size=1)

# Train
trainer = SegmentationTrainer("configs/training_config.yaml")
trainer.train(train_loader, val_loader, num_epochs=100)
```

### Use Pretrained Models

```python
from bonehub_segmentation.model_wrappers import (
    TotalSegmentatorWrapper, 
    MOOSEWrapper, 
    UnifiedSegmentationInterface
)

# Direct usage
ts = TotalSegmentatorWrapper()
result = ts.segment("image.nii.gz")

# Or unified interface
interface = UnifiedSegmentationInterface()
interface.register_model("ts", TotalSegmentatorWrapper())
interface.register_model("moose", MOOSEWrapper())

# Compare models
for model_name in interface.list_models():
    result = interface.segment("image.nii.gz", model_name)
```

### Run Examples

```bash
# Custom dataloader with MSD format
python examples/00_custom_dataloader.py

# Train custom model with StandardSegmentationLoader
python examples/01_train_custom_model.py

# Use TotalSegmentator (custom model wrapper)
python examples/02_use_totalsegmentator.py

# Use MOOSE (custom model wrapper)
python examples/03_use_moose.py

# Inference with custom model
python examples/04_infer_custom_model.py

# Compare multiple models
python examples/05_compare_models.py
```

## Project Structure

```
BoneHub-Segmentation/
├── bonehub_segmentation/         # Main package
│   ├── data_loaders/            # Data loading utilities
│   │   ├── base.py              # BaseDataLoader (abstract)
│   │   ├── standard_loader.py   # StandardSegmentationLoader
│   │   └── custom_dataloaders/  # Add custom loaders here
│   ├── model_wrappers/          # Model wrapper interfaces
│   │   ├── base.py              # SegmentationModelWrapper (abstract)
│   │   ├── unified_interface.py # UnifiedSegmentationInterface
│   │   └── custom_models/       # Add custom wrappers here
│   ├── training.py
│   ├── evaluation.py
│   ├── inference.py
│   ├── visualization.py
│   └── utils.py
│
├── configs/                      # Configuration files (YAML)
│   ├── training_config.yaml     # Training hyperparameters
│   ├── model_config.yaml        # Model architecture settings
│   ├── inference_config.yaml    # Inference parameters
│   └── project_config.yaml      # Project paths and logging
│
├── checkpoints/                 # Model checkpoints (saved models)
│   └── *.pt                     # PyTorch checkpoint files (.pt)
│                               # Automatically saved during training
│                               # Reference path: configs/model_config.yaml
│
├── examples/                    # Example scripts
├── logs/                        # Training logs (if logging enabled)
├── results/                     # Inference results and outputs
├── scripts/                     # Utility scripts
├── tests/                       # Unit tests
├── requirements.txt
└── README.md
```

### Key Directories

**checkpoints/** - Stores trained model weights
- Saves best model during training
- Reference path in `configs/model_config.yaml`
- Example: `./checkpoints/best_model.pt`
- Load via: `CustomNNUNetWrapper(checkpoint="path/to/model.pt")`

**configs/** - Configuration files (now in YAML format)
- `training_config.yaml` - Learning rate, batch size, epochs, loss function
- `model_config.yaml` - Model architecture and checkpoint paths
- `inference_config.yaml` - Inference batch size, device, output settings
- `project_config.yaml` - Project metadata and directory paths

**examples/** - Working examples showing how to use the framework
- Train custom models
- Use pretrained wrappers (TotalSegmentator, MOOSE)
- Run inference and evaluation

**logs/** - Training logs and metrics (created at runtime)
**results/** - Inference outputs and predictions (created at runtime)

## Add Custom Implementations

### Custom Data Loader

Create `bonehub_segmentation/data_loaders/custom_datasets/your_loader.py`:

```python
from ..base import BaseDataLoader

class MyLoader(BaseDataLoader):
    def load_dataset(self): pass
    def get_transforms(self, config=None): pass
    def get_dataloader(self, batch_size=1, **kwargs): pass
```

Update `bonehub_segmentation/data_loaders/__init__.py` to export it.

### Custom Model Wrapper

Create `bonehub_segmentation/model_wrappers/custom_models/your_wrapper.py`:

```python
from ..base import SegmentationModelWrapper

class MyModel(SegmentationModelWrapper):
    def load_model(self): pass
    def preprocess(self, image): pass
    def inference(self, image): pass
    def postprocess(self, output): pass
```

Update `bonehub_segmentation/model_wrappers/__init__.py` to export it.

## Key Classes

**Data Loaders:**
- `BaseDataLoader` - Abstract base
- `StandardSegmentationLoader` - Load NIfTI/NRRD images
- `create_train_val_split()` - Split dataset

**Model Wrappers:**
- `SegmentationModelWrapper` - Abstract base
- `TotalSegmentatorWrapper` - 117-class anatomical
- `MOOSEWrapper` - Musculoskeletal
- `CustomNNUNetWrapper` - Custom nnUNet
- `UnifiedSegmentationInterface` - Manage multiple models

**Training & Evaluation:**
- `SegmentationTrainer` - Training pipeline
- `SegmentationEvaluator` - Calculate metrics
- `SegmentationInference` - Inference pipeline

## Configuration

Edit YAML files in `configs/`:
- `training_config.yaml` - Learning rate, batch size, epochs, loss function
- `model_config.yaml` - Model parameters and checkpoint paths
- `inference_config.yaml` - Inference settings (batch size, device, etc.)
- `project_config.yaml` - Project paths and logging configuration

## Dependencies
See `requirements.txt` for full list.

## Notes

- Start with `examples/` for usage patterns
- Extend by creating custom loaders/wrappers in `custom_*` folders

## License

MIT
