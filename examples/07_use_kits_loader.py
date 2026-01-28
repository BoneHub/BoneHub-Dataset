"""
Example: Using KiTS (Kidney Tumor Segmentation) Custom DataLoader

This example demonstrates how to use the KiTSLoader to load and process
the KiTS dataset with its specific structure (case folders with imaging,
segmentation, and optional instance annotations).
"""

# %%
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from bonehub_segmentation.data_loaders.custom_dataset_loaders import KiTSLoader
import numpy as np

# %%
data_root = r"C:\TestDatasets\073 kits23"  # Change this to your actual KiTS data path

kits_loader = KiTSLoader(
    data_root=data_root, mode="evaluation"
)  # , target_spacing=(1.0, 1.0, 1.0), target_size=(128, 128, 128))

print(f"Found {len(kits_loader)} cases")
# %%
train_dataloader = kits_loader.get_dataloader(
    batch_size=1,
    shuffle=False,
    cache_data=False,
    num_workers=0,
)

print(f"Train batches: {len(train_dataloader)}")
# %%
try:
    batch = next(iter(train_dataloader))
    print(f"Image shape: {batch['image'].shape}")
    print(f"Label shape: {batch['label'].shape}")
    print(f"Image range: [{batch['image'].min():.3f}, {batch['image'].max():.3f}]")
    print(f"Unique labels: {batch['label'].unique().tolist()}\n")
except Exception as e:
    print(f"Error loading batch: {e}\n")

# %%
import matplotlib.pyplot as plt

image = batch["image"][0, 0].numpy()  # First batch, first channel
label = batch["label"][0, 0].numpy()  # First batch, first channel

# Select middle slice
slice_idx = image.shape[2] // 2

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.imshow(image[:, :, slice_idx], cmap="gray")
# Create an alpha channel based on the label, making zero values fully transparent
alpha = np.where(label[:, :, slice_idx] == 0, 0, 0.4)
ax.imshow(label[:, :, slice_idx], cmap="jet", alpha=alpha)
ax.set_title(f"Image with Segmentation Overlay (Slice {slice_idx})")
ax.axis("off")
plt.tight_layout()
plt.show()
