"""Training pipeline for segmentation models."""

import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from tqdm import tqdm


class SegmentationTrainer:
    """Simple trainer for segmentation models."""

    def __init__(self, model: nn.Module, train_loader, val_loader, device: str = "cuda"):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader

    def setup_training(self, num_classes: int, learning_rate: float = 1e-4):
        """Setup optimizer, loss, and metrics."""
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.loss_fn = DiceLoss(to_onehot_y=True, softmax=True, reduction="mean")
        self.dice_metric = DiceMetric(include_background=False, reduction="mean", num_classes=num_classes)

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0
        num_batches = 0

        for batch_data in tqdm(self.train_loader, desc="Training"):
            images = batch_data["image"].to(self.device)
            labels = batch_data["label"].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        return epoch_loss / num_batches

    def validate_epoch(self) -> dict:
        """Validate for one epoch."""
        self.model.eval()
        val_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch_data in tqdm(self.val_loader, desc="Validation"):
                images = batch_data["image"].to(self.device)
                labels = batch_data["label"].to(self.device)

                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)
                self.dice_metric(outputs, labels)

                val_loss += loss.item()
                num_batches += 1

        dice = self.dice_metric.aggregate().item()
        self.dice_metric.reset()

        return {"loss": val_loss / num_batches, "dice": dice}

    def save_checkpoint(self, checkpoint_path: str):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(self.model.state_dict(), checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))

    def train(self, num_epochs: int, checkpoint_dir: str = "./checkpoints"):
        """Train for multiple epochs."""
        best_dice = 0

        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_metrics = self.validate_epoch()

            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | Dice: {val_metrics['dice']:.4f}")

            if val_metrics["dice"] > best_dice:
                best_dice = val_metrics["dice"]
                self.save_checkpoint(f"{checkpoint_dir}/best_model.pt")
