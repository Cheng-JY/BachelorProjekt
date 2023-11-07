# models/ lenet.py

from __future__ import annotations
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics


class LeNet(pl.LightningModule):
    def __init__(self, in_channels: int, out_channels: int, lr: float = 2e-4):
        """
        Args:
        - in_channels: 1 for grayscale input image, 3 for RGB image
        - out_channels: Number of classes of the classifier
        """
        super().__init__()
        # Debugging tool to display intermediate input/output size of your layer (called before fit)
        self.example_input_array = torch.Tensor(16, in_channels, 32, 32)
        self.learning_rate = lr

        # torchmetrics calculate the Accuracy
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=out_channels)
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=out_channels)
        self.test_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=out_channels)

        # Convolution with 5*5 kernel + 2 padding
        # [img_size] 32 -> conv -> 32 -> (max pool) -> 16
        # with 6 output activation maps
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=6,
                kernel_size=5,
                stride=1,
                # padding=2
            ),
            nn.MaxPool2d(kernel_size=2),
        )

        # Convolution with 5*5 Kernel (no pad)
        # Pool with 2*2 Max Kernel + 2 Stride
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=6,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=0
            ),
            nn.MaxPool2d(kernel_size=2),
        )

        # FC Layer
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        # "Softmax" Layer
        self.fc3 = nn.Linear(in_features=84, out_features=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv_layer1(x))
        x = F.relu(self.conv_layer2(x))
        # flatten
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    # Implementing the training, validation, and test steps
    def configure_optimizers(self) -> torch.optim.Adam:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(
            self,
            batch: list[torch.Tensor, torch.Tensor],
            batch_idx: int
    ) -> torch.Tensor:
        """
        Function called when using 'trainer.fit()' with trainer a lightning 'Trainer' instance
        """
        x, y = batch
        logit_preds = self(x)
        loss = F.cross_entropy(logit_preds, y)
        self.train_accuracy.update(torch.argmax(logit_preds, dim=1), y)
        self.log("train_acc_step", self.train_accuracy, on_step=True, on_epoch=True, logger=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(
            self,
            batch: list[torch.Tensor, torch.Tensor],
            batch_idx: int,
            verbose:bool = True
    ) -> torch.Tensor:
        """
        Function called when using 'trainer.validate()' with trainer a lightning 'Trainer' instance
        """
        x, y = batch
        logit_preds = self(x)
        loss = F.cross_entropy(logit_preds, y)
        self.val_accuracy.update(torch.argmax(logit_preds, dim=1), y)
        self.log("train_acc_step", self.train_accuracy, on_epoch=True)
        self.log("train_loss", loss)
        return loss

    def test_step(
            self,
            batch: list[torch.Tensor, torch.Tensor],
            batch_idx: int
    ):
        x, y = batch
        logit_preds = self(x)
        loss = F.cross_entropy(logit_preds, y)
        self.test_accuracy.update(torch.argmax(logit_preds, dim=1), y)
        self.log_dict({"test_loss": loss, "test_acc": self.test_accuracy})

    def predict_step(
            self,
            batch: list[torch.Tensor, torch.Tensor],
            batch_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, _ = batch
        logit_preds = self(x)
        softmax_preds = F.softmax(logit_preds, dim=1)
        return x, softmax_preds