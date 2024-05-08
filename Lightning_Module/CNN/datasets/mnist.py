# datasets/mnist.py

import logging
from pathlib import Path

import lightning.pytorch as pl
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import v2
from torchvision.datasets import MNIST

# Creat a logger
logger = logging.getLogger(Path(__file__).stem)
logger.setLevel(logging.INFO)

_DEFAULT_MNIST_BATCH_SIZE = 32
_DEFAULT_RESIZE_SIZE = 32

class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = _DEFAULT_MNIST_BATCH_SIZE):
        super().__init__()
        self.mnist_val = None
        self.mnist_train = None
        self.mnist_test = None
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = v2.Compose(
            [
                v2.ToTensor(),
                v2.Resize((_DEFAULT_RESIZE_SIZE, _DEFAULT_MNIST_BATCH_SIZE), antialias=True),
                v2.RandomRotation(degrees=10),
                v2.Normalize((0.1307,), (0.3081,)),
            ]
        )

    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True, transform=self.transform)
        MNIST(self.data_dir, train=False, download=True, transform=self.transform)

    def setup(self, stage: str):
        logger.info(f"Stage: {stage}")
        if stage == "test" or stage == "predict":
            self.mnist_test = MNIST(self.data_dir, train=False, download=True, transform=self.transform)
        elif stage == "fit" or stage == "validate":
            mnist_full = MNIST(self.data_dir, train=False, download=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [int(0.8*len(mnist_full)), len(mnist_full) - int(0.8*len(mnist_full))])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.mnist_test, batch_size=self.batch_size)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=3)