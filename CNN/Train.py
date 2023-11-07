# Train.py

from __future__ import annotations

from argparse import ArgumentParser
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from pathlib import Path
import lightning as L

import os
import random
import torch
import torchvision

from datasets.mnist import MNISTDataModule
from models.lenet import LeNet

_PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
_BATCH_SIZE = 32
_EARLY_STOPPING_PATIENCE = 4

def save_results(
        img_tensors: list[torch.Tensor],
        output_tensors: list[torch.Tensor],
        out_dir: Path,
        max_number_of_imgs: int = 10
):
    """Save test results as images in the provided output directory.
       Args:
           img_tensors: List of the tensors containing the input images.
           output_tensors: List of softmax activation from the trained model.
           out_dir: Path to output directory.
           max_number_of_imgs: Maximum number of images to output from the provided images. The images will be selected randomly.
    """
    selected_img_indices = random.sample(range(len(img_tensors)), min(max_number_of_imgs, len(img_tensors)))
    for img_indice in selected_img_indices:
        img_filepath = out_dir / f"{img_indice}_predicted_{torch.argmax(output_tensors[img_indice], dim=1)[0]}.png"
        torchvision.utils.save_image(img_tensors[img_indice][0], fp=img_filepath)