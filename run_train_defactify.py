#!/usr/bin/env python3

import os
import torch
import yaml
from PIL import Image

from src.dataset.dataset import get_dataloader_defactify4
from src.model.model import CLIPWithMoE
from src.train import train_model


def main():
    root = "/home/jovyan/nkiselev/ddorin/project/machine_images/data/Defactify4/Train"
    config_path = "configs/train_config.yaml"

    model = CLIPWithMoE()

    transform = model.preprocess

    train_loader = get_dataloader_defactify4(
        root,
        split='train',
        batch_size=256,
        shuffle=True,
        transform=transform,
        num_workers=0,
    )

    val_loader = get_dataloader_defactify4(
        root,
        split='test',
        batch_size=256,
        shuffle=False,
        transform=transform,
        num_workers=0,
    )

    with open(config_path, 'r') as file:
        train_config = yaml.safe_load(file)

    train_model(model, train_loader, val_loader, train_config, resume=False)


if __name__ == "__main__":
    main()