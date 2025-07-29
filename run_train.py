#!/usr/bin/env python3

import os
import torch
import yaml
from PIL import Image

from src.dataset.dataset import get_dataloader, AlbumentationsAugmenter
from src.model.model import CLIPWithMoE
from src.train import train_model


def main():
    json_path = "/home/jovyan/nkiselev/ddorin/project/machine_images/data/dataset.json"
    config_path = "configs/train_config.yaml"

    model = CLIPWithMoE()

    aug_transform = AlbumentationsAugmenter()
    transform = model.preprocess

    train_loader = get_dataloader(
        json_path,
        split='train',
        batch_size=256,
        shuffle=True,
        aug_transform=None,
        transform=transform,
        num_workers=0,
    )

    val_loader = get_dataloader(
        json_path,
        split='test',
        batch_size=256,
        shuffle=False,
        aug_transform=None,
        transform=transform,
        num_workers=0,
    )

    with open(config_path, 'r') as file:
        train_config = yaml.safe_load(file)

    train_model(model, train_loader, val_loader, train_config, resume=True)


if __name__ == "__main__":
    main()