import json
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import albumentations as A
from typing import List, Dict
import numpy as np
import os

class defactify4(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.image_paths, self.labels = self._load_dataset()

    def _load_dataset(self):
        image_paths = []
        labels = []
        for class_name in os.listdir(self.root_dir):
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            images = sorted(os.listdir(class_dir))
            for img_name in images:
                img_path = os.path.join(class_dir, img_name)
                image_paths.append(img_path)
                labels.append(0 if class_name == 'coco_image' else 1)

        # Split the dataset into train and test
        total_size = len(image_paths)
        train_size = int(0.9 * total_size)
        if self.split == "train":
            image_paths = image_paths[:train_size]
            labels = labels[:train_size]
        else:
            image_paths = image_paths[train_size:]
            labels = labels[train_size:]

        return image_paths, labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def get_dataloader_defactify4(root_dir, split="train", batch_size=8, shuffle=True, transform=None, num_workers=0):
    """
    Builds and returns a PyTorch DataLoader for the specified dataset split.

    Args:
        root_dir (str): Path to the root directory of the dataset.
        split (str): Which dataset split to load ('train' or 'test').
        batch_size (int): How many samples per batch to load.
        shuffle (bool): Whether to shuffle the dataset.
        transform (callable, optional): Preprocessing transforms.
        num_workers (int): Number of subprocesses to use for data loading.

    Returns:
        DataLoader: A DataLoader instance for the specified split.
    """
    assert split in ["train", "test"], "Split must be either 'train' or 'test'"

    dataset = defactify4(root_dir, split=split, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    return loader


class artifact(Dataset):
    def __init__(self, json_list, aug_transform=None, transform=None):
        """
        Dataset for binary classification with support for image-level augmentations on PIL.Image.

        Args:
            json_list (list): List of dictionaries containing metadata.
            aug_transform (callable): Augmentation transforms that work on PIL.Image.
            transform (callable): Preprocessing transforms (e.g., ToTensor, Normalize).
        """
        self.data = json_list
        self.class_counts = {
            "class_0": sum(1 for item in self.data if item["target"] == 0),
            "class_1": sum(1 for item in self.data if item["target"] == 1),
        }
        self.aug_transform = aug_transform
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item["image_path"]
        image = Image.open(image_path).convert("RGB")
        target = item["target"]

        if self.aug_transform is not None:
            image = self.aug_transform(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, torch.tensor(target, dtype=torch.long)


def get_dataloader(json_path, split="train", batch_size=32, shuffle=True,
                   aug_transform=None, transform=None, num_workers=0):
    """
    Builds and returns a PyTorch DataLoader for the specified dataset split.

    Args:
        json_path (str): Path to the JSON file containing dataset metadata.
        split (str): Which dataset split to load ('train' or 'test').
        batch_size (int): How many samples per batch to load.
        shuffle (bool): Whether to shuffle the dataset.
        aug_transform (callable, optional): Data augmentation transforms.
        transform (callable, optional): Preprocessing transforms.
        num_workers (int): Number of subprocesses to use for data loading.

    Returns:
        DataLoader: A DataLoader instance for the specified split.
    """
    assert split in ["train", "test"], "Split must be either 'train' or 'test'"

    data_list = load_split_from_json(json_path, split)
    dataset = artifact(data_list, aug_transform=aug_transform, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    return loader


def load_split_from_json(json_path: str, split: str = "train") -> List[Dict]:
    """
    Loads the specified dataset split ('train' or 'test') from a JSON metadata file.

    Args:
        json_path (str): Path to the JSON file containing dataset metadata.
        split (str): Dataset split to load. Must be either 'train' or 'test'.

    Returns:
        list of dict: A list of dictionaries where each dictionary contains metadata
                      for one sample (e.g., 'image_path', 'target', etc.).

    Raises:
        AssertionError: If the specified split is not 'train' or 'test'.
        FileNotFoundError: If the JSON file does not exist.
        json.JSONDecodeError: If the JSON file is not valid.
    """
    assert split in ["train", "test"], f"Invalid split name: {split}. Expected 'train' or 'test'."

    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file not found at path: {json_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in file: {json_path}")

    return data[split]


class AlbumentationsAugmenter:
    """
    A class for applying albumentations-based augmentations to PIL images.
    
    Attributes:
        transform: The albumentations augmentation pipeline.
    """

    def __init__(self):
        """Initializes the augmenter with a predefined albumentations pipeline."""
        self.transform = self._get_aug()

    def _get_aug(self):
        """
        Returns an albumentations.Compose object with a standard augmentation pipeline.
        
        Includes:
            - RandomResizedCrop
            - RandomRotate90
            - Horizontal and Vertical flips
            - Grayscale conversion
            - ColorJitter
        
        Returns:
            A callable albumentations.Compose object.
        """
        return A.Compose([
            A.RandomResizedCrop(height=224, width=224, scale=(0.85, 0.95), p=0.5),
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ToGray(p=0.2),
            A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, p=0.2),
        ])

    def __call__(self, image):
        """
        Applies the defined augmentation pipeline to a PIL.Image.
        
        Args:
            image (PIL.Image): Input image to be augmented.
            
        Returns:
            PIL.Image: Augmented image.
        """
        image_np = np.array(image)
        augmented_np = self.transform(image=image_np)['image']
        return Image.fromarray(augmented_np)