"""
The preprocessing module
======================

Use it to preprocess the images.
"""

import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from doc_ufcn.train.utils import rgb_to_gray_array, rgb_to_gray_value


class TrainingDataset(Dataset):
    """
    The TrainingDataset class is used to prepare the images and labels to
    run training step.
    """

    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        colors: list,
        transform: list | None = None,
    ):
        """
        Constructor of the TrainingDataset class.
        :param images_dir: The directories containing the images.
        :param masks_dir: The directories containing the masks of the images.
        :param colors: The color codes of the different classes.
        :param transform: The list of the transformations to apply.
        """
        self.images_dir = images_dir
        self.images = [
            (dir.parent.parent.name, dir / element)
            for dir in self.images_dir
            for element in os.listdir(dir)
        ]
        self.masks_dir = masks_dir
        self.masks = {dir.parent.parent.name: dir for dir in self.masks_dir}
        self.colors = colors
        self.transform = transform

    def __len__(self) -> int:
        """
        Get the size of the dataset.
        :return: The size of the dataset.
        """
        return len(self.images)

    def __getitem__(self, idx: int) -> dict:
        """
        Get a sample from the dataset according to an index.
        :param idx: The index of the wanted sample.
        :return sample: The sample with index idx.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.images[idx]

        image = cv2.imread(str(img_name[1]))
        if len(image.shape) < 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = cv2.imread(str(self.masks[img_name[0]] / img_name[1].name))
        if len(label.shape) < 3:
            label = cv2.cvtColor(label, cv2.COLOR_GRAY2RGB)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
        label = rgb_to_gray_array(label)

        # Transform the label into a categorical label.
        new_label = np.zeros_like(label)
        for index, value in enumerate(self.colors):
            color = rgb_to_gray_value(value)
            new_label[label == color] = index

        sample = {"image": image, "mask": new_label, "size": image.shape[0:2]}

        # Apply the transformations.
        if self.transform:
            sample = self.transform(sample)

        sample["size"] = sample["image"].shape[0:2]

        return sample


class PredictionDataset(Dataset):
    """
    The PredictionDataset class is used to prepare the images to
    run prediction step.
    """

    def __init__(self, images_dir: str, transform: list | None = None):
        """
        Constructor of the PredictionDataset class.
        :param images_dir: The directories containing the images.
        :param transform: The list of the transformations to apply.
        """
        self.images_dir = images_dir
        self.images = [
            (dir.parent.parent.name, dir / element)
            for dir in self.images_dir
            for element in os.listdir(dir)
        ]
        self.transform = transform

    def __len__(self) -> int:
        """
        Get the size of the dataset.
        :return: The size of the dataset.
        """
        return len(self.images)

    def __getitem__(self, idx: int) -> dict:
        """
        Get a sample from the dataset according to an index.
        :param idx: The index of the wanted sample.
        :return sample: The sample with index idx.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.images[idx][1]
        image = cv2.imread(str(img_name))

        if len(image.shape) < 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        sample = {
            "image": image,
            "name": img_name.name,
            "dataset": self.images[idx][0],
            "size": image.shape[0:2],
        }

        # Apply the transformations.
        if self.transform:
            sample = self.transform(sample)
        return sample


# Transformations


class Rescale:
    """
    The Rescale class is used to rescale the image of a sample into a
    given size.
    """

    def __init__(self, output_size: int):
        """
        Constructor of the Rescale class.
        :param output_size: The desired new size.
        """
        assert isinstance(output_size, int)
        self.output_size = output_size

    def __call__(self, sample: dict) -> dict:
        """
        Rescale the sample image into the model input size.
        :param sample: The sample to rescale.
        :return sample: The rescaled sample.
        """
        old_size = sample["image"].shape[:2]
        # Compute the new sizes.
        ratio = float(self.output_size) / max(old_size)
        new_size = [int(x * ratio) for x in old_size]

        # Resize the image.
        if max(old_size) != self.output_size:
            image = cv2.resize(sample["image"], (new_size[1], new_size[0]))
            sample["image"] = image

        # Resize the label. MUST BE AVOIDED.
        if "mask" in sample and max(sample["mask"].shape[:2]) != self.output_size:
            mask = cv2.resize(sample["mask"], (new_size[1], new_size[0]))
            sample["mask"] = mask
        return sample


class Pad:
    """
    The Pad class is used to pad the image of a sample to make it divisible by 8.
    """

    def __init__(self):
        """
        Constructor of the Pad class.
        """
        pass

    def __call__(self, sample: dict) -> dict:
        """
        Pad the sample image with zeros.
        :param sample: The sample to pad.
        :return sample: The padded sample.
        """
        # Compute the padding parameters.
        delta_w = 0
        delta_h = 0
        if sample["image"].shape[0] % 8 != 0:
            delta_h = (
                int(8 * np.ceil(sample["image"].shape[0] / 8))
                - sample["image"].shape[0]
            )
        if sample["image"].shape[1] % 8 != 0:
            delta_w = (
                int(8 * np.ceil(sample["image"].shape[1] / 8))
                - sample["image"].shape[1]
            )

        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        # Add padding to have same size images.
        image = cv2.copyMakeBorder(
            sample["image"],
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0],
        )
        sample["image"] = image
        sample["padding"] = {"top": top, "left": left}
        return sample


class Normalize:
    """
    The Normalize class is used to normalize the image of a sample.
    The mean value and standard deviation must be first computed on the
    training dataset.
    """

    def __init__(self, mean: list, std: list):
        """
        Constructor of the Normalize class.
        :param mean: The mean values (one for each channel) of the images
                     pixels of the training dataset.
        :param std: The standard deviations (one for each channel) of the
                    images pixels of the training dataset.
        """
        assert isinstance(mean, list)
        assert isinstance(std, list)
        self.mean = mean
        self.std = std

    def __call__(self, sample: dict) -> dict:
        """
        Normalize the sample image.
        :param sample: The sample with the image to normalize.
        :return sample: The sample with the normalized image.
        """
        image = np.zeros(sample["image"].shape)
        for channel in range(sample["image"].shape[2]):
            image[:, :, channel] = (
                np.float32(sample["image"][:, :, channel]) - self.mean[channel]
            ) / self.std[channel]
        sample["image"] = image
        return sample


class ToTensor:
    """
    The ToTensor class is used convert ndarrays into Tensors.
    """

    def __call__(self, sample: dict) -> dict:
        """
        Transform the sample image and label into Tensors.
        :param sample: The initial sample.
        :return sample: The sample made of Tensors.
        """
        sample["image"] = torch.from_numpy(sample["image"].transpose((2, 0, 1)))
        if "mask" in sample:
            sample["mask"] = torch.from_numpy(sample["mask"])
        return sample
