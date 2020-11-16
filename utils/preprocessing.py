#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    The preprocessing module
    ======================

    Use it to preprocess the images.
"""

import os
import cv2
import torch
import numpy as np
from skimage import io
from torch.utils.data import Dataset
from .utils import rgb_to_gray_array, rgb_to_gray_value

class MyDataset(Dataset):
    """
    The MyDataset class is used to prepare the images and labels.
    """
    def __init__(self, images_dir: str, masks_dir: str, colors: list,
                 transform: list = None):
        """
        Constructor of the MyDataset class.
        :param images_dir: The directory containing the images.
        :param masks_dir: The directory containing the labels of the images.
        :param colors: The color codes of the different classes.
        :param transform: The list of the transformations to apply.
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.colors = colors
        self.transform = transform

    def __len__(self) -> int:
        """
        Get the size of the dataset.
        :return: The size of the dataset.
        """
        return len(os.listdir(self.images_dir))

    def __getitem__(self, idx: int) -> dict:
        """
        Get a sample from the dataset according to an index.
        :param idx: The index of the wanted sample.
        :return sample: The sample with index idx.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.listdir(self.images_dir)[idx]
        image = cv2.imread(os.path.join(self.images_dir, img_name))

        if len(image.shape) < 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.masks_dir is not None:
            label = rgb_to_gray_array(io.imread(os.path.join(self.masks_dir,
                                                             img_name)))
            # Transform the label into a categorical label.
            new_label = np.zeros_like(label)
            for index, value in enumerate(self.colors):
                color = rgb_to_gray_value(value)
                new_label[label == color] = index
        else:
            new_label = None

        sample = {'image': image, 'label': new_label, 'size': image.shape[:2]}

        # Apply the transformations.
        if self.transform:
            sample = self.transform(sample)
        return sample

# Transformations


class Rescale():
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
        assert isinstance(mean, list)
        self.mean = mean

    def __call__(self, sample: dict) -> dict:
        """
        Rescale the sample image and label into the new size.
        :param sample: The sample to rescale.
        :return: The rescaled sample.
        """
        image, label = sample['image'], sample['label']
        old_size = image.shape[:2]
        # Compute the new sizes.
        ratio = float(self.output_size) / max(old_size)
        new_size = [int(x * ratio) for x in old_size]

        # Resize the image and label.
        new_image = cv2.resize(image, (new_size[1], new_size[0]))
        if label is not None:
            new_label = cv2.resize(label, (new_size[1], new_size[0]),
                                   interpolation=cv2.INTER_NEAREST)

        delta_w = self.output_size - new_image.shape[1]
        delta_h = self.output_size - new_image.shape[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        # Add padding to have same size images.
        new_image = cv2.copyMakeBorder(new_image, top, bottom, left, right,
                                       cv2.BORDER_CONSTANT, value=self.mean)
        if label is not None:
            new_label = cv2.copyMakeBorder(new_label, top, bottom, left, right,
                                           cv2.BORDER_CONSTANT, value=0)
        else:
            new_label = None
        return {'image': new_image, 'label': new_label, 'size': sample['size']}


class Normalize():
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
        :return: The sample with the normalized image.
        """
        image = sample['image']
        new_image = np.zeros(image.shape)
        for channel in range(image.shape[2]):
            new_image[:, :, channel] = (np.float32(image[:, :, channel])
                                        - self.mean[channel]) \
                                        / self.std[channel]
        return {'image': new_image, 'label': sample['label'], 'size': sample['size']}


class ToTensor():
    """
    The ToTensor class is used convert ndarrays into Tensors.
    """
    def __call__(self, sample: dict) -> dict:
        """
        Transform the sample image and label into Tensors.
        :param sample: The initial sample.
        :return: The sample with Tensors.
        """
        image, label = sample['image'], sample['label']
        image = image.transpose((2, 0, 1))
        if label is not None:
            return {'image': torch.from_numpy(image),
                    'label': torch.from_numpy(label),
                    'size': sample['size']}
        else:
            return {'image': torch.from_numpy(image)}
