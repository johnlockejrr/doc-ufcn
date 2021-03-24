#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    The utils module
    ======================

    Generic functions used during all the steps.
"""

import torch
import numpy as np

# Useful functions.


def rgb_to_gray_value(rgb: tuple) -> int:
    """
    Compute the gray value of a RGB tuple.
    :param rgb: The RGB value to transform.
    :return: The corresponding gray value.
    """
    try:
        return int(rgb[0] * 0.299 + rgb[1] * 0.587 + rgb[2] * 0.114)
    except TypeError:
        return int(int(rgb[0]) * 0.299 + int(rgb[1]) * 0.587 +
                   int(rgb[2]) * 0.114)


def rgb_to_gray_array(rgb: np.ndarray) -> np.ndarray:
    """
    Compute the gray array (NxM) of a RGB array (NxMx3).
    :param rgb: The RGB array to transform.
    :return: The corresponding gray array.
    """
    gray_array = rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 \
        + rgb[:, :, 2] * 0.114
    return np.uint8(gray_array)


def pad_images_masks(images: list, masks: list, image_padding_value: int, mask_padding_value: int):
    """
    Pad images and masks to create batchs.
    :param images: The batch images to pad.
    :param masks: The batch masks to pad.
    :param image_padding_value: The value used to pad the images.
    :param mask_padding_value: The value used to pad the masks.
    :return padded_images: An array containing the batch padded images.
    :return padded_masks: An array containing the batch padded masks.
    """
    heights = [element.shape[0] for element in images]
    widths = [element.shape[1] for element in images]
    max_height = max(heights)
    max_width = max(widths)

    # Make the tensor shape be divisible by 8.
    if max_height % 8 != 0:
        max_height = int(8 * np.ceil(max_height / 8))
    if max_width % 8 != 0:
        max_width = int(8 * np.ceil(max_width / 8))

    padded_images = np.ones((len(images), max_height, max_width, images[0].shape[2])) * image_padding_value
    padded_masks = np.ones((len(masks), max_height, max_width)) * mask_padding_value
    for index, (image, mask) in enumerate(zip(images, masks)):
        delta_h = max_height - image.shape[0]
        delta_w = max_width - image.shape[1]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        padded_images[index, top:padded_images.shape[1]-bottom, left:padded_images.shape[2]-right, :] = image
        padded_masks[index, top:padded_masks.shape[1]-bottom, left:padded_masks.shape[2]-right] = mask

    return padded_images, padded_masks


class DLACollateFunction:

    def __init__(self):
        self.image_padding_token = 0
        self.mask_padding_token = 0
        
    def __call__(self, batch):
        image = [item['image'] for item in batch]
        mask = [item['mask'] for item in batch]
        pad_image, pad_mask = pad_images_masks(image, mask,
            self.image_padding_token, self.mask_padding_token)
        return {'image': torch.tensor(pad_image).permute(0, 3, 1, 2),
                'mask': torch.tensor(pad_mask)}

