"""
The utils module
======================

Generic functions used during all the steps.
"""

import copy
import random

import numpy as np
import torch

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
        return int(int(rgb[0]) * 0.299 + int(rgb[1]) * 0.587 + int(rgb[2]) * 0.114)


def rgb_to_gray_array(rgb: np.ndarray) -> np.ndarray:
    """
    Compute the gray array (NxM) of a RGB array (NxMx3).
    :param rgb: The RGB array to transform.
    :return: The corresponding gray array.
    """
    gray_array = rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114
    return np.uint8(gray_array)


def create_buckets(images_sizes, bin_size):
    """
    Group images into same size buckets.
    :param images_sizes: The sizes of the images.
    :param bin_size: The step between two buckets.
    :return bucket: The images indices grouped by size.
    """
    max_size = max([image_size for image_size in images_sizes.values()])
    min_size = min([image_size for image_size in images_sizes.values()])

    bucket = {}
    current = min_size + bin_size - 1
    while current < max_size:
        bucket[current] = []
        current += bin_size
    bucket[max_size] = []

    for index, value in images_sizes.items():
        dict_index = (((value - min_size) // bin_size) + 1) * bin_size + min_size - 1
        bucket[min(dict_index, max_size)].append(index)

    bucket = {
        dict_index: values for dict_index, values in bucket.items() if len(values) > 0
    }
    return bucket


class Sampler(torch.utils.data.Sampler):
    def __init__(self, data, bin_size=20, batch_size=None, nb_params=None):
        self.bin_size = bin_size
        self.batch_size = batch_size
        self.nb_params = nb_params

        self.data_sizes = [image["size"] for image in data]

        self.vertical = {
            index: image["size"][1]
            for index, image in enumerate(data)
            if image["size"][0] > image["size"][1]
        }
        self.horizontal = {
            index: image["size"][0]
            for index, image in enumerate(data)
            if image["size"][0] <= image["size"][1]
        }

        self.buckets = [
            create_buckets(self.vertical, self.bin_size)
            if len(self.vertical) > 0
            else {},
            create_buckets(self.horizontal, self.bin_size)
            if len(self.horizontal) > 0
            else {},
        ]

    def __len__(self):
        return len(self.vertical) + len(self.horizontal)

    def __iter__(self):
        buckets = copy.deepcopy(self.buckets)
        for index, bucket in enumerate(buckets):
            for key in bucket:
                random.shuffle(buckets[index][key])

        if self.batch_size is not None and self.nb_params is None:
            final_indices = []
            index_current = -1
            for bucket in buckets:
                current_batch_size = self.batch_size
                for key in sorted(bucket, reverse=True):
                    for index in bucket[key]:
                        if current_batch_size + 1 > self.batch_size:
                            current_batch_size = 0
                            final_indices.append([])
                            index_current += 1
                        current_batch_size += 1
                        final_indices[index_current].append(index)
            random.shuffle(final_indices)

        elif self.nb_params is not None:
            final_indices = []
            index_current = -1
            for bucket in buckets:
                current_params = self.nb_params
                for key in sorted(bucket, reverse=True):
                    for index in bucket[key]:
                        element_params = (
                            self.data_sizes[index][0] * self.data_sizes[index][1] * 3
                        )
                        if current_params + element_params > self.nb_params:
                            current_params = 0
                            final_indices.append([])
                            index_current += 1
                        current_params += element_params
                        final_indices[index_current].append(index)
            random.shuffle(final_indices)

        return iter(final_indices)


def pad_images_masks(
    images: list, masks: list, image_padding_value: int, mask_padding_value: int
):
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

    padded_images = (
        np.ones((len(images), max_height, max_width, images[0].shape[2]))
        * image_padding_value
    )
    padded_masks = np.ones((len(masks), max_height, max_width)) * mask_padding_value
    for index, (image, mask) in enumerate(zip(images, masks, strict=True)):
        delta_h = max_height - image.shape[0]
        delta_w = max_width - image.shape[1]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        padded_images[
            index,
            top : padded_images.shape[1] - bottom,
            left : padded_images.shape[2] - right,
            :,
        ] = image
        padded_masks[
            index,
            top : padded_masks.shape[1] - bottom,
            left : padded_masks.shape[2] - right,
        ] = mask

    return padded_images, padded_masks


class DLACollateFunction:
    def __init__(self):
        self.image_padding_token = 0
        self.mask_padding_token = 0

    def __call__(self, batch):
        image = [item["image"] for item in batch]
        mask = [item["mask"] for item in batch]
        pad_image, pad_mask = pad_images_masks(
            image, mask, self.image_padding_token, self.mask_padding_token
        )
        return {
            "image": torch.tensor(pad_image).permute(0, 3, 1, 2),
            "mask": torch.tensor(pad_mask),
        }
