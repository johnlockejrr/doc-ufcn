#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from torch import from_numpy


def resize(input_image, network_size, padding):
    """
    Resize the input image into the network input size.
    Resize the image such that the longest side is equal to the network
    input size. Pad the image such that it is divisible by 8.
    :param input_image: The input image to resize.
    :param network_size: The input size of the model.
    :param padding: The value to use as padding.
    :return: The resized input image and the padding sizes.
    """
    old_size = input_image.shape[:2]
    if max(old_size) != network_size:
        # Compute the new sizes.
        ratio = float(network_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        # Resize the image.
        resized_image = cv2.resize(input_image, (new_size[1], new_size[0]))
    else:
        new_size = old_size
        resized_image = input_image

    delta_w = 0
    delta_h = 0
    if resized_image.shape[0] % 8 != 0:
        delta_h = int(8 * np.ceil(resized_image.shape[0] / 8)) - resized_image.shape[0]
    if resized_image.shape[1] % 8 != 0:
        delta_w = int(8 * np.ceil(resized_image.shape[1] / 8)) - resized_image.shape[1]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    resized_image = cv2.copyMakeBorder(
        resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding
    )
    return resized_image, [top, left]


def preprocess_image(input_image, model_input_size, mean, std):
    """
    Preprocess the input image before feeding it to the network.
    The image is first resized, normalized and converted to a tensor.
    :param input_image: The input image to preprocess.
    :param model_input_size: The size of the model input.
    :param mean: The mean value used to normalize the image.
    :param std: The standard deviation used to normalize the image.
    :return: The resized, normalized and padded input tensor.
    """
    # Resize the image
    resized_image, padding = resize(input_image, model_input_size, padding=mean)
    # Normalize the image
    normalized_image = np.zeros(resized_image.shape)
    for channel in range(resized_image.shape[2]):
        normalized_image[:, :, channel] = (
            np.float32(resized_image[:, :, channel]) - mean[channel]
        ) / std[channel]
    # To tensor
    normalized_image = normalized_image.transpose((2, 0, 1))
    normalized_image = np.expand_dims(normalized_image, axis=0)
    return from_numpy(normalized_image), padding
