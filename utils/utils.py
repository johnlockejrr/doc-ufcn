#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    The utils module
    ======================
"""

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


def get_classes_colors(filename: str) -> list:
    """
    Read the classes.txt file and get the color tuples.
    :param filename: The name of the file that contains the colors.
    :return: The list containing the color tuples.
    """
    # Get the color codes from the classes.txt file.
    with open(filename) as classes_file:
        return [tuple(element[:-1].split(' '))
                for element in classes_file.readlines()]

