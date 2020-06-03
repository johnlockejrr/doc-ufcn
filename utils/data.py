#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    The data module
    ======================

    Use it to prepare the data.
"""

import logging
import time
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.preprocessing import MyDataset, Rescale, ToTensor, Normalize


def load_data(frame_path: str, img_size: int, mean: list, std: list,
              classes: list, mask_path: str = None, batch_size: int = 1,
              shuffle: bool = True):
    """
    Load the data for the experiment.
    Preprocess the images and generate batches of batch_size images.
    :param frame_path: The path to the images directory.
    :param img_size: The size in which the images will be resized.
    :param mean: The mean values used to normalize the images.
    :param std: The standard deviations used to normalize the images.
    :param classes: The color codes of the different classes.
    :param mask_path: The path to the labels directory.
    :param batch_size: The size of the batches to generate.
    :param shuffle: A boolean indicating whether to shuffle the images
                    when creating the batches.
    :return loader: A DataLoader object with the preprocessed images.
    """
    starting_time = time.time()
    # Generate the dataset / Load all the data (+ create batches).
    data = MyDataset(frame_path, mask_path, classes,
                     transform=transforms.Compose([Rescale(img_size),
                                                   Normalize(mean, std),
                                                   ToTensor()]))
    loader = DataLoader(data, batch_size=batch_size,
                        shuffle=shuffle, num_workers=2)
    logging.info('Loaded data in %1.5fs', (time.time() - starting_time))
    return loader
