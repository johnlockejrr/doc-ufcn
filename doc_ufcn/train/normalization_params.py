#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    The normalization params module
    ======================

    Use it to get the mean and standard deviation of the training set.
"""

import logging
import os

import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from doc_ufcn.train.utils.preprocessing import PredictionDataset, Rescale, ToTensor


def run(log_path: str, data_paths: dict, params: dict, img_size: int):
    """
    Compute the normalization parameters: mean and standard deviation on
    train set.
    :param log_path: Path to save the experiment information and model.
    :param data_paths: Path to the data folders.
    :param params: Parameters to use to find the mean and std values.
    :param img_size: The network input image size.
    """
    dataset = PredictionDataset(
        data_paths["train"]["image"],
        transform=transforms.Compose([Rescale(img_size), ToTensor()]),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    # Compute mean and std.
    mean = []
    std = []
    for data in tqdm(loader, desc="Computing parameters (prog)"):
        image = data["image"].numpy()
        mean.append(np.mean(image, axis=(0, 2, 3)))
        std.append(np.std(image, axis=(0, 2, 3)))

    mean = np.array(mean).mean(axis=0)
    std = np.array(std).mean(axis=0)

    logging.info("Mean: {}".format(np.uint8(mean)))
    logging.info(" Std: {}".format(np.uint8(std)))

    with open(os.path.join(log_path, params["mean"]), "w") as file:
        for value in mean:
            file.write(str(np.uint8(value)) + "\n")

    with open(os.path.join(log_path, params["std"]), "w") as file:
        for value in std:
            file.write(str(np.uint8(value)) + "\n")
