# -*- coding: utf-8 -*-

"""
    The normalization params module
    ======================

    Use it to get the mean and standard deviation of the training set.
"""

import logging

import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from doc_ufcn.train.utils.preprocessing import PredictionDataset, Rescale, ToTensor


def run(
    log_path: str,
    data_paths: dict,
    img_size: int,
    mean_name: str,
    std_name: str,
    num_workers: int = 2,
):
    """
    Compute the normalization parameters: mean and standard deviation on
    train set.
    :param log_path: Path to save the experiment information and model.
    :param data_paths: Path to the data folders.
    :param img_size: The network input image size.
    :param mean_name: Name of the file that will contain all the mean values.
    :param std_name: Name of the file that will contain all the std values.
    """
    dataset = PredictionDataset(
        data_paths["train"]["image"],
        transform=transforms.Compose([Rescale(img_size), ToTensor()]),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)

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

    with (log_path / mean_name).open("w") as file:
        for value in mean:
            file.write(str(np.uint8(value)) + "\n")

    with (log_path / std_name).open("w") as file:
        for value in std:
            file.write(str(np.uint8(value)) + "\n")
