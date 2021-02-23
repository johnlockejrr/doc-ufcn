#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    The normalization params module
    ======================

    Use it to get the mean and standard deviation of the training set.
"""

import logging
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.params_config import Params
import utils.preprocessing as pprocessing

def run(params: Params, img_size: int):
    """
    Compute the normalization parameters: mean and standard deviation on
    train set.
    :param params: Parameters to use to find the mean and std values.
    :param img_size: The network input image size.
    """
    dataset = pprocessing.PredictionDataset(
        params.train_image_path,
        transform=transforms.Compose([pprocessing.Rescale(img_size),
                                      pprocessing.ToTensor()]))
    loader = DataLoader(dataset, batch_size=1,
                        shuffle=False, num_workers=2)

    # Compute mean and std.
    mean = []
    std = []
    for data in tqdm(loader, desc="Computing parameters (prog)"):
        image = data['image'].numpy()
        mean.append(np.mean(image, axis=(0, 2, 3)))
        std.append(np.std(image, axis=(0, 2, 3)))

    mean = np.array(mean).mean(axis=0)
    std = np.array(std).mean(axis=0)

    logging.info('Mean: {}'.format(np.uint8(mean)))
    logging.info(' Std: {}'.format(np.uint8(std)))
    
    with open(params.mean, 'w') as file:
        for value in mean:
            file.write(str(np.uint8(value))+'\n')

    with open(params.std, 'w') as file:
        for value in std:
            file.write(str(np.uint8(value))+'\n')
