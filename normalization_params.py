#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    The normalization_params module
    ======================

    Use it to get the mean value and standard deviation of a dataset.

    :example:

    >>> python normalization_params.py with img_size=XXX
"""

import logging
import time
import torch
import numpy as np
from tqdm import tqdm
from sacred import Experiment
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.params_config import TrainingParams
from utils.preprocessing import MyDataset, Rescale, ToTensor
from utils.utils import get_classes_colors

ex = Experiment('Get normalization parameters')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


@ex.config
def default_config():
    """
    Define the default configuration for the experiment.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info('Running on {}'.format(device))
    params = TrainingParams().to_dict()
    img_size = 768


@ex.automain
def run(params: TrainingParams, img_size: int):
    """
    Main function.
    :param params: The dictionnary containing the parameters
                   of the experiment.
    :param img_size: The size in which the images are resized.
    """
    params = TrainingParams.from_dict(params)
    colors = get_classes_colors(params.classes_file)
    # Generate the training dataset / Load all the data + create batchs.
    starting_time = time.time()
    data = MyDataset(params.train_frame_path, params.train_mask_path,
                     colors,
                     transform=transforms.Compose([Rescale(img_size),
                                                   ToTensor()]))
    loader = DataLoader(data, batch_size=1,
                        shuffle=False, num_workers=2)

    logging.info('Loaded data in %1.5fs', (time.time() - starting_time))

    # Compute mean and std.
    mean = []
    std = []
    for data in tqdm(loader, desc="Computing parameters"):
        image = data['image'].numpy()
        if image.shape == (1, 3, img_size, img_size):
            mean.append(np.mean(image, axis=(0, 2, 3)))
            std.append(np.std(image, axis=(0, 2, 3)))

    mean = np.array(mean).mean(axis=0)
    std = np.array(std).mean(axis=0)

    logging.info('Mean : {}'.format(np.uint8(mean)))
    logging.info('Std  : {}'.format(np.uint8(std)))
