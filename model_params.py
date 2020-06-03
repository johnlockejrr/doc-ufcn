#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    The model_params module
    ======================

    Use it to get the summary of a model.

    :example:

    >>> python model_params.py with img_size=XXX
"""

import logging
import torch
from sacred import Experiment
from torchsummary import summary
from utils.model import Net

ex = Experiment('Get model parameters')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


@ex.config
def default_config():
    """
    Define the default configuration for the experiment.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info('Running on %s', device)
    img_size = 384
    no_of_classes = 2


@ex.automain
def run(device: str, img_size: int, no_of_classes: int):
    """
    Main function.
    :param device: The device used to run the experiment.
    :param img_size: The input size.
    :param no_of_classes: The number of classes involved in the experiment.
    """
    model = Net(no_of_classes).to(device)
    summary(model, (3, img_size, img_size))
