#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    The run experiment module
    ======================

    Use it to train, predict and evaluate a model.
"""

import os
import sys
import logging
from sacred import Experiment
from sacred.observers import MongoObserver
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import normalization_params
import train, predict, evaluate
from utils import model, utils
from utils.params_config import Params
import utils.preprocessing as pprocessing
import utils.training_utils as tr_utils

ex = Experiment('U-FCN model')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
mongo_url='mongodb://user:password@omniboard.vpn/sacred'

@ex.config
def default_config():
    """
    Define the default configuration for the experiment.
    :experiment_name: The name of the experiment that is used to save all
                      the experiment information.
    :log_path: Path to save the experiment information and model.
    :tb_path: Path to save the Tensorboard events.
    :classes_names: The names of the classes involved during the experiment.
    :img_size: Network input size.
    :no_of_epochs: Total number of epochs to run.
    :batch_size: Size of the batch to use during training.
    :min_cc: Threshold used to remove small connected components.
    :save_image: List of sets (train, val, test) for which the prediction images
                 are generated and saved.
    :params: Parameters to use during all the experiment steps.
    :learning_rate: Initial learning rate to use during training.
    :restore_model: Path to the model to restore to resume a training.
    :steps: List of the steps to run.
    """
    experiment_name = 'ufcn'
    log_path = 'runs/'+experiment_name.lower().replace(' ', '_').replace('-', '_')
    tb_path = 'events/'
    classes_names = ["Background", "Text_line"]
    img_size = 768
    no_of_epochs = 2
    batch_size = 4
    min_cc = 0
    save_image = []
    params = Params().to_dict()
    learning_rate = 5e-3
    restore_model = None
    steps = ["normalization_params", "train", "prediction", "evaluation"]
    omniboard = False
    if "train" in steps and omniboard is True:
        ex.observers.append(MongoObserver(mongo_url))


@ex.capture
def get_mean_std(params: Params) -> dict:
    """
    Retrieve the mean and std values computed during the first 'normalization
    params' step.
    :param params: Parameters to use to find the mean and std files.
    :return: A dictionary containing the mean and std values.
    """
    params = Params.from_dict(params)
    if not os.path.isfile(params.mean):
        logging.error('No file found at %s', params.mean)
        sys.exit()
    else:
        with open(params.mean, 'r') as file:
            mean = file.read().splitlines()
            mean = [int(value) for value in mean]
    if not os.path.isfile(params.std):
        logging.error('No file found at %s', params.std)
        sys.exit()
    else:
        with open(params.std, 'r') as file:
            std = file.read().splitlines()
            std = [int(value) for value in std]
    return {'mean': mean, 'std': std}


@ex.capture
def training_loaders(colors: list, norm_params: dict, params: Params,
                     img_size: int, batch_size: int) -> dict:
    """
    Generate the loaders to use during the training step.
    :param colors: Colors of the classes used during the experiment.
    :param norm_params: The mean and std values used during image normalization.
    :param params: Global experiment parameters.
    :param img_size: Network input size.
    :param batch_size: Size of the batch to use during training.
    :return loaders: A dictionary with the loaders.
    """
    params = Params.from_dict(params)
    loaders = {}
    for set, images, masks in zip(['train', 'val'],
                                  [params.train_image_path, params.val_image_path],
                                  [params.train_mask_path, params.val_mask_path]):
        dataset = pprocessing.TrainingDataset(
            images, masks,
            colors, transform=transforms.Compose([
                pprocessing.Rescale(img_size),
                pprocessing.Pad(img_size, norm_params['mean'],),
                pprocessing.Normalize(norm_params['mean'], norm_params['std']),
                pprocessing.ToTensor()])
        )
        loaders[set+'_loader'] = DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=2, pin_memory=True)
    return loaders


@ex.capture
def prediction_loaders(norm_params: dict, params: Params, img_size: int) -> dict:
    """
    Generate the loaders to use during the prediction step.
    :param norm_params: The mean and std values used during image normalization.
    :param params: Global experiment parameters.
    :param img_size: Network input size.
    :return loaders: A dictionary with the loaders.
    """
    params = Params.from_dict(params)
    loaders = {}
    for set, images in zip(['train', 'val', 'test'],
                           [params.train_image_path, params.val_image_path, params.test_image_path]):
        dataset = pprocessing.PredictionDataset(
            images,
            transform=transforms.Compose([
                pprocessing.Rescale(img_size),
                pprocessing.Pad(img_size, norm_params['mean'],),
                pprocessing.Normalize(norm_params['mean'], norm_params['std']),
                pprocessing.ToTensor()])
        )
        loaders[set+'_loader'] = DataLoader(dataset, batch_size=1, shuffle=False,
                                            num_workers=2, pin_memory=True)
    return loaders


@ex.capture
def training_initialization(classes_names: int, learning_rate: float,
                            restore_model: bool, log_path: str) -> dict:
    """
    Initialize the training step.
    :param classes_names: The names of the classes involved during the experiment.
    :param learning_rate: Initial learning rate to use during training.
    :param restore_model: Path to the model to restore to resume a training.
    :param log_path: Path to save the experiment information and model.
    :return tr_params: A dictionary with the training parameters.
    """
    no_of_classes = len(classes_names)
    ex.log_scalar('no_of_classes', no_of_classes)
    net, softmax = model.load_network(no_of_classes, ex)
    
    if restore_model is None:
        net.apply(model.weights_init)
        tr_params = {
            'net': net,
            'softmax': softmax,
            'criterion': tr_utils.Diceloss(no_of_classes),
            'optimizer': optim.Adam(net.parameters(), lr=learning_rate),
            'saved_epoch': 0,
            'best_loss': 10e5,
        }
    else:
    # Restore model to resume training.
        checkpoint, net, optimizer = model.restore_model(
            net, optim.Adam(net.parameters(), lr=learning_rate),
            log_path, restore_model)
        tr_params = {
            'net': net,
            'softmax': softmax,
            'criterion': tr_utils.Diceloss(no_of_classes),
            'optimizer': optimizer,
            'saved_epoch': checkpoint['epoch'],
            'best_loss': checkpoint['best_loss'],
        }
    return tr_params


@ex.capture
def prediction_initialization(params: dict, classes_names: int,
                              log_path: str) -> dict:
    """
    Initialize the prediction step.
    :param params: The global parameters of the experiment.
    :param classes_names: The names of the classes involved during the experiment.
    :param log_path: Path to save the experiment information and model.
    :return: A dictionary with the training parameters.
    """
    params = Params.from_dict(params)
    no_of_classes = len(classes_names)
    net, softmax = model.load_network(no_of_classes, ex)

    _, net, _ = model.restore_model(net, None, log_path, params.model_path)
    return {'net': net, 'softmax': softmax}


@ex.automain
def run(params: Params, img_size: int, log_path: str, tb_path: str, no_of_epochs: int,
        classes_names: list, save_image: list, min_cc: int, steps: list):
    """
    Main program.
    :param params: The global parameters of the experiment.
    :param img_size: Network input size.
    :param log_path: Path to save the experiment information and model.
    :param tb_path: Path to save the Tensorboard events.
    :param no_of_epochs: Total number of epochs to run.
    :param classes_names: The names of the classes involved during the experiment.
    :param save_image: List of sets (train, val, test) for which the prediction images
                       are generated and saved.
    :param min_cc: The threshold used to remove small connected components.
    :param steps: List of the steps to run.
    """
    if len(steps) == 0:
        logging.info("No step to run, exiting execution.")
    else:
        # Get the default parameters.
        params = Params.from_dict(params)

        if "normalization_params" in steps:
            normalization_params.run(params, img_size)

        if "train" in steps or "prediction" in steps:
            # Get the mean and std values.
            norm_params = get_mean_std()
            # Get the possible colors.
            colors = utils.get_classes_colors(params.classes_file)

        if "train" in steps:
            # Generate the loaders and start training.
            loaders = training_loaders(colors, norm_params)
            tr_params = training_initialization()
            train.run(params.model_path, log_path, tb_path, no_of_epochs,
                      norm_params, classes_names, loaders, tr_params, ex)

        if "prediction" in steps:
            # Generate the loaders and start predicting.
            loaders = prediction_loaders(norm_params)
            pr_params = prediction_initialization()
            predict.run(params.prediction_path, log_path, img_size, colors,
                        classes_names, save_image, min_cc, loaders, pr_params)

        if "evaluation" in steps:
            evaluate.run(log_path, classes_names, params)
