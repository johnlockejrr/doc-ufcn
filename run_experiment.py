#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    The run experiment module
    ======================

    Use it to train, predict and evaluate a model.
"""

import os
import sys
import json
import logging
from pathlib import Path
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

STEPS = ["normalization_params", "train", "prediction", "evaluation"]

ex = Experiment('Doc-UFCN')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
mongo_url='mongodb://user:password@omniboard.vpn/sacred'

@ex.config
def default_config():
    """
    Define the default configuration for the experiment.
    :classes_names: The names of the classes involved during the experiment.
    :classes_file: File containing the classes color codes.
    :img_size: Network input size.
    :no_of_epochs: Total number of epochs to run.
    :batch_size: Size of the batch to use during training.
    :learning_rate: Initial learning rate to use during training.
    :omniboard: Whether to use Omniboard.
    :min_cc: Threshold used to remove small connected components.
    :save_image: List of sets (train, val, test) for which the prediction images
                 are generated and saved.
    :params: Parameters to use during all the experiment steps.
    :experiment_name: The name of the experiment that is used to save all
                      the experiment information.
    :log_path: Path to save the experiment information and model.
    :tb_path: Path to save the Tensorboard events.
    :steps: List of the steps to run.
    :data_paths: Path to the experiment data folders.
    :restore_model: Path to the model to restore to resume a training.
    :same_data: Indicated whether the model to train uses the same data as
                the model to restore.

    """
    # Load the global experiments parameters from experiments_config.json.
    global_params = {
        "classes_names": ["background", "text_line"],
        "classes_file": "./data/classes.txt",
        "img_size": 768,
        "no_of_epochs": 100,
        "batch_size": 4,
        "learning_rate": 5e-3,
        "omniboard": False,
        "min_cc": 0,
        "save_image": []
    }
    classes_file = Path(global_params['classes_file']).expanduser()
    params = Params().to_dict()

    # Load the current experiment parameters.
    experiment_name = 'doc-ufcn'
    log_path = 'runs/'+experiment_name.lower().replace(' ', '_').replace('-', '_')
    tb_path = 'events/'
    steps = ["normalization_params", "train", "prediction", "evaluation"]
    for step in steps:
        assert step in STEPS

    data_paths = {
        "train": {
            "image": ["./data/train/images/"],
            "mask": ["./data/train/labels/"],
            "json": []
        },
        "val": {
            "image": ["./data/val/images/"],
            "mask": ["./data/val/labels/"],
            "json": []
        },
        "test": {
            "image": ["./data/test/images/"],
            "json": ["./data/test/labels_json/"]
        }
    }
    exp_data_paths = {set:
        {key: [Path(element).expanduser() for element in value]
        for key, value in paths.items()}
        for set, paths in data_paths.items()
    }

    training = {
        "restore_model": None,
        "same_data": True,
    }
    if training['same_data'] in ['true', 'True']:
        training['same_data'] = True
    elif training['same_data'] in ['false', 'False']:
        training['same_data'] = False
    else:
        training['same_data'] = None
    if "train" in steps and global_params['omniboard'] is True:
        ex.observers.append(MongoObserver(mongo_url))


@ex.capture
def save_config(log_path: str, experiment_name: str, global_params: dict,
                params: Params, steps: list, data_paths: dict, training: dict):
    """
    Save the current configuration.
    :param log_path: Path to save the experiment information and model.
    :param experiment_name: The name of the experiment that is used to save all
                      the experiment information.
    :param global_params: Global parameters of the experiment entered by the used.
    :param params: Global parameters of the experiment.
    :param steps: List of the steps to run.
    :param data_paths: Path to the experiment data folders.
    :param training: Training parameters.
    """
    os.makedirs(log_path, exist_ok=True)
    json_dict = {
        'global_params': global_params,
        'params': params,
        'steps': steps,
        'data_paths': data_paths,
        'training': training
    }
    with open(os.path.join(log_path, experiment_name+'.json'), 'w') as config_file:
        json.dump(json_dict, config_file, indent=4)


@ex.capture
def get_mean_std(log_path: str, params: Params) -> dict:
    """
    Retrieve the mean and std values computed during the first 'normalization
    params' step.
    :param log_path: Path to save the experiment information and model.
    :param params: Parameters to use to find the mean and std files.
    :return: A dictionary containing the mean and std values.
    """
    params = Params.from_dict(params)
    if not os.path.isfile(os.path.join(log_path, params.mean)):
        logging.error('No file found at %s', os.path.join(log_path, params.mean))
        sys.exit()
    else:
        with open(os.path.join(log_path, params.mean), 'r') as file:
            mean = file.read().splitlines()
            mean = [int(value) for value in mean]
    if not os.path.isfile(os.path.join(log_path, params.std)):
        logging.error('No file found at %s', os.path.join(log_path, params.std))
        sys.exit()
    else:
        with open(os.path.join(log_path, params.std), 'r') as file:
            std = file.read().splitlines()
            std = [int(value) for value in std]
    return {'mean': mean, 'std': std}


@ex.capture
def training_loaders(colors: list, norm_params: dict, exp_data_paths: dict,
                     global_params: dict) -> dict:
    """
    Generate the loaders to use during the training step.
    :param colors: Colors of the classes used during the experiment.
    :param norm_params: The mean and std values used during image normalization.
    :param exp_data_paths: Path to the data folders.
    :param global_params: Global parameters of the experiment entered by the used.
    :return loaders: A dictionary with the loaders.
    """
    loaders = {}
    for set, images, masks in zip(['train', 'val'],
                                  [exp_data_paths['train']['image'], exp_data_paths['val']['image']],
                                  [exp_data_paths['train']['mask'], exp_data_paths['val']['mask']]):
        dataset = pprocessing.TrainingDataset(
            images, masks,
            colors, transform=transforms.Compose([
                pprocessing.Rescale(global_params['img_size']),
                pprocessing.Pad(global_params['img_size'], norm_params['mean']),
                pprocessing.Normalize(norm_params['mean'], norm_params['std']),
                pprocessing.ToTensor()])
        )
        loaders[set+'_loader'] = DataLoader(dataset, batch_size=global_params['batch_size'],
                                            shuffle=True, num_workers=2, pin_memory=True)
    return loaders


@ex.capture
def prediction_loaders(norm_params: dict, exp_data_paths: dict,
                       global_params: dict) -> dict:
    """
    Generate the loaders to use during the prediction step.
    :param norm_params: The mean and std values used during image normalization.
    :param exp_data_paths: Path to the data folders.
    :param global_params: Global parameters of the experiment entered by the used.
    :return loaders: A dictionary with the loaders.
    """
    loaders = {}
    for set, images in zip(
            ['train', 'val', 'test'],
            [exp_data_paths['train']['image'], exp_data_paths['val']['image'], exp_data_paths['test']['image']]):
        dataset = pprocessing.PredictionDataset(
            images,
            transform=transforms.Compose([
                pprocessing.Rescale(global_params['img_size']),
                pprocessing.Pad(global_params['img_size'], norm_params['mean']),
                pprocessing.Normalize(norm_params['mean'], norm_params['std']),
                pprocessing.ToTensor()])
        )
        loaders[set+'_loader'] = DataLoader(dataset, batch_size=1, shuffle=False,
                                            num_workers=2, pin_memory=True)
    return loaders


@ex.capture
def training_initialization(global_params: dict, training: dict, log_path: str) -> dict:
    """
    Initialize the training step.
    :param global_params: Global parameters of the experiment entered by the used.
    :param training: Training parameters.    
    :param log_path: Path to save the experiment information and model.
    :return tr_params: A dictionary with the training parameters.
    """
    no_of_classes = len(global_params['classes_names'])
    ex.log_scalar('no_of_classes', no_of_classes)
    net, softmax = model.load_network(no_of_classes, ex)
    
    if training['restore_model'] is None:
        net.apply(model.weights_init)
        tr_params = {
            'net': net,
            'softmax': softmax,
            'criterion': tr_utils.Diceloss(no_of_classes),
            'optimizer': optim.Adam(net.parameters(), lr=global_params['learning_rate']),
            'saved_epoch': 0,
            'best_loss': 10e5,
        }
    else:
    # Restore model to resume training.
        checkpoint, net, optimizer = model.restore_model(
            net, optim.Adam(net.parameters(), lr=global_params['learning_rate']),
            log_path, training['restore_model'])
        tr_params = {
            'net': net,
            'softmax': softmax,
            'criterion': tr_utils.Diceloss(no_of_classes),
            'optimizer': optimizer,
            'saved_epoch': checkpoint['epoch'],
            'best_loss': checkpoint['best_loss'] if training['same_data'] else 10e5,
        }
    return tr_params


@ex.capture
def prediction_initialization(params: dict, global_params: dict,
                              log_path: str) -> dict:
    """
    Initialize the prediction step.
    :param params: The global parameters of the experiment.
    :param global_params: Global parameters of the experiment entered by the used.
    :param log_path: Path to save the experiment information and model.
    :return: A dictionary with the training parameters.
    """
    params = Params.from_dict(params)
    no_of_classes = len(global_params['classes_names'])
    net, softmax = model.load_network(no_of_classes, ex)

    _, net, _ = model.restore_model(net, None, log_path, params.model_path)
    return {'net': net, 'softmax': softmax}


@ex.automain
def run(global_params: dict, classes_file: Path, params: Params, log_path: str,
        tb_path: str, steps: list, exp_data_paths: dict, training: dict):
    """
    Main program.
    :param global_params: Global parameters of the experiment entered by the used.
    :param classes_file: File containing the classes color codes.
    :param params: Global parameters of the experiment.
    :param log_path: Path to save the experiment information and model.
    :param tb_path: Path to save the Tensorboard events.
    :param steps: List of the steps to run.
    :param exp_data_paths: Path to the data folders.
    :param training: Training parameters.    
    """
    if len(steps) == 0:
        logging.info("No step to run, exiting execution.")
    else:
        # Get the default parameters.
        params = Params.from_dict(params)
        save_config()

        if "normalization_params" in steps:
            normalization_params.run(log_path, exp_data_paths,
                                     params, global_params['img_size'])

        if "train" in steps or "prediction" in steps:
            # Get the mean and std values.
            norm_params = get_mean_std()
            # Get the possible colors.
            colors = utils.get_classes_colors(classes_file)

        if "train" in steps:
            # Generate the loaders and start training.
            loaders = training_loaders(colors, norm_params)
            tr_params = training_initialization()
            train.run(params.model_path, log_path, tb_path, global_params['no_of_epochs'],
                      norm_params, global_params['classes_names'], loaders, tr_params, ex)

        if "prediction" in steps:
            # Generate the loaders and start predicting.
            loaders = prediction_loaders(norm_params)
            pr_params = prediction_initialization()
            predict.run(params.prediction_path, log_path, global_params['img_size'],
                        colors, global_params['classes_names'],  global_params['save_image'],
                        global_params['min_cc'], loaders, pr_params)

        if "evaluation" in steps:
            for set in exp_data_paths.keys():
                for dataset in exp_data_paths[set]["json"]:
                    if os.path.isdir(dataset):
                        evaluate.run(log_path, global_params['classes_names'], set,
                                     exp_data_paths[set]['json'], str(dataset.parent.parent.name),
                                     params)
                    else:
                        logging.info(f"{dataset} folder not found.")
