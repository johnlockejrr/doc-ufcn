#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    The run experiment module
    ======================

    Use it to train, predict and evaluate a model.
"""

import json
import logging
import os
from pathlib import Path

from torch.cuda.amp import GradScaler
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from doc_ufcn import model
from doc_ufcn.train.evaluate import run as evaluate
from doc_ufcn.train.normalization_params import run as normalization_params
from doc_ufcn.train.predict import run as predict
from doc_ufcn.train.training import run as train
from doc_ufcn.train.utils import DLACollateFunction, Sampler
from doc_ufcn.train.utils.preprocessing import (
    Normalize,
    Pad,
    PredictionDataset,
    Rescale,
    ToTensor,
    TrainingDataset,
)
from doc_ufcn.train.utils.training import Diceloss

logger = logging.getLogger(__name__)


def save_config(config: dict):
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
    os.makedirs(config["log_path"], exist_ok=True)
    path = config["log_path"] / (config["experiment_name"] + ".json")
    with open(path, "w") as config_file:
        json.dump(config, config_file, indent=4, default=str)
        logger.info(f"Saved configuration in {path.resolve()}")


def get_mean_std(log_path: Path, params: dict) -> dict:
    """
    Retrieve the mean and std values computed during the first 'normalization
    params' step.
    :param log_path: Path to save the experiment information and model.
    :param params: Parameters to use to find the mean and std files.
    :return: A dictionary containing the mean and std values.
    """
    mean_path = log_path / params["mean"]
    if not mean_path.exists():
        raise Exception(f"No file found at {mean_path}")

    std_path = log_path / params["std"]
    if not std_path.exists():
        raise Exception(f"No file found at {std_path}")

    with mean_path.open() as f:
        mean = f.read().splitlines()
        mean = [int(value) for value in mean]

    with std_path.open() as f:
        std = f.read().splitlines()
        std = [int(value) for value in std]

    return {"mean": mean, "std": std}


def training_loaders(
    norm_params: dict, exp_data_paths: dict, global_params: dict
) -> dict:
    """
    Generate the loaders to use during the training step.
    :param norm_params: The mean and std values used during image normalization.
    :param exp_data_paths: Path to the data folders.
    :param global_params: Global parameters of the experiment entered by the used.
    :return loaders: A dictionary with the loaders.
    """
    loaders = {}
    t = tqdm(["train", "val"])
    t.set_description("Loading data")
    for set, images, masks in zip(
        t,
        [exp_data_paths["train"]["image"], exp_data_paths["val"]["image"]],
        [exp_data_paths["train"]["mask"], exp_data_paths["val"]["mask"]],
    ):
        dataset = TrainingDataset(
            images,
            masks,
            global_params["classes_colors"],
            transform=transforms.Compose(
                [
                    Rescale(global_params["img_size"]),
                    Normalize(norm_params["mean"], norm_params["std"]),
                ]
            ),
        )
        loaders[set] = DataLoader(
            dataset,
            num_workers=2,
            pin_memory=True,
            batch_sampler=Sampler(
                dataset,
                bin_size=global_params["bin_size"],
                batch_size=global_params["batch_size"],
                nb_params=global_params["no_of_params"],
            ),
            collate_fn=DLACollateFunction(),
        )
        logging.info(f"{set}: Found {len(dataset)} images")
    return loaders


def prediction_loaders(
    norm_params: dict, exp_data_paths: dict, global_params: dict
) -> dict:
    """
    Generate the loaders to use during the prediction step.
    :param norm_params: The mean and std values used during image normalization.
    :param exp_data_paths: Path to the data folders.
    :param global_params: Global parameters of the experiment entered by the used.
    :return loaders: A dictionary with the loaders.
    """
    loaders = {}
    for set, images in zip(
        ["train", "val", "test"],
        [
            exp_data_paths["train"]["image"],
            exp_data_paths["val"]["image"],
            exp_data_paths["test"]["image"],
        ],
    ):
        dataset = PredictionDataset(
            images,
            transform=transforms.Compose(
                [
                    Rescale(global_params["img_size"]),
                    Normalize(norm_params["mean"], norm_params["std"]),
                    Pad(),
                    ToTensor(),
                ]
            ),
        )
        loaders[set + "_loader"] = DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True
        )
    return loaders


def training_initialization(global_params: dict, training: dict, log_path: str) -> dict:
    """
    Initialize the training step.
    :param global_params: Global parameters of the experiment entered by the used.
    :param training: Training parameters.
    :param log_path: Path to save the experiment information and model.
    :return tr_params: A dictionary with the training parameters.
    """
    no_of_classes = len(global_params["classes_names"])
    # TODO: log number of classes on tensorboard ?
    net = model.load_network(no_of_classes, global_params["use_amp"])

    if training["restore_model"] is None:
        net.apply(model.weights_init)
        tr_params = {
            "net": net,
            "criterion": Diceloss(no_of_classes),
            "optimizer": Adam(net.parameters(), lr=global_params["learning_rate"]),
            "saved_epoch": 0,
            "best_loss": 10e5,
            "scaler": GradScaler(enabled=global_params["use_amp"]),
            "use_amp": global_params["use_amp"],
        }
    else:
        # Restore model to resume training.
        checkpoint, net, optimizer, scaler = model.restore_model(
            net,
            Adam(net.parameters(), lr=global_params["learning_rate"]),
            GradScaler(enabled=global_params["use_amp"]),
            log_path,
            training["restore_model"],
        )
        tr_params = {
            "net": net,
            "criterion": Diceloss(no_of_classes),
            "optimizer": optimizer,
            "saved_epoch": checkpoint["epoch"],
            "best_loss": checkpoint["best_loss"]
            if training["loss"] == "best"
            else 10e5,
            "scaler": scaler,
            "use_amp": global_params["use_amp"],
        }
    return tr_params


def prediction_initialization(params: dict, global_params: dict, log_path: str) -> dict:
    """
    Initialize the prediction step.
    :param params: The global parameters of the experiment.
    :param global_params: Global parameters of the experiment entered by the used.
    :param log_path: Path to save the experiment information and model.
    :return: A dictionary with the prediction parameters.
    """
    no_of_classes = len(global_params["classes_names"])
    net = model.load_network(no_of_classes, False)

    _, net, _, _ = model.restore_model(net, None, None, log_path, params.model_path)
    return net


def run(config: dict):
    """
    Main program, training a new model, using a valid configuration
    """
    if len(config["steps"]) == 0:
        logging.info("No step to run, exiting execution.")
        return

    save_config(config)

    if "normalization_params" in config["steps"]:
        normalization_params(
            config["log_path"],
            config["data_paths"],
            config["params"],
            config["img_size"],
        )

    if "train" in config["steps"] or "prediction" in config["steps"]:
        # Get the mean and std values.
        norm_params = get_mean_std(config["log_path"], config["params"])

    if "train" in config["steps"]:
        # Generate the loaders and start training.
        loaders = training_loaders(norm_params, config["data_paths"], config)
        tr_params = training_initialization(
            config, config["training"], config["log_path"]
        )
        train(
            config["params"]["model_path"],
            config["log_path"],
            config["tb_path"],
            config["no_of_epochs"],
            norm_params,
            config["classes_names"],
            loaders,
            tr_params,
        )

    if "prediction" in config["steps"]:
        # Generate the loaders and start predicting.
        loaders = prediction_loaders(norm_params, config["data_paths"], config)
        net = prediction_initialization(config["params"], config, config["log_path"])
        predict(
            config["params"]["prediction_path"],
            config["log_path"],
            config["img_size"],
            config["classes_colors"],
            config["classes_names"],
            config["save_image"],
            config["min_cc"],
            loaders,
            net,
        )

    if "evaluation" in config["steps"]:
        for set in config["data_paths"].keys():
            for dataset in config["data_paths"][set]["json"]:
                if os.path.isdir(dataset):
                    evaluate(
                        config["log_path"],
                        config["classes_names"],
                        set,
                        config["data_paths"][set]["json"],
                        str(dataset.parent.parent.name),
                        config["params"],
                    )
                else:
                    logging.info(f"{dataset} folder not found.")
