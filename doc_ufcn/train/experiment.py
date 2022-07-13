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
import sys
from pathlib import Path

from sacred import Experiment
from sacred.observers import MongoObserver
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
from doc_ufcn.train.utils.params import Params
from doc_ufcn.train.utils.preprocessing import (
    Normalize,
    Pad,
    PredictionDataset,
    Rescale,
    ToTensor,
    TrainingDataset,
)
from doc_ufcn.train.utils.training import Diceloss

STEPS = ["normalization_params", "train", "prediction", "evaluation"]

ex = Experiment("Doc-UFCN")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
mongo_url = "mongodb://user:password@omniboard.vpn/sacred"


@ex.config
def default_config():
    """
    Define the default configuration for the experiment.
    :classes_names: The names of the classes involved during the experiment.
    :classes_colors: The classes color codes.
    :img_size: Network input size.
    :no_of_epochs: Total number of epochs to run.
    :batch_size: Size of the batch to use during training.
    :no_of_params: Maximum number of parameters the system can
                   support (used to automatically compute batch size).
    :bin_size: Step (in pixels) between two groups of images.
    :learning_rate: Initial learning rate to use during training.
    :omniboard: Whether to use Omniboard.
    :min_cc: Threshold used to remove small connected components.
    :save_image: List of sets (train, val, test) for which the prediction images
                 are generated and saved.
    :use_amp: Whether to use Automatic Mixed Precision.
    :params: Parameters to use during all the experiment steps.
    :experiment_name: The name of the experiment that is used to save all
                      the experiment information.
    :log_path: Path to save the experiment information and model.
    :tb_path: Path to save the Tensorboard events.
    :steps: List of the steps to run.
    :data_paths: Path to the experiment data folders.
    :restore_model: Path to the model to restore to resume a training.
    :loss: Indicates whether to use 'initial' or 'best' saved loss.
    """
    # Load the global experiments parameters from experiments_config.json.
    global_params = {
        "classes_names": ["background", "text_line"],
        "classes_colors": [[0, 0, 0], [0, 0, 255]],
        "img_size": 768,
        "no_of_epochs": 100,
        "batch_size": None,
        "no_of_params": None,
        "bin_size": 20,
        "learning_rate": 5e-3,
        "omniboard": False,
        "min_cc": 0,
        "save_image": [],
        "use_amp": False,
    }
    assert (
        global_params["batch_size"] is not None
        or global_params["no_of_params"] is not None
    ), "Please provide a batch size or a maximum number of parameters"
    params = Params().to_dict()  # noqa: F841

    # Load the current experiment parameters.
    experiment_name = "doc-ufcn"
    log_path = "runs/" + experiment_name.lower().replace(  # noqa: F841
        " ", "_"
    ).replace("-", "_")
    tb_path = "events/"  # noqa: F841
    steps = ["normalization_params", "train", "prediction", "evaluation"]
    for step in steps:
        assert step in STEPS

    data_paths = {
        "train": {
            "image": ["./data/train/images/"],
            "mask": ["./data/train/labels/"],
            "json": ["./data/train/labels_json/"],
        },
        "val": {
            "image": ["./data/val/images/"],
            "mask": ["./data/val/labels/"],
            "json": ["./data/val/labels_json/"],
        },
        "test": {
            "image": ["./data/test/images/"],
            "json": ["./data/test/labels_json/"],
        },
    }
    exp_data_paths = {  # noqa: F841
        set: {
            key: [Path(element).expanduser() for element in value]
            for key, value in paths.items()
        }
        for set, paths in data_paths.items()
    }

    training = {"restore_model": None, "loss": "initial"}
    training["loss"] = training["loss"].lower()
    assert training["loss"] in ["initial", "best"]
    if "train" in steps and global_params["omniboard"] is True:
        ex.observers.append(MongoObserver(mongo_url))


@ex.capture
def save_config(
    log_path: str,
    experiment_name: str,
    global_params: dict,
    params: Params,
    steps: list,
    data_paths: dict,
    training: dict,
):
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
        "global_params": global_params,
        "params": params,
        "steps": steps,
        "data_paths": data_paths,
        "training": training,
    }
    with open(os.path.join(log_path, experiment_name + ".json"), "w") as config_file:
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
        logging.error("No file found at %s", os.path.join(log_path, params.mean))
        sys.exit()
    else:
        with open(os.path.join(log_path, params.mean), "r") as file:
            mean = file.read().splitlines()
            mean = [int(value) for value in mean]
    if not os.path.isfile(os.path.join(log_path, params.std)):
        logging.error("No file found at %s", os.path.join(log_path, params.std))
        sys.exit()
    else:
        with open(os.path.join(log_path, params.std), "r") as file:
            std = file.read().splitlines()
            std = [int(value) for value in std]
    return {"mean": mean, "std": std}


@ex.capture
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


@ex.capture
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


@ex.capture
def training_initialization(global_params: dict, training: dict, log_path: str) -> dict:
    """
    Initialize the training step.
    :param global_params: Global parameters of the experiment entered by the used.
    :param training: Training parameters.
    :param log_path: Path to save the experiment information and model.
    :return tr_params: A dictionary with the training parameters.
    """
    no_of_classes = len(global_params["classes_names"])
    ex.log_scalar("no_of_classes", no_of_classes)
    net = model.load_network(no_of_classes, global_params["use_amp"], ex)

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


@ex.capture
def prediction_initialization(params: dict, global_params: dict, log_path: str) -> dict:
    """
    Initialize the prediction step.
    :param params: The global parameters of the experiment.
    :param global_params: Global parameters of the experiment entered by the used.
    :param log_path: Path to save the experiment information and model.
    :return: A dictionary with the prediction parameters.
    """
    params = Params.from_dict(params)
    no_of_classes = len(global_params["classes_names"])
    net = model.load_network(no_of_classes, False, ex)

    _, net, _, _ = model.restore_model(net, None, None, log_path, params.model_path)
    return net


@ex.automain
def run(
    global_params: dict,
    params: Params,
    log_path: str,
    tb_path: str,
    steps: list,
    exp_data_paths: dict,
    training: dict,
):
    """
    Main program.
    :param global_params: Global parameters of the experiment entered by the used.
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
            normalization_params(
                log_path, exp_data_paths, params, global_params["img_size"]
            )

        if "train" in steps or "prediction" in steps:
            # Get the mean and std values.
            norm_params = get_mean_std()

        if "train" in steps:
            # Generate the loaders and start training.
            loaders = training_loaders(norm_params)
            tr_params = training_initialization()
            train(
                params.model_path,
                log_path,
                tb_path,
                global_params["no_of_epochs"],
                norm_params,
                global_params["classes_names"],
                loaders,
                tr_params,
                ex,
            )

        if "prediction" in steps:
            # Generate the loaders and start predicting.
            loaders = prediction_loaders(norm_params)
            net = prediction_initialization()
            predict(
                params.prediction_path,
                log_path,
                global_params["img_size"],
                global_params["classes_colors"],
                global_params["classes_names"],
                global_params["save_image"],
                global_params["min_cc"],
                loaders,
                net,
            )

        if "evaluation" in steps:
            for set in exp_data_paths.keys():
                for dataset in exp_data_paths[set]["json"]:
                    if os.path.isdir(dataset):
                        evaluate(
                            log_path,
                            global_params["classes_names"],
                            set,
                            exp_data_paths[set]["json"],
                            str(dataset.parent.parent.name),
                            params,
                        )
                    else:
                        logging.info(f"{dataset} folder not found.")
