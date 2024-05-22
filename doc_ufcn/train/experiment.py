"""
The run experiment module
======================

Use it to train, predict and evaluate a model.
"""

import logging
from pathlib import Path

from torch.cuda.amp import GradScaler
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from doc_ufcn import model
from doc_ufcn.models import download_model
from doc_ufcn.train.evaluate import run as evaluate
from doc_ufcn.train.mlflow_utils import start_mlflow_run
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
from doc_ufcn.utils import export_list

logger = logging.getLogger(__name__)


def get_mean_std(log_path: Path, mean_name: str, std_name: str) -> dict:
    """
    Retrieve the mean and std values computed during the first 'normalization
    params' step.
    :param log_path: Path to save the experiment information and model.
    :param mean_name: Name of the file that will contain all the mean values.
    :param std_name: Name of the file that will contain all the std values.
    :return: A dictionary containing the mean and std values.
    """
    mean_path = log_path / mean_name
    if not mean_path.exists():
        raise Exception(f"No file found at {mean_path}")

    std_path = log_path / std_name
    if not std_path.exists():
        raise Exception(f"No file found at {std_path}")

    mean = list(map(int, mean_path.read_text().splitlines()))
    std = list(map(int, std_path.read_text().splitlines()))

    return {"mean": mean, "std": std}


def training_loaders(
    norm_params: dict,
    exp_data_paths: dict,
    classes_colors: list,
    img_size: int,
    bin_size: int,
    batch_size: int,
    no_of_params: int,
    num_workers: int = 2,
) -> dict:
    """
    Generate the loaders to use during the training step.
    :param norm_params: The mean and std values used during image normalization.
    :param exp_data_paths: Path to the data folders.
    :return loaders: A dictionary with the loaders.
    """
    loaders = {}
    t = tqdm(["train", "val"])
    t.set_description("Loading data")
    for set, images, masks in zip(
        t,
        [exp_data_paths["train"]["image"], exp_data_paths["val"]["image"]],
        [exp_data_paths["train"]["mask"], exp_data_paths["val"]["mask"]],
        strict=True,
    ):
        dataset = TrainingDataset(
            images,
            masks,
            classes_colors,
            transform=transforms.Compose(
                [
                    Rescale(img_size),
                    Normalize(norm_params["mean"], norm_params["std"]),
                ]
            ),
        )
        loaders[set] = DataLoader(
            dataset,
            num_workers=num_workers,
            pin_memory=True,
            batch_sampler=Sampler(
                dataset,
                bin_size=bin_size,
                batch_size=batch_size,
                nb_params=no_of_params,
            ),
            collate_fn=DLACollateFunction(),
        )
        logging.info(f"{set}: Found {len(dataset)} images")
    return loaders


def prediction_loaders(
    norm_params: dict, exp_data_paths: dict, img_size: int, num_workers: int = 2
) -> dict:
    """
    Generate the loaders to use during the prediction step.
    :param norm_params: The mean and std values used during image normalization.
    :param exp_data_paths: Path to the data folders.
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
        strict=True,
    ):
        dataset = PredictionDataset(
            images,
            transform=transforms.Compose(
                [
                    Rescale(img_size),
                    Normalize(norm_params["mean"], norm_params["std"]),
                    Pad(),
                    ToTensor(),
                ]
            ),
        )
        loaders[f"{set}_loader"] = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
    return loaders


def training_initialization(
    training: dict,
    log_path: Path,
    classes_names: list,
    use_amp: bool,
    learning_rate: float,
) -> dict:
    """
    Initialize the training step.
    :param training: Training parameters.
    :param log_path: Path to save the experiment information and model.
    :return tr_params: A dictionary with the training parameters.
    """
    no_of_classes = len(classes_names)
    # TODO: log number of classes on tensorboard ?
    net = model.load_network(no_of_classes, use_amp)
    net.apply(model.weights_init)

    if training["restore_model"] is None:
        tr_params = {
            "net": net,
            "criterion": Diceloss(no_of_classes),
            "optimizer": Adam(net.parameters(), lr=learning_rate),
            "saved_epoch": 0,
            "best_loss": 10e5,
            "scaler": GradScaler(enabled=use_amp),
            "use_amp": use_amp,
        }
    else:
        # Restore model to resume training.
        checkpoint, net, optimizer, scaler = model.restore_model(
            net,
            Adam(net.parameters(), lr=learning_rate),
            GradScaler(enabled=use_amp),
            log_path,
            training["restore_model"],
            training["same_classes"],
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
            "use_amp": use_amp,
        }
    return tr_params


def prediction_initialization(
    model_path: Path, classes_names: list, log_path: str
) -> dict:
    """
    Initialize the prediction step.
    :param model_path: Path of the model to load in memory.
    :param log_path: Path to save the experiment information and model.
    :return: A dictionary with the prediction parameters.
    """
    no_of_classes = len(classes_names)
    net = model.load_network(no_of_classes, False)

    _, net, _, _ = model.restore_model(net, None, None, log_path, model_path)
    return net


def run(config: dict, num_workers: int = 2):
    """
    Main program, training a new model, using a valid configuration
    """
    assert len(config["steps"]) > 0, "No step to run"
    if config["mlflow"]:
        with start_mlflow_run(config["mlflow"]) as run:
            logger.info(f"Started MLflow run with ID ({run.info.run_id})")
            run_experiment(config=config, num_workers=num_workers, mlflow_logging=True)
    else:
        run_experiment(config=config, num_workers=num_workers, mlflow_logging=False)


def run_experiment(config: dict, num_workers: int = 2, mlflow_logging=False):
    """
    Main program, training a new model, using a valid configuration
    """
    assert len(config["steps"]) > 0, "No step to run"

    if "normalization_params" in config["steps"]:
        normalization_params(
            log_path=config["log_path"],
            data_paths=config["data_paths"],
            img_size=config["img_size"],
            mean_name=config["mean"],
            std_name=config["std"],
            num_workers=num_workers,
        )

    model_to_restore = config["training"].get("restore_model", "")

    if model_to_restore and not model_to_restore.endswith(".pth"):
        # Try to load a model with such a name on HuggingFace
        logger.info(f"Loading model with name {model_to_restore} from HuggingFace")
        # Store the path to the last model and its parameters
        config["training"]["restore_model"], parameters = download_model(
            name=model_to_restore
        )
        # Store mean and std values
        export_list(data=parameters["mean"], output=config["log_path"] / config["mean"])
        export_list(data=parameters["std"], output=config["log_path"] / config["std"])

    if "train" in config["steps"] or "prediction" in config["steps"]:
        # Get the mean and std values.
        norm_params = get_mean_std(config["log_path"], config["mean"], config["std"])

    if "train" in config["steps"]:
        # Generate the loaders and start training.
        loaders = training_loaders(
            norm_params=norm_params,
            exp_data_paths=config["data_paths"],
            classes_colors=config["classes_colors"],
            img_size=config["img_size"],
            bin_size=config["bin_size"],
            batch_size=config["batch_size"],
            no_of_params=config["no_of_params"],
            num_workers=num_workers,
        )

        tr_params = training_initialization(
            config["training"],
            config["log_path"],
            config["classes_names"],
            config["use_amp"],
            config["learning_rate"],
        )
        train(
            config["model_path"],
            config["log_path"],
            config["tb_path"],
            config["no_of_epochs"],
            norm_params,
            config["classes_names"],
            loaders,
            tr_params,
            mlflow_logging,
        )

    if "prediction" in config["steps"]:
        # Generate the loaders and start predicting.
        loaders = prediction_loaders(
            norm_params, config["data_paths"], config["img_size"]
        )
        net = prediction_initialization(
            config["model_path"], config["classes_names"], config["log_path"]
        )
        predict(
            config["prediction_path"],
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
        for set in config["data_paths"]:
            for dataset in config["data_paths"][set]["json"]:
                if dataset.is_dir():
                    evaluate(
                        config["log_path"],
                        config["classes_names"],
                        set,
                        config["data_paths"][set]["json"],
                        str(dataset.parent.parent.name),
                        config["prediction_path"],
                        config["evaluation_path"],
                        mlflow_logging,
                    )
                else:
                    logging.info(f"{dataset} folder not found.")
