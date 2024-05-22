"""
The train module
======================

Use it to train a model.
"""

import logging
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import doc_ufcn.train.utils.training as tr_utils
import doc_ufcn.train.utils.training_pixel_metrics as p_metrics
from doc_ufcn import model


def init_metrics(no_of_classes: int) -> dict:
    """
    Initialize the epoch metrics.
    :param no_of_classes: The number of classes involved in the experiment.
    :return: A dictionary containing the initialized metrics.
    """
    return {"matrix": np.zeros((no_of_classes, no_of_classes)), "loss": 0}


def log_metrics(epoch: int, metrics: dict, writer, step: str, mlflow_logging: bool):
    """
    Log the computed metrics to Tensorboard and Omniboard.
    :param epoch: The current epoch.
    :param metrics: The metrics to log.
    :param writer: The Tensorboard object to log information.
    :param step: String indicating whether to log training or validation metrics.
    :param mlflow_logging: Whether we should log data to MLflow.
    """
    prefixed_metrics = {f"{step}_{key}": value for key, value in metrics.items()}

    step_name = "TRAIN" if step == "Training" else "VALID"
    for tag, scalar in prefixed_metrics.items():
        writer.add_scalar(tag, scalar, epoch)
        logging.info(f"  {step_name} {epoch}: {tag}={round(scalar, 4)}")

    if mlflow_logging:
        mlflow.log_metrics(metrics=prefixed_metrics, step=epoch)


def run_one_epoch(
    loader,
    params: dict,
    writer,
    epochs: list,
    no_of_epochs: int,
    device: str,
    norm_params: dict,
    classes_names: list,
    step: str,
):
    """
    Run one epoch of training (or validation).
    :param loader: The loader containing the images and masks.
    :param params: A dictionary containing all the training parameters.
    :param writer: The Tensorboard object to log information.
    :param epochs: A list containing the current epoch number and the last saved epoch.
    :param no_of_epochs: The number of epochs to run.
    :param device: The device to run the experiment.
    :param norm_params: The mean and std values used during image normalization.
    :param classes_names: The names of the classes involved during the experiment.
    :param step: String indicating whether to run a training or validation step.
    :return params: The updated training parameters.
    :return epoch_values: The metrics computed during the epoch.
    """
    metrics = init_metrics(len(classes_names))
    epoch = epochs[0]

    t = tqdm(loader)
    step_name = "TRAIN" if step == "Training" else "VALID"
    t.set_description(f"{step_name} (prog) {epoch}/{no_of_epochs + epochs[1]}")

    for index, data in enumerate(t, 1):
        params["optimizer"].zero_grad()
        with autocast(enabled=params["use_amp"]):
            if params["use_amp"]:
                output = params["net"](data["image"].to(device).half())
            else:
                output = params["net"](data["image"].to(device).float())
            loss = params["criterion"](output, data["mask"].to(device).long())

        for pred in range(output.shape[0]):
            current_pred = np.argmax(
                output[pred, :, :, :].cpu().detach().numpy(), axis=0
            )
            current_label = data["mask"][pred, :, :].cpu().detach().numpy()
            batch_metrics = p_metrics.compute_metrics(
                current_pred, current_label, loss.item(), classes_names
            )
            metrics = p_metrics.update_metrics(metrics, batch_metrics)

        epoch_values = tr_utils.get_epoch_values(metrics, classes_names, index + 1)
        display_values = epoch_values
        display_values["loss"] = round(display_values["loss"], 4)
        t.set_postfix(values=str(display_values))

        if step == "Training":
            params["scaler"].scale(loss).backward()
            params["scaler"].step(params["optimizer"])
            params["scaler"].update()
            # Display prediction images in Tensorboard all 100 mini-batches.
            if index == 1 or index % 100 == 99:
                tr_utils.display_training(
                    output, data["image"], data["mask"], writer, epoch, norm_params
                )

    if step == "Training":
        return params, epoch_values
    else:
        return epoch_values


def run(
    model_path: str,
    log_path: Path,
    tb_path: Path,
    no_of_epochs: int,
    norm_params: dict,
    classes_names: list,
    loaders: dict,
    tr_params: dict,
    mlflow_logging: bool,
):
    """
    Run the training.
    :param model_path: The path to save the trained model.
    :param log_path: Path to save the experiment information and model.
    :param tb_path: Path to save the Tensorboard events.
    :param no_of_epochs: Total number of epochs to run.
    :param norm_params: The mean and std values used during image normalization.
    :param classes_names: The names of the classes involved during the experiment.
    :param loaders: The loaders containing the images and masks to use.
    :param tr_params: The training parameters.
    :param mlflow_logging: Whether we should log data to MLflow.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Run training.
    writer = SummaryWriter(log_path / tb_path)
    logging.info("Starting training")
    starting_time = time.time()

    for epoch in range(1, no_of_epochs + 1):
        current_epoch = epoch + tr_params["saved_epoch"]
        # Run training.
        tr_params["net"].train()
        tr_params, epoch_values = run_one_epoch(
            loaders["train"],
            tr_params,
            writer,
            [current_epoch, tr_params["saved_epoch"]],
            no_of_epochs,
            device,
            norm_params,
            classes_names,
            step="Training",
        )

        log_metrics(
            epoch=current_epoch,
            metrics=epoch_values,
            writer=writer,
            step="Training",
            mlflow_logging=mlflow_logging,
        )

        with torch.no_grad():
            # Run evaluation.
            tr_params["net"].eval()
            epoch_values = run_one_epoch(
                loaders["val"],
                tr_params,
                writer,
                [current_epoch, tr_params["saved_epoch"]],
                no_of_epochs,
                device,
                norm_params,
                classes_names,
                step="Validation",
            )
            log_metrics(
                current_epoch,
                epoch_values,
                writer,
                step="Validation",
                mlflow_logging=mlflow_logging,
            )
            # Keep best model.
            if epoch_values["loss"] < tr_params["best_loss"]:
                tr_params["best_loss"] = epoch_values["loss"]
                model.save_model(
                    current_epoch + 1,
                    tr_params["net"].state_dict(),
                    epoch_values["loss"],
                    tr_params["optimizer"].state_dict(),
                    tr_params["scaler"].state_dict(),
                    (log_path / model_path).absolute(),
                )
                logging.info("Best model (epoch %d) saved", current_epoch)

    # Save last model.
    path = str(log_path / f"last_{model_path}").replace("model", "model_0")
    index = 1
    while Path(path).exists():
        path = path.replace(str(index - 1), str(index))
        index += 1

    model.save_model(
        current_epoch,
        tr_params["net"].state_dict(),
        epoch_values["loss"],
        tr_params["optimizer"].state_dict(),
        tr_params["scaler"].state_dict(),
        path,
    )
    logging.info("Last model (epoch %d) saved", current_epoch)

    end = time.gmtime(time.time() - starting_time)
    logging.info(
        "Finished training in %2d:%2d:%2d", end.tm_hour, end.tm_min, end.tm_sec
    )
