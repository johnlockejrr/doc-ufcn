"""
The training utils module
======================

Use it to during the training stage.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import doc_ufcn.train.utils.training_pixel_metrics as p_metrics

# Useful functions.


class Diceloss(nn.Module):
    """
    The Diceloss class is used during training.
    """

    def __init__(self, num_classes: int):
        """
        Constructor of the Diceloss class.
        :param num_classes: The number of classes involved in the experiment.
        """
        super().__init__()
        self.num_classes = num_classes

    def forward(self, pred: np.ndarray, target: np.ndarray) -> float:
        """
        Compute the Dice loss between a label and a prediction mask.
        :param pred: The prediction made by the network.
        :param target: The label mask.
        :return: The Dice loss.
        """
        label = (
            nn.functional.one_hot(target, num_classes=self.num_classes)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        smooth = 1.0
        iflat = pred.contiguous().view(-1)
        tflat = label.contiguous().view(-1)
        intersection = (iflat * tflat).sum()
        A_sum = torch.sum(iflat * iflat)
        B_sum = torch.sum(tflat * tflat)
        return 1 - ((2.0 * intersection + smooth) / (A_sum + B_sum + smooth))


# Plot the prediction during training.


def plot_prediction(output: np.ndarray) -> np.ndarray:
    """
    Transform the output of the network into an array of categorical
    predictions.
    :param output: The predictions of the batch images.
    :return prediction: The array of categorical predictions.
    """
    prediction = np.zeros((output.shape[0], 1, output.shape[2], output.shape[3]))
    for pred in range(output.shape[0]):
        current_pred = output[pred, :, :, :]
        new = np.argmax(current_pred, axis=0)
        new = np.expand_dims(new, axis=0)
        prediction[pred, :, :, :] = new
    return prediction


def display_training(
    output: np.ndarray,
    image: np.ndarray,
    label: np.ndarray,
    writer,
    epoch: int,
    norm_params: list,
):
    """
    Define the figure to plot a batch images, labels and current predictions.
    Add it to Tensorboard.
    :param output: The predictions of the batch images.
    :param image: The current batch images.
    :param label: The corresponding labels.
    :param writer: The Tensorboard writer to add the figure.
    :param epoch: The current epoch.
    :param norm_params: The mean values and standard deviations used
                        to normalize the images.
    """
    predictions = plot_prediction(output.cpu().detach().numpy())
    fig, axs = plt.subplots(
        predictions.shape[0],
        3,
        figsize=(10, 3 * predictions.shape[0]),
        gridspec_kw={"hspace": 0.2, "wspace": 0.05},
    )
    for pred in range(predictions.shape[0]):
        current_input = image.cpu().numpy()[pred, :, :, :]
        current_input = current_input.transpose((1, 2, 0))
        for channel in range(current_input.shape[2]):
            current_input[:, :, channel] = (
                current_input[:, :, channel] * norm_params["std"][channel]
            ) + norm_params["mean"][channel]
        if predictions.shape[0] > 1:
            axs[pred, 0].imshow(current_input.astype(np.uint8))
            axs[pred, 1].imshow(label.cpu()[pred, :, :], cmap="gray")
            axs[pred, 2].imshow(predictions[pred, 0, :, :], cmap="gray")
        else:
            axs[0].imshow(current_input.astype(np.uint8))
            axs[1].imshow(label.cpu()[pred, :, :], cmap="gray")
            axs[2].imshow(predictions[pred, 0, :, :], cmap="gray")
    _ = [axi.set_axis_off() for axi in axs.ravel()]
    writer.add_figure("Image_Label_Prediction", fig, global_step=epoch)


# Display the metrics during training.


def get_epoch_values(metrics: dict, classes: list, batch: int) -> dict:
    """
    Get the metrics of an epoch.
    :param metrics: The metrics to get.
    :param classes: The classes names involved in the experiment.
    :param batch: The current batch.
    :return values: The computed epoch values.
    """
    values = {}
    for channel in classes[1:]:
        values[f"iou_{channel}"] = round(
            p_metrics.iou(metrics["matrix"], classes.index(channel)), 6
        )
    values["loss"] = metrics["loss"] / batch
    return values
