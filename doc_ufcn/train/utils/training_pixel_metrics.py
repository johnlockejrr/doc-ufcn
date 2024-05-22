"""
The training pixel metrics module
======================

Use it to compute different metrics during training.
Available metrics:
    - Confusion matrix
    - Intersection-over-Union
"""

import numpy as np


def compute_metrics(
    pred: np.ndarray, label: np.ndarray, loss: float, classes: list
) -> dict:
    """
    Compute the metrics between a prediction and a label mask.
    :param pred: The prediction made by the network.
    :param label: The mask of the input image.
    :param loss: The loss the the current batch.
    :param classes: The classes names involved during the experiment.
    :return metrics: The computed metrics.
    """
    metrics = {}
    metrics["matrix"] = confusion_matrix(pred, label, classes)
    metrics["loss"] = loss
    return metrics


def update_metrics(metrics: dict, batch_metrics: dict) -> dict:
    """
    Add batch metrics to the global metrics.
    :param metrics: The global epoch metrics.
    :param batch_metrics: The current batch metrics.
    :return metrics: The updated global metrics.
    """
    for i in range(metrics["matrix"].shape[0]):
        for j in range(metrics["matrix"].shape[1]):
            metrics["matrix"][i][j] += batch_metrics["matrix"][i][j]
    metrics["loss"] += batch_metrics["loss"]
    return metrics


def confusion_matrix(pred: np.ndarray, label: np.ndarray, classes: list) -> np.array:
    """
    Get the confusion matrix between the prediction and the given label.
    :param pred: The prediction made by the network.
    :param label: The mask of the input image.
    :params classes: The classes names involved during the experiment.
    :return confusion_matrix: The computed confusion matrix.
    """
    size = len(classes)
    confusion_matrix = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            bin_label = label == i
            bin_pred = pred == j
            confusion_matrix[j, i] = (bin_pred * bin_label).sum()
    return confusion_matrix


def iou(confusion_matrix: np.ndarray, channel: str) -> float:
    """
    Get the Intersection-over-Union values between the prediction and the
    given label. It returns one value for the given class.
    :param confusion_matrix: The confusion matrix obtained between the label
                             and prediction masks.
    :param channel: The name of the current class.
    :return: The computed Intersection-over-Union value.
    """
    true_positives = confusion_matrix[channel, channel]
    tpfn = np.sum(confusion_matrix[:, channel])
    tpfp = np.sum(confusion_matrix[channel, :])
    if true_positives == 0 or tpfn + tpfp == true_positives:
        return 0
    return true_positives / (tpfn + tpfp - true_positives)
