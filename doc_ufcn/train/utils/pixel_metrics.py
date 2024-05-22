"""
The pixel metrics module
======================

Use it to compute different metrics during evaluation.
Available metrics:
    - Intersection-over-Union
    - Precision
    - Recall
    - F-score
"""

import numpy as np


def compute_metrics(
    labels: list, predictions: list, classes: list, global_metrics: dict
) -> dict:
    """
    Compute the pixel level metrics between prediction and label areas of
    a given page.
    :param labels: The label's polygons.
    :param predictions: The predicted polygons.
    :param classes: The classes names involved in the experiment.
    :param global_metrics: The initialized results dictionary.
    :return global_metrics: The updated results dictionary.
    """
    for channel in classes:
        inter = 0
        for _, gt in labels[channel]:
            for _, pred in predictions[channel]:
                inter += gt.intersection(pred).area
        gt_area = np.sum([gt.area for _, gt in labels[channel]])
        pred_area = np.sum([pred.area for _, pred in predictions[channel]])

        global_metrics[channel]["iou"].append(get_iou(inter, gt_area, pred_area))
        precision = get_precision(inter, pred_area)
        recall = get_recall(inter, gt_area)
        global_metrics[channel]["precision"].append(precision)
        global_metrics[channel]["recall"].append(recall)
        if precision + recall != 0:
            global_metrics[channel]["fscore"].append(
                2 * precision * recall / (precision + recall)
            )
        else:
            global_metrics[channel]["fscore"].append(0)
    return global_metrics


def get_iou(intersection: float, label_area: float, predicted_area: float) -> float:
    """
    Get the Intersection-over-Union value between prediction and label areas of
    a given page.
    :param intersection: Area of the intersection.
    :param label_area: Area of the label objects.
    :param predicted_area: Area of the predicted objects.
    :return: The computed Intersection-over-Union value.
    """
    union = label_area + predicted_area - intersection
    # Nothing to detect and nothing predicted.
    if label_area == 0 and predicted_area == 0:
        return 1
    # Objects to detect and/or predicted but no intersection.
    if intersection == 0 and union != 0:
        return 0
    # Objects to detect and/or predicted that intersect.
    return intersection / union


def get_precision(intersection: float, predicted_area: float) -> float:
    """
    Get the precision between prediction and label areas of
    a given page.
    :param intersection: Area of the intersection.
    :param predicted_area: Area of the predicted objects.
    :return: The computed precision value.
    """
    # Nothing predicted.
    if predicted_area == 0:
        return 1
    return intersection / predicted_area


def get_recall(intersection, label_area) -> float:
    """
    Get the recall between prediction and label areas of
    a given page.
    :param intersection: Area of the intersection.
    :param label_area: Area of the label objects.
    :return: The computed recall value.
    """
    # Nothing to detect.
    if label_area == 0:
        return 1
    return intersection / label_area
