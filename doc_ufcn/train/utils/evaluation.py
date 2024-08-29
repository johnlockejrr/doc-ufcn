"""
The evaluation utils module
======================

Use it to during the evaluation stage.
"""

import json

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from shapely.geometry import Polygon


def resize_polygons(polygons: dict, gt_size: tuple, pred_size: tuple) -> dict:
    """
    Resize the detected polygons to the original image size.
    :param polygons: The polygons to resize.
    :param gt_size: The ground truth image size.
    :param pred_size: The prediction image size.
    :return polygons: The resized detected polygons.
    """
    # Compute resizing ratio
    ratio = [gt / pred for gt, pred in zip(gt_size, pred_size, strict=True)]

    for channel in polygons:
        if channel == "img_size":
            continue
        for index, polygon in enumerate(polygons[channel]):
            x_points = [int(element[1] * ratio[0]) for element in polygon["polygon"]]
            y_points = [int(element[0] * ratio[1]) for element in polygon["polygon"]]

            x_points = [
                int(element) if element < gt_size[0] else int(gt_size[0])
                for element in x_points
            ]
            y_points = [
                int(element) if element < gt_size[1] else int(gt_size[1])
                for element in y_points
            ]
            x_points = [int(element) if element > 0 else 0 for element in x_points]
            y_points = [int(element) if element > 0 else 0 for element in y_points]

            assert max(x_points) <= gt_size[0]
            assert min(x_points) >= 0
            assert max(y_points) <= gt_size[1]
            assert min(y_points) >= 0
            polygons[channel][index]["polygon"] = list(
                zip(y_points, x_points, strict=True)
            )
    return polygons


def get_polygons(regions: dict, classes: list) -> dict:
    """
    Retrieve the polygons from a read json file.
    :param regions: Regions extracted from a label / prediction file.
    :param classes: The classes names involved in the experiment.
    :return polys: A dictionary with retrieved polygons and the
                   corresponding confidence scores.
    """
    polys = {}
    for channel in classes[1:]:
        if channel in regions:
            polys[channel] = [
                (polygon["confidence"], Polygon(polygon["polygon"]).buffer(0))
                for polygon in regions[channel]
            ]
    return polys


# Save the metrics.


def save_results(
    pixel_results: dict, object_results: dict, classes: list, path: str, dataset: str
):
    """
    Save the pixel and object results into a json file.
    :param pixel_results: The results obtained at pixel level.
    :param object_results: The results obtained at object level.
    :param classes: The classes names involved in the experiment.
    :param path: The path to the results directory.
    :param dataset: The name of the current dataset.
    """
    json_dict = {channel: {} for channel in classes}

    for channel in classes:
        json_dict[channel]["iou"] = np.round(np.mean(pixel_results[channel]["iou"]), 4)
        json_dict[channel]["precision"] = np.round(
            np.mean(pixel_results[channel]["precision"]), 4
        )
        json_dict[channel]["recall"] = np.round(
            np.mean(pixel_results[channel]["recall"]), 4
        )
        json_dict[channel]["fscore"] = np.round(
            np.mean(pixel_results[channel]["fscore"]), 4
        )
        aps = object_results[channel]["AP"]
        json_dict[channel]["AP@[.5]"] = np.round(aps[50], 4)
        json_dict[channel]["AP@[.75]"] = np.round(aps[75], 4)
        json_dict[channel]["AP@[.95]"] = np.round(aps[95], 4)
        json_dict[channel]["AP@[.5,.95]"] = np.round(np.mean(list(aps.values())), 4)

    (path / f"{dataset}_results.json").write_text(json.dumps(json_dict, indent=4))


def save_graphical_results(results: dict, classes: list, path: str):
    """
    Plot various curves involving the computed metrics.
    :param results: The dictionary containing the computed object metrics.
    :param classes: The classes names involved in the experiment.
    :param path: The path to the results directory.
    """
    plot_precision_recall_curve(results, classes, path)
    plot_rank_score(results, classes, "Precision", path)
    plot_rank_score(results, classes, "Recall", path)
    plot_rank_score(results, classes, "F-score", path)


def generate_figure(params: dict, rotation: bool | None = None):
    """
    Generic function to generate a figure.
    :param params: A dictionary containing the useful information
                   the initialize a figure.
    :param rotation: A boolean indicating whether to rotate the y-axis labels.
    :return fig: The initialized figure.
    :return axis: The created axis to plot.
    :return fp_light: The loaded font property used in the figures.
    """
    fp_light = fm.FontProperties(fname="./resources/font/Quicksand-Light.ttf", size=11)
    fp_medium = fm.FontProperties(
        fname="./resources/font/Quicksand-Medium.ttf", size=11
    )
    fig = plt.figure(figsize=params["size"])
    axis = fig.add_subplot(111)
    axis.set_xlabel(params["xlabel"], fontproperties=fp_light)
    axis.set_ylabel(params["ylabel"], fontproperties=fp_light)
    axis.set_xticklabels(params["xticks"], fontproperties=fp_light)
    axis.set_yticklabels(
        params["yticks"], fontproperties=fp_light, rotation=rotation, va="center"
    )
    plt.title(params["title"], fontproperties=fp_medium, fontsize=16, pad=20)
    return fig, axis, fp_light


def plot_rank_score(scores: dict, classes: list, metric: str, path: str):
    """
    Plot the scores according to the ranks on confidence score and for
    different IoU thresholds.
    :param scores: The computed scores to plot.
    :param classes: The involved classes names.
    :param metric: The current metric to plot.
    :param path: The path to the results directory.
    """
    params = {
        "size": (12, 8),
        "title": f"{metric} vs. confidence score for various IoU thresholds",
        "xlabel": "Confidence score",
        "ylabel": metric,
        "xticks": [0, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
        "yticks": [0, 0.2, 0.4, 0.6, 0.8, 1],
    }
    colors = plt.cm.RdPu(np.linspace(0.2, 1, 10))
    for channel in classes:
        _, axis, fp_light = generate_figure(params)
        axis.grid(color="grey", alpha=0.2)
        axis.xaxis.set_major_locator(ticker.MultipleLocator(5))
        for index, iou in enumerate(range(50, 100, 5)):
            if metric == "Precision":
                score = list(scores[channel]["precision"][iou].values())
                rank = list(scores[channel]["precision"][iou])
            if metric == "Recall":
                score = list(scores[channel]["recall"][iou].values())
                rank = list(scores[channel]["recall"][iou])
            if metric == "F-score":
                score = list(scores[channel]["fscore"][iou].values())
                rank = list(scores[channel]["fscore"][iou])
            axis.plot(
                rank,
                score,
                label=f"{iou / 100:.2f}",
                alpha=1,
                color=colors[index],
                linewidth=2,
            )
            axis.scatter(
                rank,
                score,
                color=colors[index],
                facecolors="none",
                linewidth=1,
                marker="o",
            )
        axis.set_xlim([49, 96])
        axis.set_ylim([0, 1])
        plt.legend(prop=fp_light, loc="lower left")
        plt.savefig(path / f"{metric}_{channel}.png", bbox_inches="tight")


def plot_precision_recall_curve(object_metrics: dict, classes: list, path: str):
    """
    Plot the precision-recall curve for different IoU thresholds.
    :param object_metrics: The computed precisions and recalls to plot.
    :param classes: The involved classes names.
    :param path: The path to the results directory.
    """
    params = {
        "size": (12, 8),
        "title": "Precision-recall curve for various IoU thresholds",
        "xlabel": "Recall",
        "ylabel": "Precision",
        "xticks": [0, 0.2, 0.4, 0.6, 0.8, 1],
        "yticks": [0, 0.2, 0.4, 0.6, 0.8, 1],
    }
    colors = plt.cm.RdPu(np.linspace(0.2, 1, 10))
    for channel in classes:
        _, axis, fp_light = generate_figure(params)
        axis.grid(color="grey", alpha=0.2)
        for index, iou in enumerate(range(50, 100, 5)):
            current_pr = list(object_metrics[channel]["precision"][iou].values())
            current_rec = list(object_metrics[channel]["recall"][iou].values())
            axis.plot(
                current_rec,
                current_pr,
                label=f"{iou / 100:.2f}",
                alpha=1,
                color=colors[index],
                linewidth=2,
            )
            axis.scatter(
                current_rec,
                current_pr,
                color=colors[index],
                facecolors="none",
                linewidth=1,
                marker="o",
            )
        axis.set_xlim([0, 1])
        axis.set_ylim([0, 1])
        plt.legend(prop=fp_light, loc="lower right")
        plt.savefig(
            path / f"Precision-recall_{channel}.png",
            bbox_inches="tight",
        )
