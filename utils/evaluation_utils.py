#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    The evaluation utils module
    ======================
    
    Use it to during the evaluation stage.
"""

import os
import cv2
import json
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from matplotlib import ticker
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def read_json(filename: str) -> dict:
    """
    Read a label / prediction json file.
    :param filename: Path to the file to read.
    :return: A dictionary with the file content.
    """
    with open(filename, 'r') as file:
        return json.load(file)


def get_polygons(regions: dict, classes: list) -> dict:
    """
    Retrieve the polygons from a read json file.
    :param regions: Regions extracted from a label / prediction file.
    :param classes: The classes names involved in the experiment.
    :return polys: A dictionary with retrieved polygons and the
                   corresponding confidence scores.
    """
    polys = {}
    for index, channel in enumerate(classes[1:], 1):
        if channel in regions.keys():
            polys[channel] = [(polygon['confidence'], Polygon(polygon['polygon']).buffer(0))
                              for polygon in regions[channel]]
    return polys

# Save the metrics.


def save_results(pixel_results: dict, object_results: dict, classes: list, path: str):
    """
    Save the pixel and object results into a json file.
    :param pixel_results: The results obtained at pixel level.
    :param object_results: The results obtained at object level.
    :param classes: The classes names involved in the experiment.
    :param path: The path to the results directory.
    """
    json_dict = {channel: {} for channel in classes}
    
    for channel in classes:
        json_dict[channel]['iou'] = np.round(np.mean(pixel_results[channel]['iou']), 4)
        json_dict[channel]['precision'] = np.round(np.mean(pixel_results[channel]['precision']), 4)
        json_dict[channel]['recall'] = np.round(np.mean(pixel_results[channel]['recall']), 4)
        json_dict[channel]['fscore'] = np.round(np.mean(pixel_results[channel]['fscore']), 4)
        aps = object_results[channel]['AP']
        json_dict[channel]['AP@[.5]'] = np.round(aps[50], 4)
        json_dict[channel]['AP@[.75]'] = np.round(aps[75], 4)
        json_dict[channel]['AP@[.95]'] = np.round(aps[95], 4)
        json_dict[channel]['AP@[.5,.95]'] = np.round(np.mean(list(aps.values())), 4)

    with open(os.path.join(path, 'results.json'), 'w') as json_file:
        json.dump(json_dict, json_file, indent=4)

def save_graphical_results(results: dict, classes: list, path: str):
    """
    Plot various curves involving the computed metrics.
    :param results: The dictionary containing the computed object metrics.
    :param classes: The classes names involved in the experiment.
    :param path: The path to the results directory.
    """
    plot_precision_recall_curve(results, classes, path)
    plot_rank_score(results, classes, 'Precision', path)
    plot_rank_score(results, classes, 'Recall', path)
    plot_rank_score(results, classes, 'F-score', path)

def generate_figure(params: dict, rotation: bool = None):
    """
    Generic function to generate a figure.
    :param params: A dictionary containing the useful information
                   the initialize a figure.
    :param rotation: A boolean indicating whether to rotate the y-axis labels.
    :return fig: The initialized figure.
    :return axis: The created axis to plot.
    :return fp_light: The loaded font property used in the figures.
    """
    fp_light = fm.FontProperties(fname='./utils/font/Quicksand-Light.ttf',
                                 size=11)
    fp_medium = fm.FontProperties(fname='./utils/font/Quicksand-Medium.ttf',
                                  size=11)
    fig = plt.figure(figsize=params['size'])
    axis = fig.add_subplot(111)
    axis.set_xlabel(params['xlabel'], fontproperties=fp_light)
    axis.set_ylabel(params['ylabel'], fontproperties=fp_light)
    axis.set_xticklabels(params['xticks'], fontproperties=fp_light)
    axis.set_yticklabels(params['yticks'], fontproperties=fp_light,
                         rotation=rotation, va="center")
    plt.title(params['title'], fontproperties=fp_medium, fontsize=16, pad=20)
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
        'size': (12, 8),
        'title': metric+" vs. confidence score for various IoU thresholds",
        'xlabel': "Confidence score",
        'ylabel': metric,
        'xticks': [0, 0.50, 0.55, 0.60, 0.65, 0.70,
                   0.75, 0.80, 0.85, 0.90, 0.95],
        'yticks': [0, 0.2, 0.4, 0.6, 0.8, 1]
    }
    colors = plt.cm.RdPu(np.linspace(0.2, 1, 10))
    for channel in classes:
        _, axis, fp_light = generate_figure(params)
        axis.grid(color="grey", alpha=0.2)
        axis.xaxis.set_major_locator(ticker.MultipleLocator(5))
        for index, iou in enumerate(range(50, 100, 5)):
            if metric == 'Precision':
                score = list(scores[channel]['precision'][iou].values())
                rank = list(scores[channel]['precision'][iou].keys())
            if metric == 'Recall':
                score = list(scores[channel]['recall'][iou].values())
                rank = list(scores[channel]['recall'][iou].keys())
            if metric == 'F-score':
                score = list(scores[channel]['fscore'][iou].values())
                rank = list(scores[channel]['fscore'][iou].keys())
            axis.plot(rank, score, label="{:.2f}".format(iou/100),
                      alpha=1, color=colors[index], linewidth=2)
            axis.scatter(rank, score, color=colors[index],
                         facecolors='none', linewidth=1, marker='o')
        axis.set_xlim([49, 96])
        axis.set_ylim([0, 1])
        plt.legend(prop=fp_light, loc="lower left")
        plt.savefig(os.path.join(path, metric+'_'+channel+'.png'),
                    bbox_inches='tight')


def plot_precision_recall_curve(object_metrics: dict, classes: list,
                                path: str):
    """
    Plot the precision-recall curve for different IoU thresholds.
    :param object_metrics: The computed precisions and recalls to plot.
    :param classes: The involved classes names.
    :param path: The path to the results directory.
    """
    params = {
        'size': (12, 8),
        'title': "Precision-recall curve for various IoU thresholds",
        'xlabel': "Recall",
        'ylabel': "Precision",
        'xticks': [0, 0.2, 0.4, 0.6, 0.8, 1],
        'yticks': [0, 0.2, 0.4, 0.6, 0.8, 1]
    }
    colors = plt.cm.RdPu(np.linspace(0.2, 1, 10))
    for channel in classes:
        _, axis, fp_light = generate_figure(params)
        axis.grid(color="grey", alpha=0.2)
        for index, iou in enumerate(range(50, 100, 5)):
            current_pr = list(object_metrics[channel]['precision'][iou].values())
            current_rec = list(object_metrics[channel]['recall'][iou].values())
            axis.plot(current_rec, current_pr, label="{:.2f}".format(iou/100),
                      alpha=1, color=colors[index], linewidth=2)
            axis.scatter(current_rec, current_pr, color=colors[index],
                         facecolors='none', linewidth=1, marker='o')
        axis.set_xlim([0, 1])
        axis.set_ylim([0, 1])
        plt.legend(prop=fp_light, loc="lower right")
        plt.savefig(os.path.join(path, 'Precision-recall_'+channel+'.png'),
                    bbox_inches='tight')
