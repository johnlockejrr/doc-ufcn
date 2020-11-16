#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    The utils module
    ======================

"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.font_manager as fm
import imageio as io

# Useful functions.


def rgb_to_gray_value(rgb: tuple) -> int:
    """
    Compute the gray value of a RGB tuple.
    :param rgb: The RGB value to transform.
    :return: The corresponding gray value.
    """
    try:
        return int(rgb[0] * 0.299 + rgb[1] * 0.587 + rgb[2] * 0.114)
    except TypeError:
        return int(int(rgb[0]) * 0.299 + int(rgb[1]) * 0.587 +
                   int(rgb[2]) * 0.114)


def rgb_to_gray_array(rgb: np.ndarray) -> np.ndarray:
    """
    Compute the gray array (NxM) of a RGB array (NxMx3).
    :param rgb: The RGB array to transform.
    :return: The corresponding gray array.
    """
    gray_array = rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 \
        + rgb[:, :, 2] * 0.114
    return np.uint8(gray_array)


def get_classes_colors(filename: str) -> list:
    """
    Read the classes.txt file and get the color tuples.
    :param filename: The name of the file that contains the colors.
    :return: The list containing the color tuples.
    """
    # Get the color codes from the classes.txt file.
    with open(filename) as classes_file:
        return [tuple(element[:-1].split(' '))
                for element in classes_file.readlines()]

def back_to_original_size(prediction, size):
    """
    Resize the predicted channel into the original input image size.
    Compute the size of the prediction in the padded prediction.
    Extract and resize the prediction into the real image size.
    :param prediction: The predicted image to resize.
    :param size: The original input image size.
    :param interpolation: The interpolation to use when resizing the image.
    :return: The resized predicted channel.
    """
    prediction = prediction.cpu().numpy()
    # Resize
    input_size = prediction.shape[-1]
    # Compute the new sizes.
    ratio = float(input_size) / max(size)
    new_size = tuple([int(x * ratio) for x in size])
    # Extract the small image.
    delta_w = input_size - new_size[1]
    delta_h = input_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    resized_image = np.zeros(
        (prediction.shape[2] - bottom - top, prediction.shape[3] - right - left)
    )
    full_size_image = np.zeros((prediction.shape[0], prediction.shape[1], size[0], size[1]))
    for channel in range(prediction.shape[1]):
        resized_image[:, :] = prediction[0, channel,
            top : prediction.shape[2] - bottom,
            left : prediction.shape[3] - right
        ]
        full_size_image[0, channel, :, :] = cv2.resize(resized_image, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
    return full_size_image

# Plot the prediction during training.


def plot_prediction(output: np.ndarray) -> np.ndarray:
    """
    Transform the output of the network into an array of categorical
    predictions.
    :param output: The predictions of the batch images.
    :return prediction: The array of categorical predictions.
    """
    prediction = np.zeros((output.shape[0], 1, output.shape[2],
                           output.shape[3]))
    for pred in range(output.shape[0]):
        current_pred = output[pred, :, :, :]
        new = np.argmax(current_pred, axis=0)
        new = np.expand_dims(new, axis=0)
        prediction[pred, :, :, :] = new
    return prediction


def display_training(output, image, label: np.ndarray, writer, epoch: int,
                     norm_params: list):
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
    fig, axs = plt.subplots(predictions.shape[0], 3,
                            figsize=(10, 3*predictions.shape[0]),
                            gridspec_kw={'hspace': 0.2, 'wspace': 0.05})
    for pred in range(predictions.shape[0]):
        current_input = image.cpu().numpy()[pred, :, :, :]
        current_input = current_input.transpose((1, 2, 0))
        for channel in range(current_input.shape[2]):
            current_input[:, :, channel] = (current_input[:, :, channel]
                                            * norm_params['std'][channel]) \
                                            + norm_params['mean'][channel]
        if predictions.shape[0] > 1:
            axs[pred, 0].imshow(current_input.astype(np.uint8))
            axs[pred, 1].imshow(label.cpu()[pred, :, :], cmap='gray')
            axs[pred, 2].imshow(predictions[pred, 0, :, :], cmap='gray')
        else:
            axs[0].imshow(current_input.astype(np.uint8))
            axs[1].imshow(label.cpu()[pred, :, :], cmap='gray')
            axs[2].imshow(predictions[pred, 0, :, :], cmap='gray')
    _ = [axi.set_axis_off() for axi in axs.ravel()]
    writer.add_figure('Image_Label_Prediction', fig, global_step=epoch)

# Save the prediction images.


def save_prediction(prediction: np.ndarray, filename: str):
    """
    Save a prediction image.
    :param prediction: The prediction to save.
    """
    prediction = (prediction/np.max(prediction)) * 255
    io.imsave(filename, np.uint8(prediction))

# Save the metrics.


def save_results(results: dict, classes: list, path: str):
    """
    Plot various curves involving the computed metrics.
    :param results: The dictionary containing the computed object metrics.
    :param classes: The classes names.
    :param path: The path to the results directory.
    """
    plot_precision_recall_curve(results['object'], classes, path)
    plot_rank_score(results['object']['precision'], classes, 'Precision', path)
    plot_rank_score(results['object']['recall'], classes, 'Recall', path)
    plot_rank_score(results['object']['fscore'], classes, 'F-score', path)


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
    for channel in classes[1:]:
        _, axis, fp_light = generate_figure(params)
        axis.grid(color="grey", alpha=0.2)
        axis.xaxis.set_major_locator(ticker.MultipleLocator(5))
        for index, iou in enumerate(range(50, 100, 5)):
            score = list(scores[channel][iou].values())
            rank = list(scores[channel][iou].keys())
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
    precision = object_metrics['precision']
    recall = object_metrics['recall']
    for channel in classes[1:]:
        _, axis, fp_light = generate_figure(params)
        axis.grid(color="grey", alpha=0.2)
        for index, iou in enumerate(range(50, 100, 5)):
            current_pr = list(precision[channel][iou].values())
            current_rec = list(recall[channel][iou].values())
            axis.plot(current_rec, current_pr, label="{:.2f}".format(iou/100),
                      alpha=1, color=colors[index], linewidth=2)
            axis.scatter(current_rec, current_pr, color=colors[index],
                         facecolors='none', linewidth=1, marker='o')
        axis.set_xlim([0, 1])
        axis.set_ylim([0, 1])
        plt.legend(prop=fp_light, loc="lower right")
        plt.savefig(os.path.join(path, 'Precision-recall_'+channel+'.png'),
                    bbox_inches='tight')

# Plot the confusion matrix.


def plot_matrix(matrix: np.array, path: str, classes: list):
    """
    Define the figure to plot the confusion matrix.
    :param matrix: The confusion matrix to plot.
    :param path: The path to save the figure.
    :param classes: The possible classes names present in the images.
    """
    # Define the figure.
    params = {
        'size': (2.5*len(classes), 2.5*len(classes)),
        'title': "Confusion matrix",
        'xlabel': "Groundtruth",
        'ylabel': "Prediction",
        'xticks': ['']+classes,
        'yticks': ['']+classes
    }
    fig, axis, fp_light = generate_figure(params, rotation='vertical')
    cax = axis.imshow(matrix, cmap='RdPu', alpha=0.8)
    axis.xaxis.set_major_locator(ticker.MultipleLocator(1))
    axis.yaxis.set_major_locator(ticker.MultipleLocator(1))
    cbar = fig.colorbar(cax)
    for lab in cbar.ax.yaxis.get_ticklabels():
        lab.set_font_properties(fp_light)
        lab.set_fontsize(10)
    # Display the values in the matrix.
    for (ind_i, ind_j), value in np.ndenumerate(matrix):
        axis.text(ind_j, ind_i, '{:0.1f}'.format(value),
                  fontproperties=fp_light,
                  ha='center', va='center', size=10)
    # Save the figure.
    plt.savefig(os.path.join(path, 'Confusion_matrix.png'),
                bbox_inches='tight')
