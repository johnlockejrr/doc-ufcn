#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    The evaluate module
    ======================

    Use it to evaluate a trained network on testing images.

    :example:

    >>> python evaluate.py with utils/testing_config.json
"""

import os
import logging
import json
import time
import torch
import cv2
import numpy as np
from tqdm import tqdm
from sacred import Experiment
import utils.metrics as m
import utils.utils as utils
from utils.data import load_data
from utils.model import load_network, restore_model
from utils.params_config import TestParams
from utils.postprocessing import post_processing

ex = Experiment('Evaluation')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


@ex.config
def default_config():
    """
    Define the default configuration for the experiment.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info('Running on %s', device)
    params = TestParams().to_dict()
    img_size = 384
    experiment_name = 'ufcn'
    log_path = 'runs/'+experiment_name.lower().replace(' ', '_').replace('-', '_')
    no_of_classes = 2
    classes_names = ["Background", "Text_line"]
    normalization_params = {"mean": [0, 0, 0], "std": [1, 1, 1]}
    min_cc = 0


@ex.capture
def initialize_experiment(classes: list, params: TestParams, img_size: int,
                          no_of_classes: int, device: str,
                          normalization_params: dict):
    """
    Initialize the experiment.
    Load the data into batches and load the network.
    :param classes: The color codes of the different classes.
    :param params: The dictionnary containing the parameters
                   of the experiment.
    :param img_size: The size in which the images will be resized.
    :param no_of_classes: The number of classes involved in the experiment.
    :param device: The device used to run the experiment.
    :param normalization_params: The mean values and standard deviations used
                                 to normalize the images.
    :return loader: The loader containing the pre-processed images.
    :return net: The loaded network.
    :return last_layer: The last activation function to apply.
    """
    params = TestParams.from_dict(params)
    loader = load_data(params.test_frame_path, img_size,
                       normalization_params['mean'],
                       normalization_params['std'], classes,
                       mask_path=params.test_mask_path, shuffle=False)
    net, last_layer = load_network(no_of_classes, device, ex)
    return loader, net, last_layer


@ex.capture
def load_one_batch(probs: np.ndarray, min_cc: int) -> list:
    """
    Predict the images and prepare the label.
    :param probs: The probabilities of the predicted batch images.
    :param min_cc: The minimum size of the small connected
                   components to remove.
    :return predictions: A list of the predictions for the current
                         batch images with the corresponding labels
                         and probabilities.
    """
    predictions = []
    for pred in range(probs.shape[0]):
        probas = probs[pred, :, :, :].cpu().numpy()
        max_probas = np.argmax(probas, axis=0)
        for channel in range(1, probas.shape[0]):
            # Keep pixels with highest probability.
            probas_channel = np.uint8(max_probas == channel) \
                                 * probas[channel, :, :]
            # Remove small connected components.
            if min_cc > 0:
                bin_img = probas_channel.copy()
                bin_img[bin_img > 0] = 1
                _, labeled_cc = cv2.connectedComponents(np.uint8(bin_img),
                                                        connectivity=8)
                for lab in np.unique(labeled_cc):
                    if np.sum(labeled_cc == lab) < min_cc:
                        labeled_cc[labeled_cc == lab] = 0
                probas_channel = probas_channel * (labeled_cc > 0)
            probas[channel, :, :] = probas_channel
        # Generate prediction with post processing.
        predictions.append({
            'probas': probas,
            'prediction': np.argmax(probas, axis=0),
            'prediction_pp': post_processing(probas, eval_step=True),
        })
    return predictions


@ex.automain
def run(params: TestParams, device: str, log_path: str, no_of_classes: int,
        classes_names: list):
    """
    Main function. Run the evaluation over all the images.
    :param params: The dictionnary containing the parameters
                   of the experiment.
    :param device: The device used to run the experiment.
    :param log_path: The directory to save the predictions for
                     the current experiment.
    :param no_of_classes: The number of classes involved in the experiment.
    :param classes_names: The names of the classes involved in the experiment.
    """
    params = TestParams.from_dict(params)
    # Get the possible colors.
    colors = utils.get_classes_colors(params.classes_file)

    test_loader, net, last_layer = initialize_experiment(colors)
    net = restore_model(net, log_path, params.model_path)
    net.eval()
    # Create folder to save the results.
    os.makedirs(os.path.join(log_path, params.results_dir), exist_ok=True)

    logging.info('Starting evaluation')
    starting_time = time.time()
    # Initialize results objects.
    confusion_matrix = np.zeros((no_of_classes, no_of_classes))
    results = {
        'pixel': {name: {channel: [] for channel in classes_names}
                  for name in ['iou', 'precision', 'recall', 'fscore']},
        'object': {name: {channel: [] for channel in classes_names[1:]}
                   for name in ['precision', 'recall', 'fscore', 'AP']},
    }
    rank_scores = {
        channel: {
            iou: {
                rank: {'True': 0, 'Total': 0} for rank in range(95, 45, -5)
            } for iou in range(50, 100, 5)
        } for channel in classes_names[1:]}
    positives_examples = {channel: 0 for channel in classes_names[1:]}
    # Evaluate the predictions.
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Evaluating"):
            image, label = data['image'].to(device), data['label'].to(device)
            probas = last_layer(net(image.float()))

            for index, prediction in enumerate(load_one_batch(probas)):
                label = label[index, :, :].cpu().numpy()
                pixel_m = m.PixelMetrics(prediction['prediction'],
                                         label, classes_names)
                object_m = m.ObjectMetrics(prediction['prediction'],
                                           label, classes_names,
                                           prediction['probas'])

                results['pixel'], confusion_matrix = pixel_m.update_results(
                    results['pixel'], confusion_matrix)
                rank_scores, positives_examples = object_m.update_rank_scores(
                    rank_scores, positives_examples)

    # Get the mean results over the test set.
    results['pixel'] = pixel_m.get_mean_results(results['pixel'])
    results['object'] = object_m.get_mean_results(rank_scores,
                                                  positives_examples,
                                                  results['object'])
    # Print the results.
    for item, value in results['pixel'].items():
        print('  '+item, value)
    for channel in classes_names[1:]:
        aps = results['object']['AP'][channel]
        print(channel.lower()+'  AP [0.5,0.95] = ',
              np.round(np.mean(list(aps.values())), 4))
        print(channel.lower()+'  AP [IOU=0.50] = ', np.round(aps[50], 4))
        print(channel.lower()+'  AP [IOU=0.75] = ', np.round(aps[75], 4))
        print(channel.lower()+'  AP [IOU=0.95] = ', np.round(aps[95], 4))
        print('\n')
    # Save the results : confusion matrix / json file / curves.
    utils.plot_matrix(confusion_matrix,
                      os.path.join(log_path, params.results_dir),
                      classes_names)
    utils.save_results(results, classes_names,
                       os.path.join(log_path, params.results_dir))
    with open(os.path.join(log_path, params.results_dir,
                           'results.json'), 'w') as json_file:
        json.dump(results, json_file, indent=4)

    end = time.gmtime(time.time() - starting_time)
    logging.info('Finished evaluating in %2d:%2d:%2d',
                 end.tm_hour, end.tm_min, end.tm_sec)
