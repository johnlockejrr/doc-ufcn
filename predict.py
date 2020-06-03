#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    The predict module
    ======================

    Use it to predict some images from a trained network.

    :example:

    >>> python predict.py with utils/testing_config.json
    ...
"""

import os
import logging
import time
import torch
import cv2
import numpy as np
from tqdm import tqdm
from sacred import Experiment
from utils.data import load_data
from utils.model import load_network, restore_model
from utils.params_config import PredictionParams
from utils.utils import get_classes_colors, save_prediction
from utils.postprocessing import post_processing

ex = Experiment('Prediction')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


@ex.config
def default_config():
    """
    Define the default configuration for the experiment.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info('Running on %s', device)
    params = PredictionParams().to_dict()
    img_size = 384
    experiment_name = 'ufcn'
    log_path = 'runs/'+experiment_name.lower().replace(' ', '_').replace('-', '_')
    no_of_classes = 2
    classes_names = []
    normalization_params = {"mean": [0, 0, 0], "std": [1, 1, 1]}
    min_cc = 0


@ex.capture
def initialize_experiment(classes: list, params: PredictionParams,
                          img_size: int, no_of_classes: int, device: str,
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
    params = PredictionParams.from_dict(params)
    loader = load_data(params.test_frame_path, img_size,
                       normalization_params['mean'],
                       normalization_params['std'], classes, shuffle=False)
    net, last_layer = load_network(no_of_classes, device, ex)
    return loader, net, last_layer


@ex.capture
def predict_one_batch(probs: np.ndarray, classes: list, min_cc: int) -> list:
    """
    Save the predictions of one batch of images.
    :param probs: The probabilities of the predicted batch images.
    :param classes: The color codes of the classes.
    :param min_cc: The minimum size of the small connected
                   components to remove.
    :return predictions: A list of the predictions for the
                         current batch images.
    """
    predictions = []
    for pred in range(probs.shape[0]):
        probas = probs[pred, :, :, :].cpu().numpy()

        prediction_image = np.zeros((probas.shape[1], probas.shape[2], 3))
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
            # Generate the prediction image.
            color = [int(element) for element in classes[channel]]
            prediction_image[probas[channel, :, :] > 0] = color
        predictions.append({
            'prediction': prediction_image,
            'prediction_pp': post_processing(probas, colors=classes)
        })
    return predictions


@ex.automain
def run(params: PredictionParams, device: str, log_path: str):
    """
    Main function. Run the prediction of all the images.
    :param params: The dictionnary containing the parameters
                   of the experiment.
    :param device: The device used to run the experiment.
    :param log_path: The directory to save the predictions for
                     the current experiment.
    """
    params = PredictionParams.from_dict(params)
    # Get the possible colors.
    colors = get_classes_colors(params.classes_file)

    test_loader, net, last_layer = initialize_experiment(colors)
    net = restore_model(net, log_path, params.model_path)
    net.eval()
    # Create folder to save the predictions.
    os.makedirs(os.path.join(log_path, params.predictions_path),
                exist_ok=True)
    os.makedirs(os.path.join(log_path, params.predictions_pp_path),
                exist_ok=True)

    logging.info('Starting predictions')
    starting_time = time.time()
    filenames = os.listdir(params.test_frame_path)
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader, desc="Predicting"), 0):
            image = data['image'].to(device)
            probas = last_layer(net(image.float()))
            for prediction in predict_one_batch(probas, colors):
                save_prediction(prediction['prediction'],
                                os.path.join(log_path,
                                             params.predictions_path,
                                             filenames[i]))
                save_prediction(prediction['prediction_pp'],
                                os.path.join(log_path,
                                             params.predictions_pp_path,
                                             filenames[i]))

    end = time.gmtime(time.time() - starting_time)
    logging.info('Finished predicting in %2d:%2d:%2d',
                 end.tm_hour, end.tm_min, end.tm_sec)
