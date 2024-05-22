"""
The predict module
======================

Use it to predict some images from a trained network.
"""

import logging
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

import doc_ufcn.train.utils.prediction as pr_utils


def get_predicted_polygons(
    probas: np.ndarray, min_cc: int, classes_names: list
) -> dict:
    """
    Clean the predicted and retrieve the detected object coordinates.
    :param probas: The probability maps obtained by the model.
    :param min_cc: The threshold used to remove small connected components.
    :param classes_names: The classes names involved in the experiment.
    :return page_contours: The contour and confidence score obtained for each
                           detected object.
    """
    page_contours = {}
    max_probas = np.argmax(probas, axis=0)
    for channel in range(1, probas.shape[0]):
        # Keep pixels with highest probability.
        channel_probas = np.uint8(max_probas == channel) * probas[channel, :, :]
        # Retrieve the polygons contours.
        bin_img = channel_probas.copy()
        bin_img[bin_img > 0] = 1
        contours, hierarchy = cv2.findContours(
            np.uint8(bin_img), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        # Remove small connected components.
        if min_cc > 0:
            contours = [
                contour for contour in contours if cv2.contourArea(contour) > min_cc
            ]
        page_contours[classes_names[channel]] = [
            {
                "confidence": pr_utils.compute_confidence(contour, channel_probas),
                "polygon": contour,
            }
            for contour in contours
        ]
    return page_contours


def run(
    prediction_path: Path,
    log_path: Path,
    img_size: int,
    colors: list,
    classes_names: list,
    save_image: list,
    min_cc: int,
    loaders: dict,
    net,
):
    """
    Run the prediction.
    :param prediction_path: The path to save the predictions.
    :param log_path: Path to save the experiment information and model.
    :param img_size: Network input size.
    :param colors: Colors of the classes used during the experiment.
    :param classes_names: The names of the classes involved during the experiment.
    :param save_image: List of sets (train, val, test) for which the prediction images
                       are generated and saved.
    :param min_cc: The threshold used to remove small connected components.
    :param loaders: The loaders containing the images to predict.
    :param net: The loaded network.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Run prediction.
    net.eval()

    logging.info("Starting predicting")
    starting_time = time.time()

    with torch.no_grad():
        for set, loader in zip(["train", "val", "test"], loaders.values(), strict=True):
            seen_datasets = []
            # Create folders to save the predictions.
            (log_path / prediction_path / set).mkdir(exist_ok=True, parents=True)

            for data in tqdm(loader, desc=f"Prediction (prog) {set}"):
                # Create dataset folders to save the predictions.
                if data["dataset"][0] not in seen_datasets:
                    (log_path / prediction_path / set / data["dataset"][0]).mkdir(
                        exist_ok=True
                    )
                    seen_datasets.append(data["dataset"][0])

                # Generate and save the predictions.
                output = net(data["image"].to(device).float())
                input_size = [element.numpy()[0] for element in data["size"][:2]]

                assert output.shape[0] == 1
                polygons = get_predicted_polygons(
                    output[0].cpu().numpy(), min_cc, classes_names
                )
                polygons = pr_utils.resize_polygons(
                    polygons, input_size, img_size, data["padding"]
                )

                polygons["img_size"] = [int(element) for element in input_size]
                pr_utils.save_prediction(
                    polygons,
                    (
                        log_path
                        / prediction_path
                        / set
                        / data["dataset"][0]
                        / data["name"][0]
                    ),
                )
                if set in save_image:
                    pr_utils.save_prediction_image(
                        polygons,
                        colors,
                        input_size,
                        (
                            log_path
                            / prediction_path
                            / set
                            / data["dataset"][0]
                            / data["name"][0]
                        ),
                    )

    end = time.gmtime(time.time() - starting_time)
    logging.info(
        "Finished predicting in %2d:%2d:%2d", end.tm_hour, end.tm_min, end.tm_sec
    )
