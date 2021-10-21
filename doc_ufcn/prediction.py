#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np


def get_predicted_polygons(prediction, no_of_classes):
    """
    Keep the pixels with the highest probability across the channels
    and extract the contours of the connected components.
    Return a list of contours with their corresponding confidence scores.
    :param prediction: The probability maps.
    :param no_of_classes: The number of classes used to train the model.
    :return: The predicted polygons.
    """
    max_prediction = np.argmax(prediction, axis=0)
    # Get the contours of the objects.
    predicted_polygons = {}
    for channel in range(1, no_of_classes):
        probas_channel = np.uint8(max_prediction == channel) * prediction[channel, :, :]
        # Generate a binary image for the current channel.
        bin_img = probas_channel.copy()
        bin_img[bin_img > 0] = 1
        # Detect the objects contours.
        contours, _ = cv2.findContours(
            np.uint8(bin_img), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        predicted_polygons[channel] = [
            {
                "confidence": compute_confidence(contour, probas_channel),
                "polygon": contour,
            }
            for contour in contours
        ]
    return predicted_polygons


def compute_confidence(region, probs):
    """
    Compute the confidence of a given region from the probability map.
    Generates a mask of the size of the probability map to only keep the
    regions pixels. Get the sum of the probabilities within the region by
    multiplying the mask and the probability map. Return this sum divided
    by the number of pixels of the region.
    :param region: The region to compute the confidence.
    :param probs: The probability map used to compute the confidence score.
    :return: The mean of the region probabilities.
    """
    mask = np.zeros(probs.shape)
    cv2.drawContours(mask, [region], 0, 1, -1)
    confidence = np.sum(mask * probs) / np.sum(mask)
    return round(confidence, 2)


def resize_predicted_polygons(polygons, original_image_size, model_input_size, padding):
    """
    Resize the detected polygons to the original input image size.
    :param polygons: The polygons to resize.
    :param original_image_size: The original input size.
    :param model_input_size: The network input size.
    :param padding: The padding applied to the input image.
    :return polygons: The resized detected polygons.
    """
    # Compute the small size image.
    ratio = float(model_input_size) / max(original_image_size)
    new_size = tuple([int(x * ratio) for x in original_image_size])
    # Compute resizing ratio.
    ratio = [
        element / float(new) for element, new in zip(original_image_size, new_size)
    ]

    for channel in polygons.keys():
        for index, polygon in enumerate(polygons[channel]):
            x_points = [
                int((element[0][1] - padding[0]) * ratio[0])
                for element in polygon["polygon"]
            ]
            y_points = [
                int((element[0][0] - padding[1]) * ratio[1])
                for element in polygon["polygon"]
            ]

            x_points = np.clip(np.array(x_points), 0, original_image_size[0])
            y_points = np.clip(np.array(y_points), 0, original_image_size[1])

            polygons[channel][index]["polygon"] = list(zip(y_points, x_points))
        # Sort the polygons.
        polygons[channel] = sorted(
            polygons[channel],
            key=lambda item: (item["polygon"][0][1], item["polygon"][0][0]),
        )
    return polygons


def get_prediction_image(polygons, image_size, image=None):
    """
    Generate a mask with the detected polygons.
    :param polygons: The detected polygons coordinates.
    :param image_size: The original input image size.
    :param image: The input image.
    """
    if image is None:
        mask = np.zeros((image_size[0], image_size[1]))
        thickness = -1
    else:
        mask = image
        thickness = 2

    for channel in polygons.keys():
        color = int(channel * 255 / len(polygons.keys()))
        if image is not None:
            color = [0, color, 0]
        # Draw polygons.
        for polygon in polygons[channel]:
            cv2.drawContours(mask, [np.array(polygon["polygon"])], 0, color, thickness)
    return mask
