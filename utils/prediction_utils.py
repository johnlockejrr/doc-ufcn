#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    The prediction utils module
    ======================
    
    Use it to during the prediction stage.
"""

import cv2
import json
import numpy as np
import imageio as io


def resize_polygons(polygons: dict, image_size: tuple,
                    input_size: tuple) -> dict:
    """
    Resize the detected polygons to the original input image size.
    :param polygons: The polygons to resize.
    :param image_size: The original input image size.
    :param input_size: The network input size.
    :return polygons: The resized detected polygons.
    """
    # Compute the small size image.
    ratio = float(input_size) / max(image_size)
    new_size = tuple([int(x * ratio) for x in image_size])
    # Extract the small image.
    delta_w = input_size - new_size[1]
    delta_h = input_size - new_size[0]
    top = delta_h // 2
    left = delta_w // 2
    # Compute resizing ratio
    ratio = [element / float(new) for element, new in zip(image_size, new_size)]

    for channel in polygons.keys():
        for index, polygon in enumerate(polygons[channel]):
            x_points = [element[0][1] for element in polygon['polygon']]
            y_points = [element[0][0] for element in polygon['polygon']]
            x_points = [int((element - top) * ratio[0]) for element in x_points]
            y_points = [int((element - left) * ratio[1]) for element in y_points]
            
            x_points = [int(element) if element < image_size[0] else int(image_size[0]) for element in x_points]
            y_points = [int(element) if element < image_size[1] else int(image_size[1]) for element in y_points]
            x_points = [int(element) if element > 0 else 0 for element in x_points]
            y_points = [int(element) if element > 0 else 0 for element in y_points]
            assert(max(x_points) <= image_size[0])
            assert(min(x_points) >= 0)                              
            assert(max(y_points) <= image_size[1])
            assert(min(y_points) >= 0)                              
            polygons[channel][index]['polygon'] = list(zip(y_points, x_points))
    return polygons

def compute_confidence(region: np.ndarray, probas: np.ndarray) -> float:
    """
    Compute the confidence score of a detected polygon.
    :param region: The detected polygon coordinates.
    :param probas: The probability map obtained by the network.
    :return: The confidence score of the given region.
    """
    mask = np.zeros(probas.shape)
    cv2.drawContours(mask, [region], 0, 1, -1)
    confidence = np.sum(mask * probas) / np.sum(mask)
    return round(confidence, 4)

# Save the prediction coordinates and images.


def save_prediction(polygons: dict, filename: str):
    """
    Save the detected polygon coordinates and their confidence score
    into a file.
    :param polygons: The detected polygons coordinates and
                     confidence scores.
    :param filename: The filename to save the detected polygons.
    """
    with open(filename.replace('png', 'json'), 'w') as outfile:
        json.dump(polygons, outfile, indent=4)

def save_prediction_image(polygons, colors, input_size, filename: str):
    """
    Save the detected polygon to an image.
    :param polygons: The detected polygons coordinates.
    :param colors: The colors corresponding to each involved class.
    :param input_size: The origianl input image size. 
    :param filename: The filename to save the prediction image.
    """
    image = np.zeros((input_size[0], input_size[1], 3))
    index = 1
    for channel in polygons.keys():
        if channel == 'img_size':
            continue
        color = [int(element) for element in colors[index]]
        for polygon in polygons[channel]:
            cv2.drawContours(image, [np.array(polygon['polygon'])], 0, color, -1)
        index += 1
    io.imsave(filename, np.uint8(image))

