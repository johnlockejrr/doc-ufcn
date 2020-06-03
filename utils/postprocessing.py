#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    The postprocessing module
    ======================

    Use it to postprocess a prediction image to generate the bounding boxes.
"""

import cv2
import numpy as np

# Useful functions to clean the found regions.


def contains(poly1: np.array, poly2: np.array) -> bool:
    """
    Checks if the polygon poly2 is inside the polygon poly1.
    Poly2 is considered inside poly1 if all its points are inside.
    If the polygons are equals, poly1 contains poly2.
    :param poly1: The first polygon to test.
    :param poly2: The second polygon to test.
    :return: A boolean indicating whether poly1 contains poly2.
    """
    return all(
        cv2.pointPolygonTest(poly1, tuple(point), False) in (0, 1)
        for point in poly2
    )


def clean_regions(regions: list) -> list:
    """
    Removes the small regions inside bigger ones.
    :param regions: The list of all the regions.
    :return: The cleaned list of the regions.
    """
    cleaned_regions = regions.copy()
    i = 0
    while i < len(cleaned_regions):
        j = 0
        while j < len(cleaned_regions):
            if i != j and contains(cleaned_regions[i], cleaned_regions[j]):
                cleaned_regions.pop(j)
            elif i != j and contains(cleaned_regions[j], cleaned_regions[i]):
                cleaned_regions.pop(i)
                i -= 1
                break
            else:
                j += 1
        i += 1
    return cleaned_regions


def get_regions(prediction: np.ndarray) -> list:
    """
    Find the regions of a given type in the prediction.
    :param prediction: : The predicted image to find the regions.
    :return: The cleaned list of the found regions.
    """
    contours, _ = cv2.findContours(cv2.convertScaleAbs(prediction),
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    regions = [cv2.boxPoints(cv2.minAreaRect(contour)) for contour in contours]
    return clean_regions(regions)

# Run the postprocessing step.


def post_processing(prediction: np.ndarray, colors: list = None,
                    eval_step: bool = False) -> np.ndarray:
    """
    Improve the prediction by creating boxes and generating the
    corresponding image.
    :param prediction: The predicted image to improve.
    :param colors: The colors to use to fill the predicted image.
    :param eval_step: Indicates whether we are during the prediction
                      or evaluation stage.
    :return prediction_image: The improved prediction.
    """
    if eval_step:
        prediction_image = np.zeros((prediction.shape[1],
                                     prediction.shape[2]))
    else:
        prediction_image = np.zeros((prediction.shape[1],
                                     prediction.shape[2], 3))
    for channel in range(1, prediction.shape[0]):
        color = [int(element) for element in colors[channel]] \
                    if not eval_step else channel
        prediction_channel = np.uint8(prediction[channel, :, :] > 0)
        regions = get_regions(prediction_channel)
        # Draw the found regions.
        for region in regions:
            region = region.reshape((-1, 1, 2)).astype(np.int32)
            cv2.drawContours(prediction_image, [region], 0, color, -1)
    return prediction_image
