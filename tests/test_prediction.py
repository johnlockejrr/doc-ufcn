#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import cv2
import numpy as np
import pytest

from doc_ufcn import prediction

FIXTURES = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "data",
)


@pytest.fixture
def test_image():
    image = cv2.imread(os.path.join(FIXTURES, "test_image.png"))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


@pytest.fixture
def test_masked_image():
    image = cv2.imread(os.path.join(FIXTURES, "masked_image.png"))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


@pytest.mark.parametrize(
    "probabilities, no_of_classes, expected_polygons",
    [
        # Empty prediction
        (
            [
                [
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1],
                ],
                [
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                ],
            ],
            2,
            {1: []},
        ),
        # One polygon on second channel
        (
            [
                [
                    [1, 1, 1],
                    [1, 0, 1],
                    [1, 1, 1],
                ],
                [
                    [0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0],
                ],
            ],
            2,
            {1: [{"confidence": 1.0, "polygon": [[[1, 1]]]}]},
        ),
        # One polygon on third channel
        (
            [
                [
                    [1, 1, 1, 0],
                    [1, 0, 0, 0],
                    [1, 1, 1, 0],
                ],
                [
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
                [
                    [0, 0, 0, 1],
                    [0, 1, 1, 1],
                    [0, 0, 0, 1],
                ],
            ],
            3,
            {
                1: [],
                2: [
                    {
                        "confidence": 1.0,
                        "polygon": [[[3, 0]], [[2, 1]], [[1, 1]], [[2, 1]], [[3, 2]]],
                    }
                ],
            },
        ),
        # One polygon on third channel with real probabilities
        (
            [
                [
                    [0.6, 0.4, 0.4, 0.1],
                    [0.5, 0.0, 0.2, 0.3],
                    [0.4, 0.5, 0.5, 0.2],
                ],
                [
                    [0.2, 0.3, 0.4, 0.1],
                    [0.3, 0.3, 0.2, 0.2],
                    [0.3, 0.1, 0.3, 0.3],
                ],
                [
                    [0.2, 0.3, 0.2, 0.8],
                    [0.2, 0.7, 0.6, 0.5],
                    [0.3, 0.4, 0.2, 0.5],
                ],
            ],
            3,
            {
                1: [],
                2: [
                    {
                        "confidence": 0.62,
                        "polygon": [[[3, 0]], [[2, 1]], [[1, 1]], [[2, 1]], [[3, 2]]],
                    }
                ],
            },
        ),
        # Two polygons on third channel with real probabilities
        (
            [
                [
                    [0.2, 0.4, 0.4, 0.1],
                    [0.4, 0.6, 0.2, 0.3],
                    [0.3, 0.5, 0.5, 0.2],
                ],
                [
                    [0.1, 0.3, 0.4, 0.1],
                    [0.0, 0.4, 0.2, 0.2],
                    [0.3, 0.1, 0.3, 0.3],
                ],
                [
                    [0.7, 0.3, 0.2, 0.8],
                    [0.6, 0.0, 0.6, 0.5],
                    [0.4, 0.4, 0.2, 0.5],
                ],
            ],
            3,
            {
                1: [],
                2: [
                    {"confidence": 0.6, "polygon": [[[3, 0]], [[2, 1]], [[3, 2]]]},
                    {"confidence": 0.57, "polygon": [[[0, 0]], [[0, 2]]]},
                ],
            },
        ),
        # One polygon on second channel and one polygon on third channel with real probabilities
        (
            [
                [
                    [0.2, 0.4, 0.4, 0.1],
                    [0.4, 0.6, 0.0, 0.3],
                    [0.3, 0.5, 0.5, 0.2],
                ],
                [
                    [0.7, 0.3, 0.4, 0.1],
                    [0.6, 0.4, 0.0, 0.2],
                    [0.4, 0.1, 0.3, 0.3],
                ],
                [
                    [0.1, 0.3, 0.2, 0.8],
                    [0.0, 0.0, 1.0, 0.5],
                    [0.3, 0.4, 0.2, 0.5],
                ],
            ],
            3,
            {
                1: [{"confidence": 0.57, "polygon": [[[0, 0]], [[0, 2]]]}],
                2: [{"confidence": 0.7, "polygon": [[[3, 0]], [[2, 1]], [[3, 2]]]}],
            },
        ),
    ],
)
def test_get_predicted_polygons(probabilities, no_of_classes, expected_polygons):
    """
    Test of the get_predicted_polygons function.
    Check that the correct polygons are extracted from probability map.
    """
    polygons = prediction.get_predicted_polygons(np.array(probabilities), no_of_classes)
    # Polygons to list
    for key, value in polygons.items():
        for index, element in enumerate(value):
            polygons[key][index]["polygon"] = polygons[key][index]["polygon"].tolist()

    assert polygons == expected_polygons


@pytest.mark.parametrize(
    "probabilities, polygon, expected_confidence",
    [
        # One polygon with probabilities = 1
        (
            [
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
            ],
            [[[1, 1]]],
            1.0,
        ),
        # One polygon with probability = 1
        (
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
            ],
            [[[1, 1]]],
            1.0,
        ),
        # One polygon with probability = 1
        (
            [
                [0, 0, 0, 1],
                [0, 1, 1, 1],
                [0, 0, 0, 1],
            ],
            [[[3, 0]], [[2, 1]], [[1, 1]], [[2, 1]], [[3, 2]]],
            1.0,
        ),
        # One polygon with real probabilities
        (
            [
                [0.2, 0.3, 0.2, 0.8],
                [0.2, 0.7, 0.6, 0.5],
                [0.3, 0.4, 0.2, 0.5],
            ],
            [[[3, 0]], [[2, 1]], [[1, 1]], [[2, 1]], [[3, 2]]],
            0.62,
        ),
        # One polygon with real probabilities
        (
            [
                [0.1, 0.3, 0.2, 0.8],
                [0.0, 0.0, 1.0, 0.5],
                [0.3, 0.4, 0.2, 0.5],
            ],
            [[[3, 0]], [[2, 1]], [[3, 2]]],
            0.7,
        ),
    ],
)
def test_compute_confidence(probabilities, polygon, expected_confidence):
    """
    Test of the compute_confidence function.
    Check that the computed confidence is correct.
    """
    confidence = prediction.compute_confidence(
        np.array(polygon), np.array(probabilities)
    )
    assert confidence == expected_confidence


@pytest.mark.parametrize(
    "polygons, image_size, resized_image_size, padding, expected_resized_polygons",
    [
        (
            {
                1: [{"confidence": 0.5, "polygon": [[[1, 1]]]}],
            },
            (9, 9),
            3,
            [0, 0],
            {
                1: [{"confidence": 0.5, "polygon": [(3, 3)]}],
            },
        ),
        (
            {
                1: [
                    {
                        "confidence": 0.5,
                        "polygon": [[[3, 0]], [[2, 1]], [[1, 1]], [[2, 1]], [[3, 2]]],
                    }
                ],
            },
            (10, 10),
            5,
            [1, 0],
            {
                1: [
                    {
                        "confidence": 0.5,
                        "polygon": [(6, 0), (4, 0), (2, 0), (4, 0), (6, 2)],
                    }
                ],
            },
        ),
        (
            {
                1: [{"confidence": 0.57, "polygon": [[[0, 0]], [[0, 2]]]}],
                2: [{"confidence": 0.7, "polygon": [[[3, 0]], [[2, 1]], [[3, 2]]]}],
            },
            (10, 10),
            6,
            [0, 1],
            {
                1: [{"confidence": 0.57, "polygon": [(0, 0), (0, 3)]}],
                2: [{"confidence": 0.7, "polygon": [(3, 0), (1, 1), (3, 3)]}],
            },
        ),
    ],
)
def test_resize_predicted_polygons(
    polygons, image_size, resized_image_size, padding, expected_resized_polygons
):
    """
    Test of the resize_predicted_polygons function.
    Check that the polygons are correctly resized.
    """
    resized_polygons = prediction.resize_predicted_polygons(
        polygons, image_size, resized_image_size, padding
    )
    assert resized_polygons == expected_resized_polygons


@pytest.mark.parametrize(
    "polygons, image_size, image, expected_image",
    [
        (
            {1: []},
            (15, 15),
            None,
            np.zeros((15, 15)),
        ),
        (
            {1: []},
            None,
            pytest.lazy_fixture("test_image"),
            pytest.lazy_fixture("test_image"),
        ),
        (
            {
                1: [{"confidence": 0.57, "polygon": [[[0, 0]], [[5, 15]]]}],
                2: [{"confidence": 0.7, "polygon": [[[15, 0]], [[10, 10]], [[8, 6]]]}],
            },
            (15, 15),
            None,
            [
                [127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255],
                [127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255],
                [0, 127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255],
                [0, 127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 0],
                [0, 127, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 0],
                [0, 0, 127, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0],
                [0, 0, 127, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 0, 0],
                [0, 0, 127, 0, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0],
                [0, 0, 0, 127, 0, 0, 0, 0, 0, 255, 255, 255, 0, 0, 0],
                [0, 0, 0, 127, 0, 0, 0, 0, 0, 255, 255, 0, 0, 0, 0],
                [0, 0, 0, 0, 127, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0],
                [0, 0, 0, 0, 127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 127, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 127, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
        ),
        (
            {
                1: [
                    {
                        "confidence": 0.57,
                        "polygon": [[[75, 50]], [[75, 500]], [[700, 500]], [[700, 50]]],
                    }
                ],
                2: [
                    {
                        "confidence": 0.7,
                        "polygon": [[[80, 65]], [[80, 485]], [[415, 485]], [[415, 65]]],
                    }
                ],
            },
            None,
            pytest.lazy_fixture("test_image"),
            pytest.lazy_fixture("test_masked_image"),
        ),
    ],
)
def test_get_prediction_image(polygons, image_size, image, expected_image):
    """
    Test of the get_prediction_image function.
    Check that the polygons are correctly drawn over a mask or an input image.
    """
    if image_size is None:
        image_size = image.shape
    mask = prediction.get_prediction_image(polygons, image_size, image)
    assert np.array_equal(mask, expected_image)
