#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os

import cv2
import numpy as np
import pytest

from doc_ufcn import main

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
def mask_image():
    image = cv2.imread(os.path.join(FIXTURES, "mask_image.png"))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


@pytest.fixture
def overlap_image():
    image = cv2.imread(os.path.join(FIXTURES, "overlap_image.png"))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


@pytest.fixture
def test_json():
    with open(os.path.join(FIXTURES, "test_json.json"), "r") as json_file:
        return json.load(json_file)


@pytest.mark.parametrize(
    "no_of_classes",
    [
        (-1),
        ([]),
        (2.5),
        ("no_of_classes"),
    ],
)
def test_DocUFCN_wrong_no_of_classes(no_of_classes):
    """
    Test of the DocUFCN init function: check that a wrong number of classes raises an exception.
    """
    with pytest.raises(AssertionError):
        main.DocUFCN(no_of_classes, 768, "cpu")


@pytest.mark.parametrize(
    "no_of_classes",
    [
        (1),
        (5),
        (12),
    ],
)
def test_DocUFCN_correct_no_of_classes(no_of_classes):
    """
    Test of the DocUFCN init function: check that object is correctly created.
    """
    model = main.DocUFCN(no_of_classes, 768, "cpu")
    assert model.no_of_classes == no_of_classes
    assert model.model_input_size == 768
    assert model.device == "cpu"


@pytest.mark.parametrize(
    "input_size",
    [
        (-1),
        ([]),
        (2.5),
        ("input_size"),
    ],
)
def test_DocUFCN_wrong_input_size(input_size):
    """
    Test of the DocUFCN init function: check that a wrong input size raises an exception.
    """
    with pytest.raises(AssertionError):
        main.DocUFCN(2, input_size, "cpu")


@pytest.mark.parametrize(
    "input_size",
    [
        (768),
        (250),
    ],
)
def test_DocUFCN_correct_input_size(input_size):
    """
    Test of the DocUFCN init function: check that the object is correctly created.
    """
    model = main.DocUFCN(2, input_size, "cpu")
    assert model.no_of_classes == 2
    assert model.model_input_size == input_size
    assert model.device == "cpu"


@pytest.mark.parametrize(
    "model_path",
    [
        # Wrong path/file
        ("page_generic.pth"),
        (1),
        # Correct path
        (os.path.join(FIXTURES, "page_generic.pth")),
    ],
)
def test_load(model_path):
    """
    Test of the load function.
    """
    model = main.DocUFCN(2, 768, "cpu")

    if not os.path.isfile(model_path):
        with pytest.raises(AssertionError):
            model.load(model_path, [190, 182, 165], [48, 48, 45])
    elif not isinstance(model_path, str):
        with pytest.raises(AttributeError):
            model.load(model_path, [190, 182, 165], [48, 48, 45])
    else:
        model.load(model_path, [190, 182, 165], [48, 48, 45])
        assert model.net is not None
        assert model.mean == [190, 182, 165]
        assert model.std == [48, 48, 45]


@pytest.mark.parametrize(
    "mean",
    [
        (10),
        ([]),
        ([10]),
        (["test"]),
        ([10, "test"]),
        (-1.5),
        ("mean"),
    ],
)
def test_mean(mean):
    """
    Test of the load function: check that wrong mean values raise an exception.
    """
    model = main.DocUFCN(2, 768, "cpu")
    with pytest.raises(AssertionError):
        model.load(os.path.join(FIXTURES, "page_generic.pth"), mean, [48, 48, 45])


@pytest.mark.parametrize(
    "std",
    [
        (10),
        ([]),
        ([10]),
        (["test"]),
        ([10, "test"]),
        (-1.5),
        ("mean"),
    ],
)
def test_std(std):
    """
    Test of the load function: check that wrong std values raise an exception.
    """
    model = main.DocUFCN(2, 768, "cpu")
    with pytest.raises(AssertionError):
        model.load(os.path.join(FIXTURES, "page_generic.pth"), [190, 182, 165], std)


@pytest.mark.parametrize(
    "input_image",
    [([1]), ("input_image"), (1)],
)
def test_predict_input_image(input_image):
    """
    Test of the predict function: check that wrong input images raise an exception.
    """
    model = main.DocUFCN(2, 768, "cpu")
    model.load(
        os.path.join(FIXTURES, "page_generic.pth"), [190, 182, 165], [48, 48, 45]
    )

    with pytest.raises(AssertionError):
        model.predict(input_image)


def test_predict(test_image, test_json):
    """
    Test of the predict function.
    """
    model = main.DocUFCN(2, 768, "cpu")
    model.load(
        os.path.join(FIXTURES, "page_generic.pth"), [190, 182, 165], [48, 48, 45]
    )

    # Only one output
    output = model.predict(test_image)
    for key in output[0].keys():
        assert len(output[0][key]) == len(test_json[str(key)])
        for polygon, expected_polygon in zip(output[0][key], test_json[str(key)]):
            polygon["polygon"] = [list(element) for element in polygon["polygon"]]
            assert polygon["polygon"] == expected_polygon["polygon"]
            assert polygon["confidence"] == expected_polygon["confidence"]
    assert output[1:4] == (None, None, None)


def test_predict_all_outputs(test_image, mask_image, overlap_image):
    """
    Test of the predict function with all four outputs.
    """
    model = main.DocUFCN(2, 768, "cpu")
    model.load(
        os.path.join(FIXTURES, "page_generic.pth"), [190, 182, 165], [48, 48, 45]
    )

    # All outputs
    output = model.predict(
        test_image, raw_output=True, mask_output=True, overlap_output=True
    )
    assert isinstance(output[0], dict)
    assert output[1].shape == (2, test_image.shape[0], test_image.shape[1])
    assert np.array_equal(output[2].astype(np.uint8), mask_image)
    assert np.array_equal(output[3], overlap_image)
