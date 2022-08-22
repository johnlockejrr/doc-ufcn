# -*- coding: utf-8 -*-
import json
import os
from pathlib import Path

import cv2
import pytest
import yaml

FIXTURES = Path(__file__).resolve().parent / "data"


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


@pytest.fixture
def mask_image():
    image = cv2.imread(os.path.join(FIXTURES, "mask_image.png"))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


@pytest.fixture
def test_parameters():
    with open(os.path.join(FIXTURES, "test_params.yaml"), "r") as f:
        parameters = yaml.safe_load(f)
    return parameters


@pytest.fixture
def overlap_image():
    image = cv2.imread(os.path.join(FIXTURES, "overlap_image.png"))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


@pytest.fixture
def test_json():
    with open(os.path.join(FIXTURES, "test_json.json"), "r") as json_file:
        return json.load(json_file)


@pytest.fixture
def page_generic_model_path():
    return os.path.join(FIXTURES, "page_generic.pth")


@pytest.fixture
def expected_mask_path():
    return os.path.join(FIXTURES, "expected_mask.png")
