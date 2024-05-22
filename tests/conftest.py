import json

import cv2
import pytest
import yaml

from tests import FIXTURES


@pytest.fixture()
def test_image():
    image = cv2.imread(str(FIXTURES / "test_image.png"))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


@pytest.fixture()
def test_masked_image():
    image = cv2.imread(str(FIXTURES / "masked_image.png"))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


@pytest.fixture()
def mask_image():
    image = cv2.imread(str(FIXTURES / "mask_image.png"))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


@pytest.fixture()
def test_parameters():
    return yaml.safe_load((FIXTURES / "test_params.yaml").read_bytes())


@pytest.fixture()
def overlap_image():
    image = cv2.imread(str(FIXTURES / "overlap_image.png"))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


@pytest.fixture()
def test_json():
    return json.loads((FIXTURES / "test_json.json").read_text())


@pytest.fixture()
def page_generic_model_path():
    return FIXTURES / "page_generic.pth"


@pytest.fixture()
def expected_mask_path():
    return FIXTURES / "expected_mask.png"
