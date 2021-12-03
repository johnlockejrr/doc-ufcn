#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import pytest
import requests
import yaml

from doc_ufcn import models

FIXTURES = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "data",
)


@pytest.fixture
def test_parameters():
    with open(os.path.join(FIXTURES, "test_params.yaml"), "r") as f:
        parameters = yaml.safe_load(f)
    return parameters


@pytest.mark.parametrize(
    "name, version, expected_model_path, expected_parameters",
    [
        # Correct name and version
        (
            "generic_page_detection",
            "0.0.2",
            "~/.cache/doc-ufcn/models/",
            pytest.lazy_fixture("test_parameters"),
        ),
        # Correct name and incorrect version
        ("generic_page_detection", "version", None, None),
        # Correct name and no version
        (
            "generic_page_detection",
            None,
            "~/.cache/doc-ufcn/models/",
            pytest.lazy_fixture("test_parameters"),
        ),
        # Incorrect name and correct version
        ("page_model", "0.0.2", None, None),
        # Incorrect name and incorrect version
        ("page_model", "version", None, None),
        # Incorrect name and no version
        ("page_model", None, None, None),
    ],
)
def test_download_model(name, version, expected_model_path, expected_parameters):
    """
    Test of the download_model function.
    Check that the correct model is loaded.
    """
    if expected_model_path is None and expected_parameters is None:
        if version is None:
            with pytest.raises(AssertionError):
                model_path, parameters = models.download_model(name, version)
        else:
            with pytest.raises(requests.exceptions.HTTPError):
                model_path, parameters = models.download_model(name, version)
    else:
        model_path, parameters = models.download_model(name, version)
        assert model_path == os.path.join(
            os.path.expanduser(expected_model_path), name, "model.pth"
        )
        assert parameters == expected_parameters
