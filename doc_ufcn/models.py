#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os

import requests
import yaml

from doc_ufcn.utils import md5sum

# Gitlab project: https://gitlab.com/teklia/dla/doc-ufcn
GITLAB_PROJECT_ID = 30605923
PACKAGES_URL = f"https://gitlab.com/api/v4/projects/{GITLAB_PROJECT_ID}/packages/"


def download_model(name, version=None):

    base_dir = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    dir_path = os.path.join(base_dir, "doc-ufcn", "models", name)
    os.makedirs(dir_path, exist_ok=True)

    model_path = os.path.join(dir_path, "model.pth")
    parameters_path = os.path.join(dir_path, "parameters.yml")

    # If no version given, get latest version.
    if version is None:
        packages = requests.get(
            PACKAGES_URL + f"?package_name={name}&order_by=version&sort=desc"
        )
        packages.raise_for_status()
        packages = yaml.safe_load(packages.content)
        assert len(packages) > 0, f"Model {name} not available"
        version = packages[0]["version"]

    # Download and save the parameters.
    parameters = requests.get(PACKAGES_URL + f"generic/{name}/{version}/parameters.yml")
    parameters.raise_for_status()
    logging.info(f"Loaded parameters: {name} (version {version})")
    parameters = yaml.safe_load(parameters.content)
    with open(parameters_path, "w") as f:
        yaml.safe_dump(parameters, f)

    # Check if model already in cache. Return the cached model and parameters.
    if os.path.isfile(model_path) and md5sum(model_path) == parameters["md5sum"]:
        logging.info(f"Loaded model from cache: {name} (version {version})")
        return model_path, parameters["parameters"]

    # Download the model if not in cache and save it.
    model = requests.get(PACKAGES_URL + f"generic/{name}/{version}/model.pth")
    model.raise_for_status()
    logging.info(f"Loaded model: {name} (version {version})")
    with open(model_path, "wb") as f:
        f.write(model.content)

    return model_path, parameters["parameters"]
