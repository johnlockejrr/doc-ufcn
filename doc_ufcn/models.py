#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os

import requests
import yaml

logging.basicConfig(
    format="[%(levelname)s] %(message)s",
    level=logging.INFO,
)

PROJECT_ID = "30605923"


def download_model(name, version=None):

    dir_path = os.path.join("doc-ufcn", "models", name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    model_path = os.path.join(dir_path, "model.pth")
    parameters_path = os.path.join(dir_path, "parameters.yml")

    # Check model name.
    packages = requests.get(
        f"https://gitlab.com/api/v4/projects/{PROJECT_ID}/packages/"
    )
    available_models = set(
        [package["name"] for package in yaml.safe_load(packages.content)]
    )
    assert (
        name in available_models
    ), f"Model not in available models: {list(available_models)}"

    # Check model version. If None, get latest version.
    packages = requests.get(
        f"https://gitlab.com/api/v4/projects/{PROJECT_ID}/packages/?package_name={name}&order_by=version&sort=desc"
    )
    if version is None:
        latest_package = yaml.safe_load(packages.content)[0]
        version = latest_package["version"]
    else:
        available_versions = set(
            [package["version"] for package in yaml.safe_load(packages.content)]
        )
        assert (
            version in available_versions
        ), f"Not existing version: {list(available_versions)}"

    # Download the model and save it.
    model = requests.get(
        f"https://gitlab.com/api/v4/projects/{PROJECT_ID}/packages/generic/{name}/{version}/model.pth"
    )
    logging.info(f"Loaded model: {name} (version {version})")
    with open(model_path, "wb") as file:
        file.write(model.content)

    # Download the parameters and save them.
    parameters = requests.get(
        f"https://gitlab.com/api/v4/projects/{PROJECT_ID}/packages/generic/{name}/{version}/parameters.yml"
    )
    logging.info(f"Loaded parameters: {name} (version {version})")
    parameters = yaml.safe_load(parameters.content)
    with open(parameters_path, "w") as file:
        yaml.safe_dump(parameters, file)

    parameters = {
        list(parameter.keys())[0]: list(parameter.values())[0]
        for parameter in parameters["parameters"]
    }

    return model_path, parameters
