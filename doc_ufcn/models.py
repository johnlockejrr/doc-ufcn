#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import requests
import yaml

PROJECT_ID = "30605923"


def download_model(name, version=None):

    dir_path = os.path.join("doc-ufcn", "models", name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    model_path = os.path.join(dir_path, "model.pth")
    parameters_path = os.path.join(dir_path, "parameters.yml")

    model = requests.get(
        f"https://gitlab.com/api/v4/projects/{PROJECT_ID}/packages/generic/{name}/{version}/model.pth"
    )
    with open(model_path, "wb") as file:
        file.write(model.content)

    parameters = requests.get(
        f"https://gitlab.com/api/v4/projects/{PROJECT_ID}/packages/generic/{name}/{version}/parameters.yml"
    )
    parameters = yaml.safe_load(parameters.content)
    with open(parameters_path, "w") as file:
        yaml.safe_dump(parameters, file)

    parameters = {
        list(parameter.keys())[0]: list(parameter.values())[0]
        for parameter in parameters["parameters"]
    }

    return model_path, parameters
