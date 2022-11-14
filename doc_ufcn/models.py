# -*- coding: utf-8 -*-

import logging
import os

import yaml
from huggingface_hub import hf_hub_download

HUGGING_FACE_REPO_PREFIX = "Teklia/doc-ufcn-"


logger = logging.getLogger(__name__)


def download_model(name, version=None):
    # Strip the model name prefix if provided
    name = name.replace("doc-ufcn-", "")
    logger.info(f"Will look for model @ {HUGGING_FACE_REPO_PREFIX + name}")

    cache_dir = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    dir_path = os.path.join(cache_dir, "doc-ufcn", "models", name)
    # Retrieve parameters.yml
    parameters_path = hf_hub_download(
        repo_id=HUGGING_FACE_REPO_PREFIX + name,
        filename="parameters.yml",
        cache_dir=dir_path,
        revision=version,
    )
    # Retrieve parameters.yml
    model_path = hf_hub_download(
        repo_id=HUGGING_FACE_REPO_PREFIX + name,
        filename="model.pth",
        cache_dir=dir_path,
        revision=version,
    )
    with open(parameters_path) as f:
        parameters = yaml.safe_load(f)

    return model_path, parameters["parameters"]
