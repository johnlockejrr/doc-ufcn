# -*- coding: utf-8 -*-

import logging
import os

import yaml
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError

HUGGING_FACE_REPO_PREFIX = "Teklia/doc-ufcn-"


logger = logging.getLogger(__name__)


def download_model(name, version=None):
    # Strip the model name prefix if provided
    name = name.replace("doc-ufcn-", "")
    logger.info(f"Will look for model @ {HUGGING_FACE_REPO_PREFIX + name}")

    cache_dir = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    dir_path = os.path.join(cache_dir, "doc-ufcn", "models", name)

    try:
        # Retrieve parameters.yml
        parameters_path = hf_hub_download(
            repo_id=HUGGING_FACE_REPO_PREFIX + name,
            filename="parameters.yml",
            cache_dir=dir_path,
            revision=version,
        )
        # Retrieve model.pth
        model_path = hf_hub_download(
            repo_id=HUGGING_FACE_REPO_PREFIX + name,
            filename="model.pth",
            cache_dir=dir_path,
            revision=version,
        )
    except RepositoryNotFoundError as e:
        logger.error(
            f"Repository with name {name} was not found or you may not have access to it."
        )
        print(str(e))
        raise
    except RevisionNotFoundError as e:
        logger.error(
            f"Revision {version} was not found on the repository with name {name}."
        )
        print(str(e))
        raise

    with open(parameters_path) as f:
        parameters = yaml.safe_load(f)

    return model_path, parameters["parameters"]
