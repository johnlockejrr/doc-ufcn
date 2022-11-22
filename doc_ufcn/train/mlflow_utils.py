# -*- coding: utf-8 -*-
import os
from contextlib import contextmanager

import mlflow
from mlflow.exceptions import MlflowException

from doc_ufcn.train import logger


def check_environment():
    needed_variables = [
        "MLFLOW_S3_ENDPOINT_URL",
        "MLFLOW_TRACKING_URI",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
    ]
    try:
        for variable in needed_variables:
            assert os.getenv(variable)
    except AssertionError:
        raise Exception(
            f"{variable} is missing in the environment, cannot use MLflow logging."
        )


@contextmanager
def start_mlflow_run(config):
    # Make sure needed environment variables are set
    check_environment()

    # Set experiment from config
    experiment_id = config.get("experiment_id")
    assert experiment_id, "Missing MLflow experiment ID in the configuration"

    try:
        mlflow.set_experiment(experiment_id=experiment_id)
    except MlflowException as e:
        logger.error(f"Couldn't set Mlflow experiment with ID: {experiment_id}")
        raise e

    # Start run
    yield mlflow.start_run(run_name=config.get("run_name"))
