import os
from contextlib import contextmanager

import mlflow
from mlflow.exceptions import MlflowException

from doc_ufcn.train import logger


def setup_environment(config):
    needed_variables = {
        "MLFLOW_S3_ENDPOINT_URL": "s3_endpoint_url",
        "MLFLOW_TRACKING_URI": "tracking_uri",
        "AWS_ACCESS_KEY_ID": "aws_access_key_id",
        "AWS_SECRET_ACCESS_KEY": "aws_secret_access_key",
    }
    for variable_name, config_key in needed_variables.items():
        if config_key in config:
            os.environ[variable_name] = config[config_key]


@contextmanager
def start_mlflow_run(config):
    # Set needed variables in environment
    setup_environment(config)

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

    # End active run
    mlflow.end_run()
