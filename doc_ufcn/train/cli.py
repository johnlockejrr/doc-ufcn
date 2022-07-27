# -*- coding: utf-8 -*-
import argparse
import json
import logging
from pathlib import Path

from teklia_toolbox.config import ConfigParser, ConfigurationError

from doc_ufcn.train.experiment import run

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


logger = logging.getLogger(__name__)

STEPS = ["normalization_params", "train", "prediction", "evaluation"]


def parse_args():
    parser = argparse.ArgumentParser(description="Run a Doc-UFCN training experiment")
    parser.add_argument(
        "config",
        type=Path,
        nargs="+",
        help="Path towards an experiment or global configuration file. All files will be merged in order.",
    )

    return parser.parse_args()


def parse_configurations(paths):
    def _step(value: str):
        if value not in STEPS:
            raise ConfigurationError(f"Invalid step {value}")
        return value

    def _user_path(value: str):
        return Path(value).expanduser()

    def _loss(value: str):
        value = value.lower()
        if value not in ["initial", "best"]:
            raise ConfigurationError(f"Invalid loss {value}")
        return value

    parser = ConfigParser()
    parser.add_option("experiment_name", type=str, default="doc-ufcn")

    # List of the steps to run.
    parser.add_option("steps", type=_step, many=True, default=STEPS)

    # Path to save the Tensorboard events.
    parser.add_option("tb_path", type=Path, default=Path("events"))

    # Path to save the experiment information and model.
    parser.add_option("log_path", type=Path, default=None)

    # Path to the data folders.
    data_paths = parser.add_subparser("data_paths")

    train_paths = data_paths.add_subparser("train")
    train_paths.add_option(
        "image", type=_user_path, default=Path("./data/train/images"), many=True
    )
    train_paths.add_option(
        "mask", type=_user_path, default=Path("./data/train/labels"), many=True
    )
    train_paths.add_option(
        "json", type=_user_path, default=Path("./data/train/labels_json"), many=True
    )

    val_paths = data_paths.add_subparser("val")
    val_paths.add_option(
        "image", type=_user_path, default=Path("./data/val/images"), many=True
    )
    val_paths.add_option(
        "mask", type=_user_path, default=Path("./data/val/labels"), many=True
    )
    val_paths.add_option(
        "json", type=_user_path, default=Path("./data/val/labels_json"), many=True
    )

    test_paths = data_paths.add_subparser("test")
    test_paths.add_option(
        "image", type=_user_path, default=Path("./data/test/images"), many=True
    )
    test_paths.add_option(
        "json", type=_user_path, default=Path("./data/test/labels_json"), many=True
    )

    # Training parameters.
    training = parser.add_subparser("training", default={})
    training.add_option("restore_model", type=Path, default=None)
    training.add_option("loss", type=str, default="initial")

    # Global parameters of the experiment.
    params = parser.add_subparser("params", default={})
    params.add_option("mean", type=str, default="mean")
    params.add_option("std", type=str, default="std")
    params.add_option("model_path", type=Path, default=Path("model.pth"))
    params.add_option("prediction_path", type=Path, default=Path("prediction"))

    # Global parameters of the experiment entered by the user.
    global_params = parser.add_subparser("global_params", default={})
    global_params.add_option(
        "classes_names", type=str, many=True, default=["background", "text_line"]
    )
    global_params.add_option(
        "classes_colors", type=str, many=True, default=[[0, 0, 0], [0, 0, 255]]
    )
    global_params.add_option("img_size", type=int, default=768)
    global_params.add_option("no_of_epoch", type=int, default=100)
    global_params.add_option("batch_size", type=int, default=None)
    global_params.add_option("no_of_params", type=int, default=None)
    global_params.add_option("bin_size", type=int, default=20)
    global_params.add_option("learning_rate", type=float, default=5e-3)
    global_params.add_option("omniboard", type=bool, default=False)
    global_params.add_option("min_cc", type=int, default=0)
    global_params.add_option("save_image", type=str, many=True, default=[])
    global_params.add_option("use_amp", type=bool, default=False)

    # Merge all provided configuration files into a single payload
    # that will be validated by the configuration parser described above
    raw = {}
    for path in paths:
        try:
            raw.update(json.load(path.open()))
        except Exception as e:
            logger.error(f"Failed to parse config {path} : {e}")
            raise Exception("Invalid configuration")

    out = parser.parse_data(raw)

    assert (
        out["global_params"]["batch_size"] is not None
        or out["global_params"]["no_of_params"] is not None
    ), "Please provide a batch size or a maximum number of parameters"

    # Update log path using experiment name
    if out["log_path"] is None:
        slug = out["experiment_name"].lower().replace(" ", "_").replace("-", "_")
        out["log_path"] = Path("./runs") / slug

    return out


def main():

    args = parse_args()

    # Parse all configuration files provided from CLI
    config = parse_configurations(args.config)

    # Run experiment
    run(config)


if __name__ == "__main__":
    main()
