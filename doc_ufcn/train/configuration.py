import json
from pathlib import Path

from teklia_toolbox.config import ConfigParser, ConfigurationError

from doc_ufcn.train import logger

STEPS = ["normalization_params", "train", "prediction", "evaluation"]


def parse_configurations(paths: list[Path]):
    """
    Parse multiple JSON configuration files into a single source
    of configuration for the whole training workflow
    """

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

    def _same_classes(value: str):
        value = value.lower()
        if value not in ["true", "false"]:
            raise ConfigurationError(
                f"Invalid same classes argument: {value}. This value should be set to True or False"
            )
        return value == "true"

    def _rgb(value: list):
        if not isinstance(value, list | tuple):
            raise ConfigurationError("This RGB value should be a list or tuple")

        if len(value) != 3:
            raise ConfigurationError("This RGB value should be a list with 3 items")

        if not all(isinstance(v, int) for v in value):
            raise ConfigurationError("This RGB value should be set with integers only")

        if not all(0 <= v <= 255 for v in value):
            raise ConfigurationError(
                "This RGB value should be set with integers in range 0-255"
            )

        return value

    parser = ConfigParser()
    parser.add_option("experiment_name", type=str, default="doc-ufcn")

    # List of the steps to run.
    parser.add_option("steps", type=_step, many=True, default=STEPS)

    # Global parameters of the experiment entered by the user.
    parser.add_option(
        "classes_names", type=str, many=True, default=["background", "text_line"]
    )
    parser.add_option(
        "classes_colors", type=_rgb, many=True, default=[[0, 0, 0], [0, 0, 255]]
    )
    parser.add_option("img_size", type=int, default=768)
    parser.add_option("no_of_epochs", type=int, default=100)
    parser.add_option("batch_size", type=int, default=None)
    parser.add_option("no_of_params", type=int, default=None)
    parser.add_option("bin_size", type=int, default=20)
    parser.add_option("learning_rate", type=float, default=5e-3)
    parser.add_option("min_cc", type=int, default=0)
    parser.add_option("save_image", type=str, many=True, default=[])
    parser.add_option("use_amp", type=bool, default=False)
    parser.add_option("mean", type=str, default="mean")
    parser.add_option("std", type=str, default="std")
    parser.add_option("model_path", type=Path, default=Path("model.pth"))
    parser.add_option("prediction_path", type=Path, default=Path("prediction"))
    parser.add_option("evaluation_path", type=Path, default=Path("evaluation"))

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
    training.add_option("restore_model", type=str, default=None)
    training.add_option("same_classes", type=_same_classes, default="true")
    training.add_option("loss", type=_loss, default="initial")

    # MLflow parameters
    mlflow = parser.add_subparser("mlflow", default=None)
    mlflow.add_option("experiment_id", type=str)
    mlflow.add_option("tracking_uri", type=str)
    mlflow.add_option("s3_endpoint_url", type=str)
    mlflow.add_option("aws_access_key_id", type=str, default=None)
    mlflow.add_option("aws_secret_access_key", type=str, default=None)
    mlflow.add_option("run_name", type=str, default=None)

    # Merge all provided configuration files into a single payload
    # that will be validated by the configuration parser described above
    raw = {}
    for path in paths:
        try:
            raw.update(json.loads(path.read_text()))
        except Exception as e:
            logger.error(f"Failed to parse config {path} : {e}")
            raise Exception("Invalid configuration") from e

    # Promote deprecated parameters to root level
    for deprecated_key in ("params", "global_params"):
        if deprecated_key in raw:
            logger.warn(
                f"Promoting {deprecated_key} to root configuration level. You should update your configuration to promote the parameters directly."
            )
            deprecated = raw.pop(deprecated_key)
            raw.update(deprecated)

    out = parser.parse_data(raw)

    assert (
        out["batch_size"] is not None or out["no_of_params"] is not None
    ), "Please provide a batch size or a maximum number of parameters"

    # Update log path using experiment name
    if out["log_path"] is None:
        slug = out["experiment_name"].lower().replace(" ", "_").replace("-", "_")
        out["log_path"] = Path("./runs") / slug

    return out


def save_configuration(config: dict):
    """
    Save the current configuration.
    :param log_path: Path to save the experiment information and model.
    :param experiment_name: The name of the experiment that is used to save all
                      the experiment information.
    :param config : Full configuration payload that will be saved and usable to retry the experiment
    """
    config["log_path"].mkdir(exist_ok=True)
    path = config["log_path"] / (f"{config['experiment_name']}.json")
    path.write_text(json.dumps(config, indent=4, default=str, sort_keys=True))
    logger.info(f"Saved configuration in {path.resolve()}")
