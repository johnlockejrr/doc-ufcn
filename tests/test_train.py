import json
import shutil
from pathlib import Path

import pytest

from doc_ufcn.train.configuration import parse_configurations
from doc_ufcn.train.experiment import run_experiment
from doc_ufcn.utils import export_list
from tests import FIXTURES

DATASET = FIXTURES / "training"


@pytest.fixture()
def config():
    return parse_configurations([DATASET / "config.json"])


def test_training_experiment(tmp_path, config):
    wk_dir = Path(tmp_path)

    # Add missing key-values
    config["log_path"] = wk_dir
    config["tb_path"] = wk_dir / "tensorboard_events"
    config["steps"] = ["normalization_params", "train"]

    # Create model folder
    (wk_dir / "model").mkdir()

    # Create log directories
    for epoch in range(config["no_of_epochs"]):
        (wk_dir / f"last_model_{epoch}").mkdir(exist_ok=True)

    run_experiment(
        config=config,
        num_workers=0,
    )

    # Check MEAN/STD computation
    for filename, value in (
        ("mean", "244"),
        ("std", "23"),
    ):
        assert (wk_dir / filename).read_text().splitlines() == [value] * 3

    # Check model creation
    assert (wk_dir / config["model_path"]).exists()


def test_evaluation_experiment(tmp_path, config):
    wk_dir = Path(tmp_path)

    config["log_path"] = wk_dir
    config["tb_path"] = wk_dir / "tensorboard_events"
    config["steps"] = ["prediction", "evaluation"]

    # Place the model in the right folder
    model_path = DATASET / "model.pth"
    shutil.copy(model_path, wk_dir / config["model_path"])

    # Set correct mean/std values
    export_list(data=[244] * 3, output=wk_dir / "mean")
    export_list(data=[23] * 3, output=wk_dir / "std")

    run_experiment(
        config=config,
        num_workers=0,
    )

    # Check prediction results
    for filename in (config["prediction_path"]).rglob("*.json"):
        # No text line predicted
        assert json.loads(filename.read_text()) == {
            "text_line": [],
            "img_size": [768, 537],
        }

    # Check evaluation results
    for filename in (config["evaluation_path"]).rglob("*/training_results.json"):
        # Model overfits
        assert json.loads(filename.read_text()) == {
            "text_line": {
                "iou": 1.0,
                "precision": 1.0,
                "recall": 1.0,
                "fscore": 1.0,
                "AP@[.5]": 1.0,
                "AP@[.75]": 1.0,
                "AP@[.95]": 1.0,
                "AP@[.5,.95]": 1.0,
            }
        }
