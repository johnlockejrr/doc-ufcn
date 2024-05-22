#!/usr/bin/env python

"""
The retrieve experiments configs module
======================

Use it to get the configurations for running the experiments.
"""

import argparse
import csv
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


STAGE_TRAIN = "train"
STAGE_VAL = "val"
STAGE_TEST = "test"

CONVERT = {"images": "image", "labels": "mask", "labels_json": "json"}


def generate_configurations(csv_path):
    """
    Read configuration references from a CSV and generate a configuration
    for each line, using related local files
    """
    assert csv_path.exists(), f"Missing CSV {csv_path}"

    workdir = csv_path.parent.resolve()

    reader = csv.DictReader(csv_path.open(), delimiter=",")
    for row in reader:
        config = {}
        # Get experiment name.
        assert row["experiment_name"] != "", "Missing experiment name"
        config["experiment_name"] = row["experiment_name"]

        # Get steps as a list of names.
        config["steps"] = row["steps"].split(";")

        # Get train/val/test folders.
        config["data_paths"] = {}
        for dataset in [STAGE_TRAIN, STAGE_VAL, STAGE_TEST]:
            config["data_paths"][dataset] = {}

            # Handle conversion table, no labels when running tests
            conversion_table = dict(CONVERT)
            if dataset == STAGE_TEST:
                del conversion_table["labels"]

            for folder, key in CONVERT.items():
                if row[dataset] != "":
                    config["data_paths"][dataset][key] = [
                        (workdir / element / dataset / folder).as_posix()
                        for element in row[dataset].split(";")
                    ]
                else:
                    config["data_paths"][dataset][key] = []

        # Get restore model.
        if row["restore_model"] != "":
            config["training"] = {"restore_model": row["restore_model"]}
            if row["same_classes"] != "":
                config["training"]["same_classes"] = row["same_classes"]
            if row["loss"] != "":
                config["training"]["loss"] = row["loss"]

        yield config


def run(csv_path, output: Path):
    """
    Retrieve the configurations for the experiments from a config csv file.
    Save each configuration into TMP_DIR/experiment_name file.
    """
    output.mkdir(exist_ok=True)

    for index, config in enumerate(generate_configurations(csv_path), 1):
        # Save each configuration in a dedicated file file.
        json_file = f"{index}_{config['experiment_name']}.json"
        (output / json_file).write_text(
            json.dumps(
                {key: value for key, value in config.items() if value},
                indent=4,
            )
        )

    logger.info(
        f"Retrieved {index} experiment configurations from {csv_path}"
        if index > 1
        else f"Retrieved {index} experiment configuration from {csv_path}"
    )


def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser(
        description="Script to retrieve the experiments configs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "config", type=Path, help="Path to the configurations file (CSV format)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./tmp"),
        help="Directory where the generated configurations will be stored",
    )

    args = parser.parse_args()
    run(args.config, args.output)


if __name__ == "__main__":
    main()
