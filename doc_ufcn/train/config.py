#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    The retrieve experiments configs module
    ======================

    Use it to get the configurations for running the experiments.
"""

import argparse
import csv
import json
import logging
import os
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

STEPS = ["normalization_params", "train", "prediction", "evaluation"]


def generate_configurations(csv_path):
    """
    Read configuration references from a CSV and generate a configuration
    for each line, using related local files
    """

    reader = csv.DictReader(csv_path.open(), delimiter=",")
    for row in reader:

        config = {}
        # Get experiment name.
        assert row["experiment_name"] != ""
        config["experiment_name"] = row["experiment_name"]

        # Get steps as a list of names.
        config["steps"] = row["steps"].split(";")

        # Get train/val/test folders.
        config["data_paths"] = {}
        for dataset in ["train", "val", "test"]:
            config["data_paths"][dataset] = {}
            if dataset in ["train", "val"]:
                for folder, key in zip(
                    ["images", "labels", "labels_json"], ["image", "mask", "json"]
                ):
                    if row[dataset] != "":
                        config["data_paths"][dataset][key] = [
                            os.path.join(element, dataset, folder)
                            for element in row[dataset].split(";")
                        ]
                    else:
                        config["data_paths"][dataset][key] = []
            else:
                for folder, key in zip(["images", "labels_json"], ["image", "json"]):
                    if row[dataset] != "":
                        config["data_paths"][dataset][key] = [
                            os.path.join(element, dataset, folder)
                            for element in row[dataset].split(";")
                        ]
                    else:
                        config["data_paths"][dataset][key] = []

        # Get restore model.
        if row["restore_model"] != "":
            config["training"] = {"restore_model": row["restore_model"]}
            if row["loss"] != "":
                config["training"]["loss"] = row["loss"]

        yield config


def run(csv_path, output):
    """
    Retrieve the configurations for the experiments from a config csv file.
    Save each configuration into TMP_DIR/experiment_name file.
    """
    os.makedirs(output, exist_ok=True)

    for index, config in enumerate(generate_configurations(csv_path), 1):

        # Save each configuration in a dedicated file file.
        json_file = str(index) + "_" + config["experiment_name"] + ".json"
        with open(os.path.join(output, json_file), "w") as file:
            json.dump(
                {key: value for key, value in config.items() if value},
                file,
                indent=4,
            )

    logging.info(
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
