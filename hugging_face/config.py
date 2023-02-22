# -*- coding: utf-8 -*-

from pathlib import Path

from teklia_toolbox.config import ConfigParser


def parse_configurations(config_path: Path):
    """
    Parse multiple JSON configuration files into a single source
    of configuration for the HuggingFace app

    :param config_path: pathlib.Path, Path to the .json config file
    :return: dict, containing the configuration. Ensures config is complete and with correct typing
    """

    parser = ConfigParser()

    parser.add_option(
        "model_name", type=str, default="doc-ufcn-generic-historical-line"
    )
    parser.add_option("classes_colors", type=list, default=["green", "green"])
    parser.add_option("title", type=str)
    parser.add_option("description", type=str)
    parser.add_option("examples", type=list)

    return parser.parse(config_path)
