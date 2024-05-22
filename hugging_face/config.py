from pathlib import Path

from teklia_toolbox.config import ConfigParser


def parse_configurations(config_path: Path):
    """
    Parse multiple YAML configuration files into a single source
    of configuration for the HuggingFace app

    :param config_path: pathlib.Path, Path to the .yaml config file
    :return: dict, containing the configuration. Ensures config is complete and with correct typing
    """
    parser = ConfigParser()

    parser.add_option("title")
    parser.add_option("description")
    parser.add_option("examples", type=list)
    model_parser = parser.add_subparser("models", many=True)

    model_parser.add_option("model_name")
    model_parser.add_option("title")
    model_parser.add_option("description")
    model_parser.add_option("classes_colors", type=list)

    return parser.parse(config_path)
