import argparse
import logging
from pathlib import Path

from doc_ufcn.train.configuration import parse_configurations, save_configuration
from doc_ufcn.train.experiment import run

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


logger = logging.getLogger(__name__)


def parse_args():
    """
    Configure the CLI arguments for the training workflow
    """
    parser = argparse.ArgumentParser(description="Run a Doc-UFCN training experiment")
    parser.add_argument(
        "config",
        type=Path,
        nargs="+",
        help="Path towards an experiment or global configuration file. All files will be merged in order.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Parse all configuration files provided from CLI
    config = parse_configurations(args.config)

    # Ensure we have at least one step to run
    if len(config["steps"]) == 0:
        logger.error("No step to run, exiting execution.")
        return

    # Save configuration to be able to re-run the experiment
    save_configuration(config)

    # Run experiment
    run(config)


if __name__ == "__main__":
    main()
