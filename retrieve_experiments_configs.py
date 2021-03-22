#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    The retrieve experiments configs module
    ======================

    Use it to get the configurations for running the experiments.
"""

import os
import csv
import json
import logging
import argparse

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

TMP_DIR = './tmp'
STEPS = ["normalization_params", "train", "prediction", "evaluation"]


def run(config):
    """
    Retrieve the configurations for the experiments from a config csv file.
    Save each configuration into TMP_DIR/experiment_name file.
    """
    os.makedirs(TMP_DIR, exist_ok=True)
    
    with open(config) as config_file:
        reader = csv.DictReader(config_file, delimiter=',')
        for index, row in enumerate(reader, 1):

            json_dict = {}
            # Get experiment name.
            assert row['experiment_name'] != ''
            json_dict['experiment_name'] = row['experiment_name']

            # Get steps as a list of names.
            json_dict['steps'] = row['steps'].split(';')

            # Get train/val/test folders.
            json_dict['data_paths'] = {}
            for set in ['train', 'val', 'test']:
                json_dict['data_paths'][set] = {}
                if set in ['train', 'val']:
                    for folder, key in zip(['images', 'labels', 'labels_json'], ['image', 'mask', 'json']):
                        if row[set] != '':
                            json_dict['data_paths'][set][key] = [
                                os.path.join(element, set, folder) for element in row[set].split(';')]
                        else:
                            json_dict['data_paths'][set][key] = []
                else:
                    for folder, key in zip(['images', 'labels_json'], ['image', 'json']):
                        if row[set] != '':
                            json_dict['data_paths'][set][key] = [
                                os.path.join(element, set, folder) for element in row[set].split(';')]
                        else:
                            json_dict['data_paths'][set][key] = []

            # Get restore model.
            if row['restore_model'] != '':
                json_dict['training'] = {'restore_model': row['restore_model']}
                if row['same_data'] != '':
                    json_dict['training']['same_data'] = row['same_data']

            # Save configuration file.
            json_file = str(index)+'_'+row['experiment_name']+'.json'
            with open(os.path.join(TMP_DIR, json_file), 'w') as file:
                json.dump({key: value for key, value in json_dict.items() if value}, file, indent=4)

    logging.info(f"Retrieved {index} experiment configurations from {config}"
                 if index > 1 else f"Retrieved {index} experiment configuration from {config}")


def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser(
        description="Script to retrieve the experiments configs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the configurations file')

    args = parser.parse_args()
    run(**(vars(args)))


if __name__ == '__main__':
    main()
