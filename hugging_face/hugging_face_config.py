# -*- coding: utf-8 -*-

import json


def get_config(config):
    with open(config, "r") as f:
        data = json.load(f)
    return data
