#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path

import requests

config_path = Path("~/.notify-slack-cfg").expanduser()
if not config_path.exists():
    error_msg = """
    ERROR: A Webhook URL is required for running notify-slack.
    Create yours here: https://my.slack.com/services/new/incoming-webhook/
    Once you have your KEY, please create a file at ${HOME}/.notify-slack-cfg containng only the KEY. Eg: T02TKKSAX/B246MJ6HX/WXt2BWPfNhSKxdoFNFblczW9
    """
    raise ValueError(error_msg)

SLACK_NOTIFY_ICON = ":ubuntu:"
SLACK_BOT_USERNAME = "Bash Notifier"
SLACK_WEBHOOK_SERVICE = config_path.read_text().strip()
SLACK_URL = f"https://hooks.slack.com/services/{SLACK_WEBHOOK_SERVICE}"
LOG_PATH = Path("DLA_train.log")
DLA_LENGTH = 18

ICONS = {
    "INFO": ":information_source: ",
    "WARN": ":warning: ",
    "ERROR": ":github_changes_requested: ",
}


def get_icon(message):
    for log_prefix, icon in ICONS.items():
        if message.startswith(log_prefix):
            return icon
    return ""


def read_log(log_file, number_of_lines):
    if log_file.exists() and number_of_lines > 0:
        # read `number_of_lines` last lines of the log file (to get the CER and WER if possible)
        lines = log_file.read_text().split("\n")[-number_of_lines:]
        if lines:
            return f"{log_file}:\n```" + "\n".join(lines) + "```"
    else:
        print(f"{log_file} doesn't exist")
    return ""


def run(message, log_file, number_of_lines):
    icon = get_icon(message)
    dla_log = read_log(log_file, number_of_lines)
    payload = {
        "text": f"{icon}{message}\n{dla_log} from {os.uname().nodename}:{Path().cwd()}",
        "icon_emoji": SLACK_NOTIFY_ICON,
        "username": SLACK_BOT_USERNAME,
    }
    res = requests.post(
        SLACK_URL,
        data=json.dumps(payload),
        # headers={'Content-Type':'application/x-www-form-urlencoded'}
    )
    assert res.status_code == 200


def main():
    parser = argparse.ArgumentParser("Send message to slack")
    parser.add_argument("message", help="Message to be sent to slack", type=str)
    parser.add_argument(
        "--log_file",
        help="Log file from where last N files will be included with the message. "
        "Use None as file name if don't want to include a log file.",
        type=Path,
        default=LOG_PATH,
    )
    parser.add_argument(
        "-N",
        "--number_of_lines",
        help="Number of lines to be included from the end of the log file."
        "Use 0 to not include a log file.",
        type=int,
        default=DLA_LENGTH,
    )
    args = parser.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
