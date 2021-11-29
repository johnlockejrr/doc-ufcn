# -*- coding: utf-8 -*-
import hashlib
import os

import requests
import yaml


def md5sum(path):
    with open(path, mode="rb") as f:
        d = hashlib.md5()
        while True:
            buf = f.read(4096)
            if not buf:
                break
            d.update(buf)
        return d.hexdigest()


def upload(path, name, version):
    token = os.environ.get("CI_JOB_TOKEN")
    assert token, "Missing CI token"
    project_id = os.environ.get("CI_PROJECT_ID")
    assert project_id, "Missing CI project ID"

    headers = {
        "JOB-TOKEN": token,
    }
    filename = os.path.basename(path)
    url = f"https://gitlab.com/api/v4/projects/{project_id}/packages/generic/{name}/{version}/{filename}"
    print(url)
    r = requests.post(url, headers=headers, data=open(path, "rb"))
    print(r)
    print(r.content)
    r.raise_for_status()


def publish_model(model_name):

    # Check required files are present
    model_path = os.path.join("models", model_name, "model.pth")
    parameters_path = os.path.join("models", model_name, "parameters.yml")
    assert os.path.exists(model_path), f"Missing model {model_path}"
    assert os.path.exists(parameters_path), f"Missing parameters {parameters_path}"

    # Parse parameters
    parameters = yaml.safe_load(open(parameters_path))
    version = parameters.get("version")
    assert version, f"Missing version in {parameters_path}"
    print(f"Publishing {model_name} @ {version}")

    # Hash model
    model_hash = md5sum(model_path)

    # Retrieve remote parameters to check for hash

    # Add version to the parameters
    with open(parameters_path, "w") as f:
        parameters["md5sum"] = model_hash
        yaml.safe_dump(parameters, f)

    # Upload the model on generic package registry
    upload(model_path, model_name, version)
    upload(parameters_path, model_name, version)


if __name__ == "__main__":
    assert os.environ.get("CI"), "Can only run on Gitlab CI"
    for p in os.listdir("models"):
        publish_model(p)
