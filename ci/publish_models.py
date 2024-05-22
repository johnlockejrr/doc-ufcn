import hashlib
import os
from pathlib import Path

import requests
import yaml

assert os.environ.get("CI"), "Can only run on Gitlab CI"
TOKEN = os.environ.get("CI_JOB_TOKEN")
assert TOKEN, "Missing CI token"
PROJECT_ID = os.environ.get("CI_PROJECT_ID")
assert PROJECT_ID, "Missing CI project ID"


def md5sum(path: Path) -> str:
    """
    Calc the MD5 hash of a binary file
    """
    with path.open(mode="rb") as f:
        d = hashlib.md5()
        while True:
            buf = f.read(4096)
            if not buf:
                break
            d.update(buf)
        return d.hexdigest()


def remote_parameters(name: str, version: str) -> dict:
    """
    Fetch the parameters of a pre-existing model
    stored on the package registry
    """
    headers = {
        "JOB-TOKEN": TOKEN,
    }
    url = f"https://gitlab.teklia.com/api/v4/projects/{PROJECT_ID}/packages/generic/{name}/{version}/parameters.yml"
    r = requests.get(url, headers=headers)
    if not r.ok:
        # Normal case where the model is not available
        if r.status_code == 404:
            return

        r.raise_for_status()

    # Parse the parameters as YAML
    return yaml.safe_load(r.content)


def upload(path: Path, name: str, version: str) -> None:
    """
    Upload any file on the Gitlab generic package registry
    """
    headers = {
        "JOB-TOKEN": TOKEN,
    }
    url = f"https://gitlab.teklia.com/api/v4/projects/{PROJECT_ID}/packages/generic/{name}/{version}/{path.name}"
    r = requests.put(url, headers=headers, data=path.open(mode="rb"))
    r.raise_for_status()
    print(f"Uploaded {path}")


def publish_model(base_dir: Path) -> None:
    """
    Publish a single model on the Gitlab generic registry
    - will publish model.pth & parameters.yml
    - only if they are not already published for that version & model hash
    """
    # Check required files are present
    model_path = base_dir / "model.pth"
    parameters_path = base_dir / "parameters.yml"
    assert model_path.is_file(), f"Missing model {model_path}"
    assert parameters_path.is_file(), f"Missing parameters {parameters_path}"

    # Parse parameters
    parameters = yaml.safe_load(parameters_path.open())
    version = parameters.get("version")
    assert version, f"Missing version in {parameters_path}"
    print(f"Publishing {base_dir.name} @ {version}")

    # Hash model and update the parameters file
    parameters["md5sum"] = md5sum(model_path)

    # Retrieve remote parameters to check if they differ
    if parameters == remote_parameters(base_dir.name, version):
        print("Model is already available, skipping.")
        return

    # Add version to the parameters
    with parameters_path.open(mode="w") as f:
        yaml.safe_dump(parameters, f)

    # Upload the model on generic package registry
    upload(model_path, base_dir.name, version)
    upload(parameters_path, base_dir.name, version)


if __name__ == "__main__":
    for p in Path("models").iterdir():
        publish_model(p)
