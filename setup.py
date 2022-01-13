#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

from setuptools import find_packages, setup


def parse_requirements():
    path = Path(__file__).parent.resolve() / "requirements.txt"
    assert path.exists(), f"Missing requirements: {path}"
    return list(map(str.strip, path.read_text().splitlines()))


setup(
    name="doc-ufcn",
    version=open("VERSION").read(),
    description="Doc-UFCN",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Mélodie Boillet",
    author_email="boillet@teklia.com",
    url="https://gitlab.com/teklia/dla/doc-ufcn",
    install_requires=parse_requirements(),
    packages=find_packages(),
)
