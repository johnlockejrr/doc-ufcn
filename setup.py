#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

from setuptools import find_packages, setup

MODULE = "doc_ufcn"


def parse_requirements():
    path = Path(__file__).parent.resolve() / "requirements.txt"
    assert path.exists(), f"Missing requirements: {path}"
    return list(map(str.strip, path.read_text().splitlines()))


setup(
    name=MODULE,
    version=open("VERSION").read(),
    description="Doc-UFCN",
    author="Mélodie Boillet",
    author_email="boillet@teklia.com",
    install_requires=parse_requirements(),
    packages=find_packages(),
)
