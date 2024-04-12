#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

from setuptools import find_packages, setup


def parse_requirements(name):
    path = Path(__file__).parent.resolve() / name
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
    project_urls={
        "Source": "https://gitlab.teklia.com/dla/doc-ufcn/",
        "Tracker": "https://gitlab.teklia.com/dla/doc-ufcn/issues/",
    },
    url="https://gitlab.teklia.com/dla/doc-ufcn",
    install_requires=parse_requirements("requirements.txt"),
    extras_require={
        "training": parse_requirements("training-requirements.txt"),
    },
    packages=find_packages(exclude=["tests"]),
    python_requires=">= 3.10, < 3.11",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        # Specify the Python versions you support here.
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.10",
        # Topics
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
)
