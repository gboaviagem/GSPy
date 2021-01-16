#!/usr/bin/env python
"""Setup to Gleam pip package."""

from setuptools import setup
import os

thelibFolder = os.path.dirname(os.path.realpath(__file__))
requirementPath = thelibFolder + '/requirements.txt'
install_requires = []
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requires = f.read().splitlines()
install_requires = list(filter(lambda x: x[:2] != "--", install_requires))

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="gspy",
    version="0.0.1",
    author="Guilherme Boaviagem",
    author_email="guilherme.boaviagem@gmail.com",
    description="Utilities for Graph Signal Processing",
    install_requires=install_requires,
    packages=['gspy'])
