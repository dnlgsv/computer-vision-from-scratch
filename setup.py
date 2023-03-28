#!\usr\bin\python
"""Setuptools-based installation."""
from pathlib import Path

from setuptools import find_packages, setup

from transformers import __name__ as description
from transformers.version import __version__

readme = Path(__file__).parent / "README.md"

if readme.is_file():
    LONG_DESCRIPTION = (Path(__file__).parent / "README.md").open(encoding="utf-8").read()
else:
    LONG_DESCRIPTION = description

with Path("requirements.txt").open() as file:
    dependencies = [line.rstrip() for line in file if not line.startswith("-") and "@git" not in line]


setup(
    name="transformers",
    packages=find_packages(),
    description=description,
    long_description=LONG_DESCRIPTION,
    author="PetProject",
    version=__version__,
    include_package_data=True,
    install_requires=dependencies,
)
