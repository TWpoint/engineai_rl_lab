# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Installation script for the 'engineai_rl_lab' python package."""

import os
import tomllib

from setuptools import find_namespace_packages, setup

# Obtain the extension data from the extension.toml file
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
# Read the extension.toml file
with open(os.path.join(EXTENSION_PATH, "config", "extension.toml"), "rb") as extension_file:
    EXTENSION_TOML_DATA = tomllib.load(extension_file)

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "numpy",
    "onnx",
    "PyYAML",
    "rsl-rl-lib>=5.4.1,<6",
    "torch",
    "wandb",
]

# Installation operation
setup(
    name="engineai_rl_lab",
    packages=find_namespace_packages(
        include=["engineai_rl_lab", "engineai_rl_lab.*"],
        exclude=["engineai_rl_lab.assets", "engineai_rl_lab.assets.*"],
    ),
    author=EXTENSION_TOML_DATA["package"]["author"],
    maintainer=EXTENSION_TOML_DATA["package"]["maintainer"],
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    install_requires=INSTALL_REQUIRES,
    license="BSD-3-Clause",
    include_package_data=True,
    package_data={"engineai_rl_lab": ["assets/**/*", "assets/**/**/*"]},
    python_requires=">=3.12,<3.13",
    classifiers=[
        "Natural Language :: English",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.12",
    ],
    zip_safe=False,
)
