<h1 align="center">AUDIO SPLIT</h1>

_<h4 align="center">Extract indivudal instruments from an audio file.</h4>_

Most important points of this project's structure :
- Using [pyproject.toml](pyproject.toml) instead of requirements.txt
- Splitting package into smaller modules (see [project's documentation](/documentation/README.md))
- Training [models into sub apps](/apps/models/README.md)
- [Custom audiosplit cli](/audiosplit/cli/README.md) for easier workflow
- Extensive [.gitignore](.gitignore)
- Code quality tools (see [here](/documentation/best_practices.md#code-quality) on how to use them)
- [Makefile](Makefile) with commands shortcuts

## Summary

- [Summary](#summary)
- [Getting Started](#getting-started)
  - [Requirements](#requirements)
  - [Setup](#setup)
  - [Working on the project](#working-on-the-project)
- [Project structure](#project-structure)
  - [Package](#package)
  - [Apps](#apps)
  - [Overall architecture](#overall-architecture)
- [Raising issues](#raising-issues)

## Getting Started

The following section describes how to get started with this template.

### Requirements

**Pyenv (Required)**

```bash
pyenv --version
# pyenv 2.5.0
```

**Make (Optionnal)**

```bash
make --version
# GNU Make 4.3
# Built for x86_64-pc-linux-gnu
# Copyright (C) 1988-2020 Free Software Foundation, Inc.
# License GPLv3+: GNU GPL version 3 or later <http://gnu.org/licenses/gpl.html>
# This is free software: you are free to change and redistribute it.
# There is NO WARRANTY, to the extent permitted by law.
```

**Git (Optionnal)**

```bash
git --version
git version 2.34.1
```

### Setup

Configure python.

```bash
make pyenv-setup
make env-setup
```

Install the dependencies.

```bash
# Run this command when you update the pyproject.toml
make install

# Install the dev and test dependencies
make dev-install
make test-install
```

### Working on the project

See [this document on contribution guidelines](/CONTRIBUTING).

See [this document on coding best practices](/documentation/best_practices.md) and [this document on git and github workflow](/documentation/git_github_workflow.md) to see what is an ideal workflow for this project.

**Generate new model**

Since we are bound to create lots of models, use the following utils to create your models.

```bash
make new-model MODEL_NAME="model_name"
```

> This will generate a new model inside the [models](/apps/models/) directory, with a `main.py` and a `README.md`.  
> The `main.py` is generated from [this template file](/assets/templates/model_template.py) so update it for more complete starter models.

Add all the code you need from the package, and use this new file to create your model's architecture, and other required code. If you are implementing a feature multiple time in your models, maybe it can be written once inside the package.

## Project structure

This project is split into two main parts : the [package](#package), and the [apps](#apps).

There are also additional directories that can be used :
- [assets](assets/README.md) to store images and other documents
- [documentation](documentation/README.md) to store project's documentation, researches, etc...
- [notebooks](notebooks/README.md) to store exploratory works
- [scripts](scripts/README.md) for standalone scripts (cleaning, downloads, ...)
- [data](data/README.md) for storing ligthweight data files (data samples, best models, ...)
These directories are here to organize the project but are 

**All the code concerning this project must be contained within this github repository.**

### Package

The package contains all the source code for data handling, data processing, model building, cli, etc... It's the code common to all contributors, and that will be shared between the API, the interface, the docker image, and so on.

More informations about the package can be found [here](audiosplit/README.md).

### Apps

The apps are all standalone scripts that use the package. In this directory, we will store the API, the frontend, etc...

> Right now, models are created within this directory as well. This can be changed if this creates to much developer friction.

More informations about the package can be found [here](apps/README.md).

### Overall architecture

```bash
ds_project_template/
├── apps
│   ├── models
│   │   ├── model_1
│   │   │   ├── cache
│   │   │   └── main.py
│   │   └── README.md
│   └── website
│       └── README.md
├── data
│   └── README.md
├── documentation
├── makefile
├── package
│   ├── cli
│   │   ├── cli_main.py
│   │   └── README.md
│   ├── config
│   │   ├── environment.py
│   │   ├── logger.py
│   │   └── README.md
│   ├── data
│   │   ├── database.py
│   │   ├── loader.py
│   │   ├── preprocessing.py
│   │   └── README.md
│   ├── __init__.py
│   ├── measures
│   │   └── README.md
│   └── models
│       └── README.md
├── pyproject.toml
├── README.md
├── requirements.txt
└── scripts
    └── README.md
```

## Raising issues

If you find issues or misleading informations, feel free to raise an issue.
