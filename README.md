# SPQR: Streaming Product Quantization for MoleculaR data

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/afloresep/SPiQ/blob/master/.github/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-0.1.0-blue)](https://github.com/afloresep/SPiQ)

SPQR is a Python library designed for large-scale clustering of molecular data using Streaming Product Quantization (PQ). It allows you to process billions of molecules in a streaming fashion, transforming SMILES strings into compact PQ-codes for efficient clustering.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Pipeline Example](#pipeline-example)
  - [Tutorial Notebook](#tutorial-notebook)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

SPQR leverages the concept of Streaming Product Quantization to cluster high-dimensional molecular data without requiring the entire dataset to be in memory. Starting from SMILES strings, SPQR calculates molecular fingerprints, applies PQ encoding, and ultimately clusters the data efficiently.

## Features

- **Scalability:** Process data in chunks for datasets that don't fit in memory.
- **Efficiency:** Drastically reduce memory usage by converting high-dimensional fingerprints to compact PQ-codes.
- **Modular Design:** Separate modules for encoding, clustering, and data streaming.
- **Ease of Use:** Simple API with well-documented functions and usage examples.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/afloresep/spqr.git
   cd spqr
   ```

	2.	Create a virtual environment and install dependencies:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

	3.	Optional: Install the package in editable mode:

    ```bash
    pip install -e . 
    ```

## Usage

### Pipeline Example

The entire pipeline from SMILES strings to clustering can be run via the main script:

```bash
python scripts/main.py
```

This script integrates all modules (from data streaming to fingerprint calculation and PQ encoding) for clustering molecular data.

### Tutorial Notebook

For an interactive tutorial, check out `examples/tutorial.ipynb`. 

## Project Structure

```bash
├── data
│   ├── data_lite.txt           # Data for the tutorial example
├── docs                        # Documentation (Sphinx configuration, guides, etc.)
├── examples
│   ├── tutorial.ipynb          # Notebook demonstrating the API usage
├── pyproject.toml
├── README.md
├── requirements.txt
├── scripts
│   └── main.py                 # Main pipeline: SMILES -> cluster
├── setup.py
├── spiq
│   ├── clustering              # Clustering modules and implementations
│   ├── encoder
│   │   ├── encoder_base.py
│   │   ├── encoder.py
│   │   └── __init__.py
│   ├── __init__.py
│   ├── streamer
│   │   ├── data_streamer.py
│   │   └── __init__.py
│   └── utils
│       ├── fingerprints.py
│       ├── helper_functions.py
│       └── __init__.py
└── tests                       # Unit tests for all modules
    ├── __init__.py
    ├── test_clustering.py
    ├── test_data_streamer.py
    ├── test_encoder.py
    ├── test_fingerprints.py
    ├── test_trainer.py
    └── test_utils.py
```

# Documentation
- API Reference: Documentation is auto-generated from the code’s docstrings using Sphinx. See the docs folder for more details.
- Tutorials & Guides: Refer to the Jupyter Notebook in examples/tutorial.ipynb for a hands-on introduction.

# Testing

During the development I've been writing different tests to ensure the key functionalities remain working as expected after some changes. To run all the tests use:

```bash
pytest tests/
```

If instead you want to run a single group of test -for the data_streamer module for instance-, you can do: 

```bash
pytest test/test_data_streamer.py
```


# Contributing
Contributions are welcome! Please follow these guidelines:
1.	Fork the repository and create your branch: `git checkout -b feature/my-feature`
2.	Ensure your proper docstrings.
3.	(Ideally) Write tests for new features.
4.	Open a pull request describing your changes.

# License
This project is licensed under the MIT License. See the LICENSE file for details.

