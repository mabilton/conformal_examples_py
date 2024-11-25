# `conformal_examples_py`

This repository contains a collection of [`marimo` notebooks](https://github.com/marimo-team/marimo) demonstrating how to implement [*Split Conformal Prediction*](https://en.wikipedia.org/wiki/Conformal_prediction) 'from scratch' in Python. 

These notebooks were prepared to accompany a talk given by myself (Matt Bilton) and Lucia Lu for the 2024 New Zealand Society of Actuaries (NZSA) conference in Napier.

---

## Overview of Notebooks

There are currently (all of which can be found in the `src` directory):

1. `src/0-split-cp-intro.py` provides a simple introduction to split conformal prediction and 'numerically verifies' its coverage properties. 
2. `src/1-cp-vs-freq-vs-bayes.py` compares spit conformal prediction to 'classical' Frequentist and Bayesian methods. **This is currently a work in progress**.
3. `src/2-text-classification.py` applies split conformal prediction to a real-world text classification problem.
4. `src/3-motor-insurance.py` applies split conformal prediction to a motor insurance regression problem.

At this point in time, the notebooks contain limited explanatory text and the code itself is scarcely documented. Additional notebook commentary and code documentation will be added in the coming weeks :).

## Installation and Running Notebooks

The dependencies for this project are managed using the [`uv` package manager](https://github.com/astral-sh/uv). Please follow these steps to set up your environment:

### 1. Install `uv`

First, install `uv` using pip:

```bash
pip install uv
```

### 2. Clone the Repository

Clone this repository to your local machine:

```bash
git clone https://github.com/mabilton/conformal_examples_py.git
```

If you're behind a corporate firewall, you might be unable to `git clone` repository from Github. If so, you can download this repository as a zip file by clicking the green 'Code' button on the repository page and selecting 'Download ZIP'.

### 3. Install Dependencies using `uv`

Navigate your terminal to the cloned repository and execute

```bash
uv sync
```

to have `uv` automatically create a virtual environment and download the project dependencies to that environment. 

Once again, if you're behind a corporate firewall, you may have to 'tell' `uv` to download the dependencies from your company's `pypi` mirror (as opposed to the 'official' publically available index). You can do this by specifying the `--default-index` argument when calling `uv sync`. For example, if your `pypi` mirror url is `https://pypi.org/simple`, you would execute:

```bash
uv sync --default-index https://pypi.org/simple 
```

### 4. Running One of the Notebooks

You should now be able to run any of the notebooks in the `src` directory, say `0-split-cp-intro.py`, by running:

```bash
uv run marimo edit src/0-split-cp-intro.py 
```
