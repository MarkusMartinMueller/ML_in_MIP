<!-- #region -->
# Aneurysm Detection

Created by Leo Seeger, Markus Müller, Ramona Bendias from TU Berlin.

## Introduction

We performed cerebral aneurysm detection and analysis. The data used in the experiments was provided by TU Berlin and the same as provided in the [CADA Challenge](https://cada.grand-challenge.org/). This work can be used for aneurysm detection, segmentation and rupture risk estimation.


## Project Structure

- **[develop](./develop)**: Experimental notebooks or scripts to try out new ideas and prepare experiments. Naming convention: `short-description`. If you cannot use a notebook and have multiple scripts/files for an experiment, create a folder with the same naming convention. Each file should be handled by one person only. Notebooks in the `develop` folder should for most cases not contain any output. In case you want to store output from experiment runs or visualizations, duplicate the notebook and move it to the `experiments` folder.
- **[experiments](./experiments)**: Refactored notebooks that contain valuable insights or results from experiments (e.g. visualizations, training runs). Notebooks should be refactored, documented, contain outputs, and use the following naming schema: `YYYY-MM-DD_short-description`. Notebooks in the experiments folder should not be changed or rerun. If you want to rerun a Notebook from the experiment folder, please duplicate it into the `develop` folder.
- **[templates](./templates)** (Python): Refactored Notebooks that are reusable for a specific task (e.g. model training, data exploration). Notebooks should be refactored, documented, not contain any output, and use the following naming schema: `short-description`. If you like to make use of a template Notebook, duplicate the notebook into `develop` folder.
- **[utils](./utils)**: Utility functions that are distilled from the research phase and used across multiple notebooks/scripts. Should only contain refactored and tested Python scripts/modules. Installable via pip from project folder: `pip install -e ./utils`.

## Results


<!-- #endregion -->
