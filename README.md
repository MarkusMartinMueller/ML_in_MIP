<!-- #region -->
# Aneurysm Detection

Created by Leo Seeger, Markus Müller, Ramona Bendias from TU Berlin.

## Introduction

We performed cerebral aneurysm detection and analysis. The data used in the experiments was provided by TU Berlin and the same as provided in the [CADA Challenge](https://cada.grand-challenge.org/). This work can be used for aneurysm detection, segmentation and rupture risk estimation.


## Project Structure

- **[templates](./templates)** (Python): Refactored Notebooks that are reusable for a specific task (e.g. model training, prediction, data exploration). Notebooks should be used in the following order: train, create_preds, postprocessing_evaluation. For Unet & PointNet we have different files.
- **[utils](./utils)**: Utility functions that are used across multiple notebooks/scripts. Installable via pip from project folder: `pip install -e ./utils`.
- **[models](./utils)**: The folder contains the final models.


<!-- #endregion -->
