# Supplemental Material for "Evaluating the Utility of Conformal Prediction Sets for AI-Advised Image Labeling"

Authors: Dongping Zhang, Angelos Chatzimparmpas, Negar Kamali, Jessica Hullman

arXiv: https://arxiv.org/abs/2401.08876

## About this repository:

This directory contains all supplemental materials associated with the ACM CHI'24 submission titled [**Evaluating the Utility of Conformal Prediction Sets for AI-Advised Image Labeling**](https://arxiv.org/abs/2401.08876). Due to file size constraints, certain input and output files, including _ILSVRC 2012_ datasets and the fitted Bayesian models, are not included within this repository. Instead, directories have been created to store the modeling object, should viewers choose to reproduce the analysis. The structure and contents of the repository are detailed in the table below.

## Content

Our supplemental material includes seven directories indicating different components.

| Directory                | Content Description                                                                                                                                                                                                                                                                                                                                                              |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `1-stimuli`              | Includes a CSV file with indices of selected stimuli from the ILSVRC 2012 validation set. Contains: (1) class ID and names, (2) cross-entropy loss as predicted by our base classifier, (3) corruption type, (4) corruption severity, (5) categories (in- or out-of-distribution, difficulty, size), (6) Top-k prediction accuracy, and RAPS coverage with set size information. |
| `2-representatives`      | Contains a CSV file with 1000 rows, each row corresponding to a label class of ILSVRC 2012, with the chosen index of each label-representative image in a separate column.                                                                                                                                                                                                       |
| `3-class_hierarchy`      | Presents the label space hierarchy organized by WordNet in a JSON file named `wordnet-pruned.json`, pruned to ensure each label belongs to one parent category.                                                                                                                                                                                                                  |
| `4-videos`               | Contains 4 sub-directories named by conditions (`top1`, `top10`, `raps`, `nopred`). Each includes instructional videos demonstrating interface features: autocomplete, keyword search, bottom-up search, exploring predictions. `nopred` includes videos for autocomplete and keyword search only.                                                                               |
| `5-conformalization`     | Provides Python scripts for conformalizing our base classifier to produce Regularized Adaptive Prediction Sets (RAPS). Includes instructions for downloading necessary data and training the model, which can take several hours. Output is saved in `output/model`.                                                                                                             |
| `6-analysis`             | Includes R scripts to reproduce our analysis. R script `model-analysis.R` contains helper functions to reproduce results, and the collected response data is in `data/model.rds`. Recommends using R (version 4.3.1) and BRMS (version 2.20.1), and includes a `knitr`-generated HTML report from the RMD file.                                                                  |
| `7-qualitative_analysis` | Provides five CSV files for qualitative analysis. `readme.csv` offers an overview of open codes with descriptions. Four additional CSV files, named by `[#]-[condition]`, contain the coded strategies for each condition.                                                                                                                                                       |
