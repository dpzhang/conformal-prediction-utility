# Supplemental Material for "Evaluating the Utility of Conformal Prediction Sets for AI-Advised Image Labeling"
Authors: Dongping Zhang, Angelos Chatzimparmpas, Negar Kamali, Jessica Hullman
arXiv: https://arxiv.org/abs/2401.08876

## About this repository:
This directory includes all supplemental materials of our ACM CHI'24 paper titled "Evaluating the Utility of Conformal Prediction Sets for AI-Advised Image Labeling". Some input and output files, such as ILSVRC 2012 and the fitted models, are not included due to file size, but we have created a corresponding sub-directory to store the output objects. I present detailed description below.

## Content
Our supplemental material includes seven directories indicating different components. 

| Directory           | Content Description                                                                                                                                                                                                                                                                                                                                              |
|---------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `1-stimuli`         | Includes a CSV file with indices of selected stimuli from the ILSVRC 2012 validation set. Contains: (1) class ID and names, (2) cross-entropy loss as predicted by our base classifier, (3) corruption type, (4) corruption severity, (5) categories (in- or out-of-distribution, difficulty, size), (6) Top-k prediction accuracy, and RAPS coverage with set size information. |
| `2-representatives` | Contains a CSV file with 1000 rows, each row corresponding to a label class of ILSVRC 2012, with the chosen index of each label-representative image in a separate column.                                                                                                                                                                                           |
| `3-class_hierarchy` | Presents the label space hierarchy organized by WordNet in a JSON file named `wordnet-pruned.json`, pruned to ensure each label belongs to one parent category.                                                                                                                                                                                                    |
| `4-videos`          | Contains 4 sub-directories named by conditions (`top1`, `top10`, `raps`, `nopred`). Each includes instructional videos demonstrating interface features: autocomplete, keyword search, bottom-up search, exploring predictions. `nopred` includes videos for autocomplete and keyword search only.                                                                    |
| `5-conformalization`| Provides Python scripts for conformalizing our base classifier to produce Regularized Adaptive Prediction Sets (RAPS). Includes instructions for downloading necessary data and training the model, which can take several hours. Output is saved in `output/model`.                                                                                                  |
| `6-analysis`        | Includes R scripts to reproduce our analysis. R script `model-analysis.R` contains helper functions to reproduce results, and the collected response data is in `data/model.rds`. Recommends using R (version 4.3.1) and BRMS (version 2.20.1), and includes a `knitr`-generated HTML report from the RMD file.                                                        |
| `7-qualitative_analysis` | Provides five CSV files for qualitative analysis. `readme.csv` offers an overview of open codes with descriptions. Four additional CSV files, named by `[#]-[condition]`, contain the coded strategies for each condition.                                                                                                                                             |



│
├── 1-stimuli/
│   ├── stimuli.csv : selected stimuli from ILSVRC 2012 for the main experiment.
│
├── 2-representatives/
│   └── representatives.csv : selected representative for the experiment
│ 
├── 3-class_hierarchy/
│   └── wordnet-pruned.json : label space hierarchy based on WordNet we used after pruning duplicated categories
│
├── 4-videos/
│   ├── top1/
│   │   ├── top1-bottomup-search.mp4 : video demonstrating how to use bottom-up search of the interface
│   │   ├── top1-autocomplete.mp4 : video demonstrating the autocomplete feature of the interface
│   │   ├── top1-keyword-search.mp4 : video demonstrating the keyword search of the interface
│   │   └── top1-explore-submit.mp4 : video demonstrating how to explore predictions (e.g., click, and hover)
│   │
│   ├── nopred/
│   │   ├── nopred-keyword-search.mp4
│   │   └── nopred-autocomplete.mp4
│   │
│   ├── top10/
│   │   ├── top10-autocomplete.mp4
│   │   ├── top10-bottomup-search.mp4
│   │   ├── top10-explore-submit.mp4
│   │   └── top10-keyword-search.mp4
│   │
│   ├── raps/
│   │   ├── raps-explore-submit.mp4
│   │   ├── raps-keyword-search.mp4
│   │   ├── raps-autocomplete.mp4
│   │   └── raps-bottomup-search.mp4
│   │
│
├── 5-conformalization/
│   ├── output/
│   │   ├── model/ : directory to store the conformal model to create prediction sets. 
│   │   │
│   │
│   ├── scripts/
│   │   ├── train_conformal.py : python script used to train conformal model by `python3 train_conformal.py`
│   │   ├── classCP.py : module for RAPS. 
│   │   ├── utils_classCP.py : helper functions for `classCP.py`
│   │   └── utils_cp.py : helper functions to facilitate model training and to create prediction sets.
│   │
│   ├── data/
│   │   ├── imagenet-simple-labels.json : class labels we used for ILSVRC 2012.
│   │
│
├── 6-analysis/
│   │
│   ├── data/
│   │   └── model.rds : analysis data used for accuracy and shortest path model fitting
│   │
│   ├── results.Rmd : RMarkdown file showing code to reproduce our results
│   ├── model-analysis.R : helper functions to produce results
│   └── results.html : Report that shows how to reproduce our results
│
├── 7-qualitative_analysis/
│   ├── readme.csv : open code categories
│   ├── 0-baseline.csv : labeled quotes for baseline condition
│   ├── 1-top1.csv : labeled quotes for Top-1 condition
│   ├── 2-top10.csv : labeled quotes for Top-10 condition
│   └── 3-raps.csv : labeled quotes for RAPS condition
│
└── readme.txt