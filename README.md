
![ActiveDelta](https://github.com/RekerLab/ActiveDelta/assets/127516906/c5424406-01cc-4571-8179-8610d184ff3b)

## Overview

ActiveDelta leverages paired molecular representations to predict molecular improvements from the best current training compound to prioritize molecules for training set expansion.  

## Requirements
* [RDKit](https://www.rdkit.org/docs/Install.html)
* [scikit-learn](https://scikit-learn.org/stable/)
* [numpy](https://numpy.org/)
* [pandas](https://github.com/pandas-dev/pandas)

Comparison Models
* [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
* [ChemProp v1.5.2](https://github.com/chemprop/chemprop)

Given the larger size of delta datasets, we recommend using a GPU for significantly faster training.

To use ChemProp with GPUs, you will need:
* cuda >= 8.0
* cuDNN

<br />


## Descriptions of Folders

### Code

Python code for evaluating ActiveDelta and traditional models based on their ability to predict potency differences between two molecules, identify the most potent lead in external test sets, and perform exploitative active learning.

### Datasets

56 curated benchmarking training and test sets for potency prediction from the [SIMPD preprint](https://chemrxiv.org/engage/chemrxiv/article-details/6406049e6642bf8c8f10e189), 99 curated benchmarking training and test sets from the [SIMPD publication](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-023-00787-9), and 3 random splits of the training data used for our exploitative active learning evaluations.

### Results

Results from 5x10-fold cross-validation, external test set evaluations, and active learning.

<br />

## License

The copyrights of the software are owned by Duke University. As such, two licenses for this software are offered:
1. An open-source license under the GPLv2 license for non-commercial academic use.
2. A custom license with Duke University, for commercial use or uses without the GPLv2 license restrictions. 
