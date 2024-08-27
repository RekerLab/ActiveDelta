
![GraphicalAbstractSimple](https://github.com/user-attachments/assets/96af7b76-dcb8-43b0-b9da-3c454e0678a0)

## Overview

ActiveDelta is an adaptive active learning approach that leverages paired molecular representations to predict molecular improvements from the best current training compound to prioritize molecules for data aquisition.  

For more information, please refer to the [associated publication](https://www.beilstein-journals.org/bjoc/articles/20/185)

If you use this data or code, please kindly cite: Fralish, Z. & Reker, D. (2024). Finding the most potent compounds using active learning on
molecular pairs. Beilstein J. Org. Chem. 20, 2152-2162

We would like to thank the Chemprop, XGBoost, and the Scikit-learn developers for making their machine learning algorithms publicly available.

## Requirements
* [RDKit](https://www.rdkit.org/docs/Install.html)
* [scikit-learn](https://scikit-learn.org/stable/)
* [numpy](https://numpy.org/)
* [pandas](https://github.com/pandas-dev/pandas)

Base Machine Learning Models
* [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
* [ChemProp v1.5.2](https://github.com/chemprop/chemprop)
* [XGBoost](https://xgboost.readthedocs.io/en/stable/gpu/index.html)

Given the larger size of delta datasets, we recommend using a GPU for significantly faster training.

To use ChemProp with GPUs, you will need:
* cuda >= 8.0
* cuDNN

<br />


## Descriptions of Folders

### Code

Python code for evaluating ActiveDelta and traditional approaches based on their ability to identify the most potent leads during exploitative active learning.

### Datasets

99 curated benchmarking training and test sets from the [SIMPD publication](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-023-00787-9) and 3 random splits of the training data used for our exploitative active learning evaluations.

### Results

Results from exploitative active learning.

<br />

## License

The copyrights of the software are owned by Duke University. As such, two licenses for this software are offered:
1. An open-source license under the GPLv2 license for non-commercial academic use.
2. A custom license with Duke University, for commercial use or uses without the GPLv2 license restrictions. 
