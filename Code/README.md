## Code for Model Evaluation

#### models.py
* Functions for the following machine learning models:
  * [DeepDelta](https://github.com/RekerLab/ActiveDelta)
  * [ChemProp](https://github.com/chemprop/chemprop) 
  * [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)

#### cross_validations.py
* Test model performance using 5x10-fold cross-validation on 56 benchmarking datasets.

#### external_test.py
* Test model performance on 56 external test sets. 

#### scaffold_analysis.py
* Compare model performance on molecular pairs with shared scaffolds to pairs that do not share scaffolds. 

#### same_molecule_pairs.py
* Calculate property differences of same molecule pairs for Eq 1 (with same molecule for both inputs, predictions should be zero):
```math
DeepDelta(x,x) = 0. 
```

#### exploitative_active_learning.py
* Test model performance during exploitative active learning starting from 2 random datapoints on 56 benchmarking datasets.
* Contains methods for run 3 models (DeepDelta, ChemProp, and Random Forest) and random selection.

#### exploration_of_chemical_space.py
* t-SNE analysis of active learning results to map model exploration of chemical space.
