## Code for Model Evaluation

#### cross_validations.py
* Test model performance using 5x10-fold cross-validation on benchmarking datasets.

#### exploitative_active_learning.py
* Test model performance during exploitative active learning starting from 2 random datapoints on benchmarking datasets.
* Contains methods to run 3 models (DeepDelta, ChemProp, and Random Forest) and random selection.

#### exploration_of_chemical_space.py
* t-SNE analysis of active learning results to map model exploration of chemical space.

#### external_test.py
* Test model performance on external test sets. 

#### models.py
* Functions for [DeepDelta](https://github.com/RekerLab/ActiveDelta), [ChemProp](https://github.com/chemprop/chemprop), and [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) machine learning models.






