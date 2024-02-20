## Code for Model Evaluation

#### chemical_diversity_analysis.py
* Evaluate the scaffold composition and Tanimoto similarity of selected leads during active learning.

#### exploitative_active_learning.py
* Test model performance during exploitative active learning starting from 2 random datapoints on benchmarking datasets.
* Contains methods to run 3 models (DeepDelta, ChemProp, and Random Forest) and random selection.

#### exploration_of_chemical_space.py
* t-SNE analysis of active learning results to map model exploration of chemical space.

#### models.py
* Functions for [ActiveDelta](https://github.com/RekerLab/ActiveDelta), [ChemProp](https://github.com/chemprop/chemprop), and [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) machine learning models.






