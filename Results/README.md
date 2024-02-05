## Cross_Validation_Results

#### ChemProp50_CV
* Provides results for 5x10-fold cross-validation on 99 cross-merged benchmarking datasets for ChemProp with 50 epochs.

#### DeepDelta5_CV
* Provides results for 5x10-fold cross-validation on 99 cross-merged benchmarking datasets for the DeepDelta approach with 5 epochs.

#### RandomForest_CV
* Provides results for 5x10-fold cross-validation on on 99 cross-merged benchmarking datasets for Random Forest.



## External_Test_Results

#### ChemProp50_Ext_Test
* Provides results for 99 cross-merged external test sets for ChemProp with 50 epochs.

#### DeepDelta5_Ext_Test
* Provides results for 99 cross-merged external test sets for the DeepDelta approach with 5 epochs.

#### RandomForest_Ext_Test
* Provides results for 99 cross-merged external test sets for Random Forest.

  

## Exploitative_Active_Learning_Results

#### AL_Exploitative_AD_R1, AL_Exploitative_AD_R2, & AL_Exploitative_AD_R3
* Provides results for 3 repeats of exploitative active learning for 200 iterations starting from 2 random molecules for the ActiveDelta approach with 5 epochs.

#### AL_Exploitative_CP_R1, AL_Exploitative_CP_R2, & AL_Exploitative_CP_R3
* Provides results for 3 repeats of exploitative active learning for 200 iterations starting from 2 random molecules for ChemProp with 50 epochs.

#### AL_Exploitative_RF_R1, AL_Exploitative_RF_R2, & AL_Exploitative_RF_R3
* Provides results for 3 repeats of exploitative active learning for 200 iterations starting from 2 random molecules for Random Forest with 50 epochs.

#### AL_Exploitative_Random_R1, AL_Exploitative_Random_R2, & AL_Exploitative_Random_R3
* Provides results for 3 repeats of random molecule selection for 200 iterations starting from 2 random molecules.

#### AL100_External_Test_AD_R1, AL100_External_Test_AD_R2, & AL100_External_Test_AD_R3
* Provides results for external test sets for ActiveDelta after training on 100 molecules selected from exploitative active learning.

#### AL100_External_Test_CP_R1, AL100_External_Test_CP_R2, & AL100_External_Test_CP_R3
* Provides results for external test sets for ChemProp after training on 100 molecules selected from exploitative active learning.

#### AL100_External_Test_RF_R1, AL100_External_Test_RF_R2, & AL100_External_Test_RF_R3
* Provides results for external test sets for Random Forest after training on 100 molecules selected from exploitative active learning.

#### Plotting
* Contains results from first repeat of active learning with SMILES, true value, predicted value, and the iteration each molecule was added for the three models and random selection. These results are used for the plotting of chemical space through t-SNE.
