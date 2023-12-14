# Imports
import math
import statistics
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import KFold
from scipy import stats as stats
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs


############################
### Original 56 Datasets ###
############################ 


datasets = ['CHEMBL1267245',
 'CHEMBL1267247',
 'CHEMBL1267248',
 'CHEMBL1267250',
 'CHEMBL3705282',
 'CHEMBL3705362',
 'CHEMBL3705464',
 'CHEMBL3705542',
 'CHEMBL3705647',
 'CHEMBL3705655',
 'CHEMBL3705790',
 'CHEMBL3705791',
 'CHEMBL3705813',
 'CHEMBL3705899',
 'CHEMBL3705924',
 'CHEMBL3705960',
 'CHEMBL3705971',
 'CHEMBL3706037',
 'CHEMBL3706089',
 'CHEMBL3706310',
 'CHEMBL3706316',
 'CHEMBL3706373',
 'CHEMBL3707951',
 'CHEMBL3707962',
 'CHEMBL3721139',
 'CHEMBL3734252',
 'CHEMBL3734552',
 'CHEMBL3880337',
 'CHEMBL3880338',
 'CHEMBL3880340',
 'CHEMBL3887033',
 'CHEMBL3887061',
 'CHEMBL3887063',
 'CHEMBL3887188',
 'CHEMBL3887296',
 'CHEMBL3887679',
 'CHEMBL3887757',
 'CHEMBL3887758',
 'CHEMBL3887759',
 'CHEMBL3887796',
 'CHEMBL3887849',
 'CHEMBL3887887',
 'CHEMBL3887945',
 'CHEMBL3887987',
 'CHEMBL3888087',
 'CHEMBL3888190',
 'CHEMBL3888194',
 'CHEMBL3888268',
 'CHEMBL3888295',
 'CHEMBL3888825',
 'CHEMBL3888966',
 'CHEMBL3888977',
 'CHEMBL3888980',
 'CHEMBL3889082',
 'CHEMBL3889083',
 'CHEMBL3889139']
 
######################################
### Analysis from Cross-Validation ###
######################################

all_scores = pd.DataFrame(columns=['Dataset', 'MAE', 'RMSE'])

for dataset in datasets:
  scoring_intermediate = pd.DataFrame(columns=['MAE', 'RMSE'])

  # dataset loading:
  test_set = pd.read_csv('../Datasets/Original_56/Train/{}_train.csv'.format(dataset)) 

  for j in range(5):
    preds = pd.read_csv('../Results/Original_56/Cross_Validation_Results/DeepDelta5_CV/{}_DeepDelta5_{}.csv'.format(dataset, j)).T 
    preds.columns =['True', 'Delta']

    # Set up for cross validation
    cv = KFold(n_splits=10, random_state=1, shuffle=True)
    datapoint_x = []
    datapoint_y = []

    # Cross validation training of the model
    for train_index, test_index in cv.split(test_set):
      train_df = test_set[test_set.index.isin(train_index)]
      test_df = test_set[test_set.index.isin(test_index)]
      pair_subset_test = pd.merge(test_df, test_df, how='cross')
      datapoint_x += [pair_subset_test.SMILES_x]
      datapoint_y += [pair_subset_test.SMILES_y]
      del pair_subset_test

    datapoints = pd.DataFrame(data={'SMILES_x':  np.concatenate(datapoint_x), 'SMILES_y':  np.concatenate(datapoint_y)})

    trues = preds['True'].tolist()
    trues = [float(i) for i in trues]
    datapoints['True'] = trues

    Deltas = preds['Delta']
    Deltas = [float(i) for i in Deltas]
    datapoints['Delta_preds'] = Deltas
    
    # Grab the datapoints with matching SMILES 
    matching = datapoints[datapoints['SMILES_x'] == datapoints['SMILES_y']]

    # Calculate statistics
    MAE = metrics.mean_absolute_error(matching['True'],matching['Delta_preds'])
    RMSE = math.sqrt(metrics.mean_squared_error(matching['True'], matching['Delta_preds']))
    R2 = metrics.r2_score(matching['True'], matching['Delta_preds'])
    scoring = pd.DataFrame({'MAE': [round(MAE, 4)], 'RMSE': [round(RMSE, 4)]})
    scoring_intermediate = pd.concat([scoring, scoring_intermediate])

  #Statistics for all rounds
  average = pd.DataFrame({'MAE': [round(np.mean(scoring_intermediate['MAE']), 3)], 
                          'RMSE': [round(np.mean(scoring_intermediate['RMSE']), 3)]})
  std = pd.DataFrame({'MAE': [round(np.std(scoring_intermediate['MAE']), 3)], 
                      'RMSE': [round(np.std(scoring_intermediate['RMSE']), 3)]})

  #Make summary statistics 
  scores = pd.DataFrame({'Dataset': dataset,
                    'MAE': average["MAE"].astype(str) + "±" + std["MAE"].astype(str),
                    'RMSE': average["RMSE"].astype(str) + "±" + std["RMSE"].astype(str)})
  all_scores = pd.concat([all_scores, scores])

all_scores.to_csv("DeepDelta_CV_SameMolecularPairs_Original_56.csv", index = False) # Save results 




########################################
### Analysis from External Test Sets ###
########################################

all_scores = pd.DataFrame(columns=['Dataset', 'MAE', 'RMSE'])

for dataset in datasets:
  scoring_intermediate = pd.DataFrame(columns=['MAE', 'RMSE'])
  
  # dataset loading:
  test_set = pd.read_csv('../Datasets/Original_56/Test/{}_test.csv'.format(dataset)) 
  preds = pd.read_csv('../Results/Original_56/External_Test_Results/DeepDelta5_Ext_Test/{}_DeepDelta5_Test.csv'.format(dataset)).T 
  preds.columns =['True', 'Delta']

  datapoint_x = []
  datapoint_y = []

  pair_subset_test = pd.merge(test_set, test_set, how='cross')
  datapoint_x += [pair_subset_test.SMILES_x]
  datapoint_y += [pair_subset_test.SMILES_y]
  del pair_subset_test

  datapoints = pd.DataFrame(data={'SMILES_x':  np.concatenate(datapoint_x), 'SMILES_y':  np.concatenate(datapoint_y)})

  trues = preds['True'].tolist()
  trues = [float(i) for i in trues]
  datapoints['True'] = trues

  Deltas = preds['Delta']
  Deltas = [float(i) for i in Deltas]
  datapoints['Delta_preds'] = Deltas
  
  # Grab the datapoints with matching SMILES 
  matching = datapoints[datapoints['SMILES_x'] == datapoints['SMILES_y']]

  # Calculate statistics
  MAE = metrics.mean_absolute_error(matching['True'],matching['Delta_preds'])
  RMSE = math.sqrt(metrics.mean_squared_error(matching['True'], matching['Delta_preds']))
  R2 = metrics.r2_score(matching['True'], matching['Delta_preds'])
  scoring = pd.DataFrame({'MAE': [round(MAE, 3)], 'RMSE': [round(RMSE, 3)]})
  scoring_intermediate = pd.concat([scoring, scoring_intermediate])

  #Make Summary Statistics Easy to put into a table 
  scores = pd.DataFrame({'Dataset': dataset,  
                    'MAE': scoring_intermediate["MAE"], 
                    'RMSE': scoring_intermediate["RMSE"]})
  all_scores = pd.concat([all_scores, scores])

all_scores.to_csv("DeepDelta_Ext_SameMolecularPairs_Original_56.csv", index = False) # Save results 




###########################
### Updated 99 Datasets ###
###########################


properties = ['CHEMBL1075104-1',
'CHEMBL4153-1',
'CHEMBL4361-1',
'CHEMBL4361-2',
'CHEMBL4616-1',
'CHEMBL4792-1',
'CHEMBL4792-2',
'CHEMBL4794-1',
'CHEMBL4860-1',
'CHEMBL4908-1',
'CHEMBL4908-2',
'CHEMBL5071-1',
'CHEMBL5102-1',
'CHEMBL5112-1',
'CHEMBL5113-1',
'CHEMBL5113-2',
'CHEMBL302-1',
'CHEMBL313-1',
'CHEMBL313-2',
'CHEMBL318-1',
'CHEMBL318-2',
'CHEMBL344-1',
'CHEMBL344-2',
'CHEMBL3155-1',
'CHEMBL3227-1',
'CHEMBL3371-1',
'CHEMBL3510-1',
'CHEMBL3729-1',
'CHEMBL3759-1',
'CHEMBL3769-1',
'CHEMBL3837-1',
'CHEMBL3952-1',
'CHEMBL3952-2',
'CHEMBL3952-3',
'CHEMBL4005-1',
'CHEMBL4078-1',
'CHEMBL4105864-1',
'CHEMBL240-1',
'CHEMBL240-2',
'CHEMBL245-1',
'CHEMBL251-1',
'CHEMBL253-1',
'CHEMBL255-1',
'CHEMBL259-1',
'CHEMBL264-1',
'CHEMBL269-1',
'CHEMBL269-2',
'CHEMBL270-1',
'CHEMBL270-2',
'CHEMBL270-3',
'CHEMBL273-1',
'CHEMBL273-2',
'CHEMBL284-1',
'CHEMBL287-1',
'CHEMBL2492-1',
'CHEMBL2820-1',
'CHEMBL2954-1',
'CHEMBL222-1',
'CHEMBL222-2',
'CHEMBL223-1',
'CHEMBL224-1',
'CHEMBL224-2',
'CHEMBL225-1',
'CHEMBL225-2',
'CHEMBL228-1',
'CHEMBL228-2',
'CHEMBL229-1',
'CHEMBL229-2',
'CHEMBL231-1',
'CHEMBL232-1',
'CHEMBL233-1',
'CHEMBL234-1',
'CHEMBL236-1',
'CHEMBL237-1',
'CHEMBL238-1',
'CHEMBL2326-1',
'CHEMBL2366517-1',
'CHEMBL210-1',
'CHEMBL211-1',
'CHEMBL214-1',
'CHEMBL216-1',
'CHEMBL217-1',
'CHEMBL218-1',
'CHEMBL219-1',
'CHEMBL219-2',
'CHEMBL1800-1',
'CHEMBL1821-1',
'CHEMBL1833-1',
'CHEMBL1833-2',
'CHEMBL1862-1',
'CHEMBL1871-1',
'CHEMBL1889-1',
'CHEMBL1945-1',
'CHEMBL1946-1',
'CHEMBL2014-1',
'CHEMBL2035-1',
'CHEMBL2056-1',
'CHEMBL1293269-1',
'CHEMBL1908389-1']

######################################
### Analysis from Cross-Validation ###
######################################

all_scores = pd.DataFrame(columns=['Dataset', 'MAE', 'RMSE'])

for dataset in datasets:
  scoring_intermediate = pd.DataFrame(columns=['MAE', 'RMSE'])

  # dataset loading:
  test_set = pd.read_csv('../Datasets/Updated_99/Train/{}_train.csv'.format(dataset)) 

  for j in range(5):
    preds = pd.read_csv('../Results/Updated_99/Cross_Validation_Results/DeepDelta5_CV/{}_DeepDelta5_{}.csv'.format(dataset, j)).T 
    preds.columns =['True', 'Delta']

    # Set up for cross validation
    cv = KFold(n_splits=10, random_state=1, shuffle=True)
    datapoint_x = []
    datapoint_y = []

    # Cross validation training of the model
    for train_index, test_index in cv.split(test_set):
      train_df = test_set[test_set.index.isin(train_index)]
      test_df = test_set[test_set.index.isin(test_index)]
      pair_subset_test = pd.merge(test_df, test_df, how='cross')
      datapoint_x += [pair_subset_test.SMILES_x]
      datapoint_y += [pair_subset_test.SMILES_y]
      del pair_subset_test

    datapoints = pd.DataFrame(data={'SMILES_x':  np.concatenate(datapoint_x), 'SMILES_y':  np.concatenate(datapoint_y)})

    trues = preds['True'].tolist()
    trues = [float(i) for i in trues]
    datapoints['True'] = trues

    Deltas = preds['Delta']
    Deltas = [float(i) for i in Deltas]
    datapoints['Delta_preds'] = Deltas
    
    # Grab the datapoints with matching SMILES 
    matching = datapoints[datapoints['SMILES_x'] == datapoints['SMILES_y']]

    # Calculate statistics
    MAE = metrics.mean_absolute_error(matching['True'],matching['Delta_preds'])
    RMSE = math.sqrt(metrics.mean_squared_error(matching['True'], matching['Delta_preds']))
    R2 = metrics.r2_score(matching['True'], matching['Delta_preds'])
    scoring = pd.DataFrame({'MAE': [round(MAE, 4)], 'RMSE': [round(RMSE, 4)]})
    scoring_intermediate = pd.concat([scoring, scoring_intermediate])

  #Statistics for all rounds
  average = pd.DataFrame({'MAE': [round(np.mean(scoring_intermediate['MAE']), 3)], 
                          'RMSE': [round(np.mean(scoring_intermediate['RMSE']), 3)]})
  std = pd.DataFrame({'MAE': [round(np.std(scoring_intermediate['MAE']), 3)], 
                      'RMSE': [round(np.std(scoring_intermediate['RMSE']), 3)]})

  #Make summary statistics 
  scores = pd.DataFrame({'Dataset': dataset,
                    'MAE': average["MAE"].astype(str) + "±" + std["MAE"].astype(str),
                    'RMSE': average["RMSE"].astype(str) + "±" + std["RMSE"].astype(str)})
  all_scores = pd.concat([all_scores, scores])

all_scores.to_csv("DeepDelta_CV_SameMolecularPairs_Updated_99.csv", index = False) # Save results 




########################################
### Analysis from External Test Sets ###
########################################

all_scores = pd.DataFrame(columns=['Dataset', 'MAE', 'RMSE'])

for dataset in datasets:
  scoring_intermediate = pd.DataFrame(columns=['MAE', 'RMSE'])
  
  # dataset loading:
  test_set = pd.read_csv('../Datasets/Updated_99/Test/{}_test.csv'.format(dataset)) 
  preds = pd.read_csv('../Results/Updated_99/External_Test_Results/DeepDelta5_Ext_Test/{}_DeepDelta5_Test.csv'.format(dataset)).T 
  preds.columns =['True', 'Delta']

  datapoint_x = []
  datapoint_y = []

  pair_subset_test = pd.merge(test_set, test_set, how='cross')
  datapoint_x += [pair_subset_test.SMILES_x]
  datapoint_y += [pair_subset_test.SMILES_y]
  del pair_subset_test

  datapoints = pd.DataFrame(data={'SMILES_x':  np.concatenate(datapoint_x), 'SMILES_y':  np.concatenate(datapoint_y)})

  trues = preds['True'].tolist()
  trues = [float(i) for i in trues]
  datapoints['True'] = trues

  Deltas = preds['Delta']
  Deltas = [float(i) for i in Deltas]
  datapoints['Delta_preds'] = Deltas
  
  # Grab the datapoints with matching SMILES 
  matching = datapoints[datapoints['SMILES_x'] == datapoints['SMILES_y']]

  # Calculate statistics
  MAE = metrics.mean_absolute_error(matching['True'],matching['Delta_preds'])
  RMSE = math.sqrt(metrics.mean_squared_error(matching['True'], matching['Delta_preds']))
  R2 = metrics.r2_score(matching['True'], matching['Delta_preds'])
  scoring = pd.DataFrame({'MAE': [round(MAE, 3)], 'RMSE': [round(RMSE, 3)]})
  scoring_intermediate = pd.concat([scoring, scoring_intermediate])

  #Make Summary Statistics Easy to put into a table 
  scores = pd.DataFrame({'Dataset': dataset,  
                    'MAE': scoring_intermediate["MAE"], 
                    'RMSE': scoring_intermediate["RMSE"]})
  all_scores = pd.concat([all_scores, scores])

all_scores.to_csv("DeepDelta_Ext_SameMolecularPairs_Updated_99.csv", index = False) # Save results 