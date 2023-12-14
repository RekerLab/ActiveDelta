# Imports
import math
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold 
from sklearn import metrics
from scipy import stats as stats
from sklearn.model_selection import KFold

############################
### Original 56 Datasets ###
############################

# Read local training data
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
 
 
########################
### Cross-Validation ###
########################
 

all_scoresNM = pd.DataFrame(columns=['Dataset', 'Pearson\'s r RF', 'MAE RF', 'RMSE RF',
                                    'Pearson\'s r CP', 'MAE CP', 'RMSE CP', 'Pearson\'s r DD', 'MAE DD', 'RMSE DD']) # For pairs of Nonmatching Scaffolds
all_scoresM = pd.DataFrame(columns=['Dataset', 'Pearson\'s r RF', 'MAE RF', 'RMSE RF',
                                    'Pearson\'s r CP', 'MAE CP', 'RMSE CP', 'Pearson\'s r DD', 'MAE DD', 'RMSE DD']) # For pairs with Matching Scaffolds

# Evaluate Matching and Non-matching Scaffold Pairs for all Datasets
for dataset in datasets:
    dataframe = pd.read_csv('../Datasets/Original_56/Train/{}_train.csv'.format(dataset)) # Training dataset

    # 3 Models to Evaluate
    # 1 - Random Forest 
    predictions_RF = pd.read_csv('../Results/Original_56/Cross_Validation_Results/RandomForest_CV/{}_RandomForest_1.csv'.format(dataset)).T
    predictions_RF.columns =['True', 'Delta']
    # 2 - ChemProp
    predictions_CP = pd.read_csv('../Results/Original_56/Cross_Validation_Results/ChemProp50_CV/{}_ChemProp50_1.csv'.format(dataset)).T
    predictions_CP.columns =['True', 'Delta']
    # 3 - DeepDelta
    predictions_DD = pd.read_csv('../Results/Original_56/Cross_Validation_Results/DeepDelta5_CV/{}_DeepDelta5_1.csv'.format(dataset)).T
    predictions_DD.columns =['True', 'Delta']

    # Prepare Scaffolds
    mols = [Chem.MolFromSmiles(s) for s in dataframe.SMILES]
    scaffolds = [Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m)) for m in mols]
    data = pd.DataFrame(data={'Scaffold':  scaffolds})
    del dataframe

    # Emulate previous train-test splits and save the scaffolds from this
    cv = KFold(n_splits=10, random_state=1, shuffle=True)
    datapoint_x = []
    datapoint_y = []

    for train_index, test_index in cv.split(data):
        train_df = data[data.index.isin(train_index)]
        test_df = data[data.index.isin(test_index)]
        pair_subset_test = pd.merge(test_df, test_df, how='cross')
        datapoint_x += [pair_subset_test.Scaffold_x]
        datapoint_y += [pair_subset_test.Scaffold_y]
        del pair_subset_test

    datapoints = pd.DataFrame(data={'X':  np.concatenate(datapoint_x), 'Y':  np.concatenate(datapoint_y)})

    # Add the actual deltas and predicted deltas
    trues = predictions_CP['True']
    trues = [float(i) for i in trues]
    datapoints['True'] = trues

    DeltasRF = predictions_RF['Delta']
    DeltasRF = [float(i) for i in DeltasRF]
    datapoints['DeltaRF'] = DeltasRF

    DeltasCP = predictions_CP['Delta']
    DeltasCP = [float(i) for i in DeltasCP]
    datapoints['DeltaCP'] = DeltasCP

    DeltasDD = predictions_DD['Delta']
    DeltasDD = [float(i) for i in DeltasDD]
    datapoints['DeltaDD'] = DeltasDD

    # Grab the datapoints with matching scaffolds 
    matching = datapoints[datapoints['X'] == datapoints['Y']]
    
    # Grab the nonmatching datapoints
    nonmatching = datapoints[datapoints['X'] != datapoints['Y']]

    # Run Stats - Non-matching Scaffolds
    pearson_NM_RF = stats.pearsonr(nonmatching["True"], (nonmatching['DeltaRF']))
    MAE_NM_RF = metrics.mean_absolute_error(nonmatching["True"], (nonmatching['DeltaRF']))
    RMSE_NM_RF = math.sqrt(metrics.mean_squared_error(nonmatching["True"], (nonmatching['DeltaRF'])))

    pearson_NM_CP = stats.pearsonr(nonmatching["True"], (nonmatching['DeltaCP']))
    MAE_NM_CP = metrics.mean_absolute_error(nonmatching["True"], (nonmatching['DeltaCP']))
    RMSE_NM_CP = math.sqrt(metrics.mean_squared_error(nonmatching["True"], (nonmatching['DeltaCP'])))

    pearson_NM_DD = stats.pearsonr(nonmatching["True"], (nonmatching['DeltaDD']))
    MAE_NM_DD = metrics.mean_absolute_error(nonmatching["True"], (nonmatching['DeltaDD']))
    RMSE_NM_DD = math.sqrt(metrics.mean_squared_error(nonmatching["True"], (nonmatching['DeltaDD'])))

    scoringNM = pd.DataFrame({'Dataset': [dataset], 'Pearson\'s r RF': [round(pearson_NM_RF[0], 4)], 'MAE RF': [round(MAE_NM_RF, 4)], 'RMSE RF': [round(RMSE_NM_RF, 4)],
                            'Pearson\'s r CP': [round(pearson_NM_CP[0], 4)], 'MAE CP': [round(MAE_NM_CP, 4)], 'RMSE CP': [round(RMSE_NM_CP, 4)],
                            'Pearson\'s r DD': [round(pearson_NM_DD[0], 4)], 'MAE DD': [round(MAE_NM_DD, 4)], 'RMSE DD': [round(RMSE_NM_DD, 4)]})

    all_scoresNM = pd.concat([all_scoresNM, scoringNM])

    # Run Stats - Matching Scaffolds
    pearson_M_RF = stats.pearsonr(matching["True"], (matching['DeltaRF']))
    MAE_M_RF = metrics.mean_absolute_error(matching["True"], (matching['DeltaRF']))
    RMSE_M_RF = math.sqrt(metrics.mean_squared_error(matching["True"], (matching['DeltaRF'])))

    pearson_M_CP = stats.pearsonr(matching["True"], (matching['DeltaCP']))
    MAE_M_CP = metrics.mean_absolute_error(matching["True"], (matching['DeltaCP']))
    RMSE_M_CP = math.sqrt(metrics.mean_squared_error(matching["True"], (matching['DeltaCP'])))

    pearson_M_DD = stats.pearsonr(matching["True"], (matching['DeltaDD']))
    MAE_M_DD = metrics.mean_absolute_error(matching["True"], (matching['DeltaDD']))
    RMSE_M_DD = math.sqrt(metrics.mean_squared_error(matching["True"], (matching['DeltaDD'])))

    scoringM = pd.DataFrame({'Dataset': [dataset], 'Pearson\'s r RF': [round(pearson_M_RF[0], 4)], 'MAE RF': [round(MAE_M_RF, 4)], 'RMSE RF': [round(RMSE_M_RF, 4)],
                            'Pearson\'s r CP': [round(pearson_M_CP[0], 4)], 'MAE CP': [round(MAE_M_CP, 4)], 'RMSE CP': [round(RMSE_M_CP, 4)],
                            'Pearson\'s r DD': [round(pearson_M_DD[0], 4)], 'MAE DD': [round(MAE_M_DD, 4)], 'RMSE DD': [round(RMSE_M_DD, 4)]})

    all_scoresM = pd.concat([all_scoresM, scoringM])

all_scoresNM.to_csv("DeepDelta_CV_Scaffold_NonMatching_Original_56.csv", index = False) # Save results for unmatched scaffold pairs
all_scoresM.to_csv("DeepDelta_CV_Scaffold_Matching_Original_56.csv", index = False) # Save results for matched scaffold pairs

#########################




##########################
### External Test Sets ###
##########################
 

all_scoresNM = pd.DataFrame(columns=['Dataset', 'Pearson\'s r RF', 'MAE RF', 'RMSE RF',
                                    'Pearson\'s r CP', 'MAE CP', 'RMSE CP', 'Pearson\'s r DD', 'MAE DD', 'RMSE DD']) # For pairs of Nonmatching Scaffolds
all_scoresM = pd.DataFrame(columns=['Dataset', 'Pearson\'s r RF', 'MAE RF', 'RMSE RF',
                                    'Pearson\'s r CP', 'MAE CP', 'RMSE CP', 'Pearson\'s r DD', 'MAE DD', 'RMSE DD']) # For pairs with Matching Scaffolds

# Evaluate Matching and Non-matching Scaffold Pairs for all Datasets
for dataset in datasets:
    dataframe = pd.read_csv('../Datasets/Original_56/Test/{}_test.csv'.format(dataset)) # External Test Sets

    # 3 Models to Evaluate
    # 1 - Random Forest 
    predictions_RF = pd.read_csv('../Results/Original_56/External_Test_Results/RandomForest_Ext_Test/{}_RandomForest_Test.csv'.format(dataset)).T
    predictions_RF.columns =['True', 'Delta']
    # 2 - ChemProp
    predictions_CP = pd.read_csv('../Results/Original_56/External_Test_Results/ChemProp50_Ext_Test/{}_ChemProp50_Test.csv'.format(dataset)).T
    predictions_CP.columns =['True', 'Delta']
    # 3 - DeepDelta
    predictions_DD = pd.read_csv('../Results/Original_56/External_Test_Results/DeepDelta5_Ext_Test/{}_DeepDelta5_Test.csv'.format(dataset)).T
    predictions_DD.columns =['True', 'Delta']

    # Prepare Scaffolds
    mols = [Chem.MolFromSmiles(s) for s in dataframe.SMILES]
    scaffolds = [Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m)) for m in mols]
    data = pd.DataFrame(data={'Scaffold':  scaffolds})
    del dataframe

    # Cross-Merge the External Test Sets and Extract the scaffolds
    datapoint_x = []
    datapoint_y = []

    pair_subset_test = pd.merge(data, data, how='cross')
    datapoint_x += [pair_subset_test.Scaffold_x]
    datapoint_y += [pair_subset_test.Scaffold_y]
    del pair_subset_test

    datapoints = pd.DataFrame(data={'X':  np.concatenate(datapoint_x), 'Y':  np.concatenate(datapoint_y)})

    # Add the actual deltas and predicted deltas
    trues = predictions_CP['True']
    trues = [float(i) for i in trues]
    datapoints['True'] = trues

    DeltasRF = predictions_RF['Delta']
    DeltasRF = [float(i) for i in DeltasRF]
    datapoints['DeltaRF'] = DeltasRF

    DeltasCP = predictions_CP['Delta']
    DeltasCP = [float(i) for i in DeltasCP]
    datapoints['DeltaCP'] = DeltasCP

    DeltasDD = predictions_DD['Delta']
    DeltasDD = [float(i) for i in DeltasDD]
    datapoints['DeltaDD'] = DeltasDD

    # Grab the datapoints with matching scaffolds 
    matching = datapoints[datapoints['X'] == datapoints['Y']]
    
    # Grab the nonmatching datapoints
    nonmatching = datapoints[datapoints['X'] != datapoints['Y']]

    # Run Stats - Non-matching Scaffolds
    pearson_NM_RF = stats.pearsonr(nonmatching["True"], (nonmatching['DeltaRF']))
    MAE_NM_RF = metrics.mean_absolute_error(nonmatching["True"], (nonmatching['DeltaRF']))
    RMSE_NM_RF = math.sqrt(metrics.mean_squared_error(nonmatching["True"], (nonmatching['DeltaRF'])))

    pearson_NM_CP = stats.pearsonr(nonmatching["True"], (nonmatching['DeltaCP']))
    MAE_NM_CP = metrics.mean_absolute_error(nonmatching["True"], (nonmatching['DeltaCP']))
    RMSE_NM_CP = math.sqrt(metrics.mean_squared_error(nonmatching["True"], (nonmatching['DeltaCP'])))

    pearson_NM_DD = stats.pearsonr(nonmatching["True"], (nonmatching['DeltaDD']))
    MAE_NM_DD = metrics.mean_absolute_error(nonmatching["True"], (nonmatching['DeltaDD']))
    RMSE_NM_DD = math.sqrt(metrics.mean_squared_error(nonmatching["True"], (nonmatching['DeltaDD'])))

    scoringNM = pd.DataFrame({'Dataset': [dataset], 'Pearson\'s r RF': [round(pearson_NM_RF[0], 4)], 'MAE RF': [round(MAE_NM_RF, 4)], 'RMSE RF': [round(RMSE_NM_RF, 4)],
                            'Pearson\'s r CP': [round(pearson_NM_CP[0], 4)], 'MAE CP': [round(MAE_NM_CP, 4)], 'RMSE CP': [round(RMSE_NM_CP, 4)],
                            'Pearson\'s r DD': [round(pearson_NM_DD[0], 4)], 'MAE DD': [round(MAE_NM_DD, 4)], 'RMSE DD': [round(RMSE_NM_DD, 4)]})

    all_scoresNM = pd.concat([all_scoresNM, scoringNM])

    # Run Stats - Matching Scaffolds
    pearson_M_RF = stats.pearsonr(matching["True"], (matching['DeltaRF']))
    MAE_M_RF = metrics.mean_absolute_error(matching["True"], (matching['DeltaRF']))
    RMSE_M_RF = math.sqrt(metrics.mean_squared_error(matching["True"], (matching['DeltaRF'])))

    pearson_M_CP = stats.pearsonr(matching["True"], (matching['DeltaCP']))
    MAE_M_CP = metrics.mean_absolute_error(matching["True"], (matching['DeltaCP']))
    RMSE_M_CP = math.sqrt(metrics.mean_squared_error(matching["True"], (matching['DeltaCP'])))

    pearson_M_DD = stats.pearsonr(matching["True"], (matching['DeltaDD']))
    MAE_M_DD = metrics.mean_absolute_error(matching["True"], (matching['DeltaDD']))
    RMSE_M_DD = math.sqrt(metrics.mean_squared_error(matching["True"], (matching['DeltaDD'])))

    scoringM = pd.DataFrame({'Dataset': [dataset], 'Pearson\'s r RF': [round(pearson_M_RF[0], 4)], 'MAE RF': [round(MAE_M_RF, 4)], 'RMSE RF': [round(RMSE_M_RF, 4)],
                            'Pearson\'s r CP': [round(pearson_M_CP[0], 4)], 'MAE CP': [round(MAE_M_CP, 4)], 'RMSE CP': [round(RMSE_M_CP, 4)],
                            'Pearson\'s r DD': [round(pearson_M_DD[0], 4)], 'MAE DD': [round(MAE_M_DD, 4)], 'RMSE DD': [round(RMSE_M_DD, 4)]})

    all_scoresM = pd.concat([all_scoresM, scoringM])

all_scoresNM.to_csv("DeepDelta_Ext_Scaffold_NonMatching_Original_56.csv", index = False) # Save results for unmatched scaffold pairs
all_scoresM.to_csv("DeepDelta_Ext_Scaffold_Matching_Original_56.csv", index = False) # Save results for matched scaffold pairs

#########################



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
 
 
########################
### Cross-Validation ###
########################
 

all_scoresNM = pd.DataFrame(columns=['Dataset', 'Pearson\'s r RF', 'MAE RF', 'RMSE RF',
                                    'Pearson\'s r CP', 'MAE CP', 'RMSE CP', 'Pearson\'s r DD', 'MAE DD', 'RMSE DD']) # For pairs of Nonmatching Scaffolds
all_scoresM = pd.DataFrame(columns=['Dataset', 'Pearson\'s r RF', 'MAE RF', 'RMSE RF',
                                    'Pearson\'s r CP', 'MAE CP', 'RMSE CP', 'Pearson\'s r DD', 'MAE DD', 'RMSE DD']) # For pairs with Matching Scaffolds

# Evaluate Matching and Non-matching Scaffold Pairs for all Datasets
for dataset in datasets:
    dataframe = pd.read_csv('../Datasets/Updated_99/Train/{}_train.csv'.format(dataset)) # Training dataset

    # 3 Models to Evaluate
    # 1 - Random Forest 
    predictions_RF = pd.read_csv('../Results/Updated_99/Cross_Validation_Results/RandomForest_CV/{}_RandomForest_1.csv'.format(dataset)).T
    predictions_RF.columns =['True', 'Delta']
    # 2 - ChemProp
    predictions_CP = pd.read_csv('../Results/Updated_99/Cross_Validation_Results/ChemProp50_CV/{}_ChemProp50_1.csv'.format(dataset)).T
    predictions_CP.columns =['True', 'Delta']
    # 3 - DeepDelta
    predictions_DD = pd.read_csv('../Results/Updated_99/Cross_Validation_Results/DeepDelta5_CV/{}_DeepDelta5_1.csv'.format(dataset)).T
    predictions_DD.columns =['True', 'Delta']

    # Prepare Scaffolds
    mols = [Chem.MolFromSmiles(s) for s in dataframe.SMILES]
    scaffolds = [Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m)) for m in mols]
    data = pd.DataFrame(data={'Scaffold':  scaffolds})
    del dataframe

    # Emulate previous train-test splits and save the scaffolds from this
    cv = KFold(n_splits=10, random_state=1, shuffle=True)
    datapoint_x = []
    datapoint_y = []

    for train_index, test_index in cv.split(data):
        train_df = data[data.index.isin(train_index)]
        test_df = data[data.index.isin(test_index)]
        pair_subset_test = pd.merge(test_df, test_df, how='cross')
        datapoint_x += [pair_subset_test.Scaffold_x]
        datapoint_y += [pair_subset_test.Scaffold_y]
        del pair_subset_test

    datapoints = pd.DataFrame(data={'X':  np.concatenate(datapoint_x), 'Y':  np.concatenate(datapoint_y)})

    # Add the actual deltas and predicted deltas
    trues = predictions_CP['True']
    trues = [float(i) for i in trues]
    datapoints['True'] = trues

    DeltasRF = predictions_RF['Delta']
    DeltasRF = [float(i) for i in DeltasRF]
    datapoints['DeltaRF'] = DeltasRF

    DeltasCP = predictions_CP['Delta']
    DeltasCP = [float(i) for i in DeltasCP]
    datapoints['DeltaCP'] = DeltasCP

    DeltasDD = predictions_DD['Delta']
    DeltasDD = [float(i) for i in DeltasDD]
    datapoints['DeltaDD'] = DeltasDD

    # Grab the datapoints with matching scaffolds 
    matching = datapoints[datapoints['X'] == datapoints['Y']]
    
    # Grab the nonmatching datapoints
    nonmatching = datapoints[datapoints['X'] != datapoints['Y']]

    # Run Stats - Non-matching Scaffolds
    pearson_NM_RF = stats.pearsonr(nonmatching["True"], (nonmatching['DeltaRF']))
    MAE_NM_RF = metrics.mean_absolute_error(nonmatching["True"], (nonmatching['DeltaRF']))
    RMSE_NM_RF = math.sqrt(metrics.mean_squared_error(nonmatching["True"], (nonmatching['DeltaRF'])))

    pearson_NM_CP = stats.pearsonr(nonmatching["True"], (nonmatching['DeltaCP']))
    MAE_NM_CP = metrics.mean_absolute_error(nonmatching["True"], (nonmatching['DeltaCP']))
    RMSE_NM_CP = math.sqrt(metrics.mean_squared_error(nonmatching["True"], (nonmatching['DeltaCP'])))

    pearson_NM_DD = stats.pearsonr(nonmatching["True"], (nonmatching['DeltaDD']))
    MAE_NM_DD = metrics.mean_absolute_error(nonmatching["True"], (nonmatching['DeltaDD']))
    RMSE_NM_DD = math.sqrt(metrics.mean_squared_error(nonmatching["True"], (nonmatching['DeltaDD'])))

    scoringNM = pd.DataFrame({'Dataset': [dataset], 'Pearson\'s r RF': [round(pearson_NM_RF[0], 4)], 'MAE RF': [round(MAE_NM_RF, 4)], 'RMSE RF': [round(RMSE_NM_RF, 4)],
                            'Pearson\'s r CP': [round(pearson_NM_CP[0], 4)], 'MAE CP': [round(MAE_NM_CP, 4)], 'RMSE CP': [round(RMSE_NM_CP, 4)],
                            'Pearson\'s r DD': [round(pearson_NM_DD[0], 4)], 'MAE DD': [round(MAE_NM_DD, 4)], 'RMSE DD': [round(RMSE_NM_DD, 4)]})

    all_scoresNM = pd.concat([all_scoresNM, scoringNM])

    # Run Stats - Matching Scaffolds
    pearson_M_RF = stats.pearsonr(matching["True"], (matching['DeltaRF']))
    MAE_M_RF = metrics.mean_absolute_error(matching["True"], (matching['DeltaRF']))
    RMSE_M_RF = math.sqrt(metrics.mean_squared_error(matching["True"], (matching['DeltaRF'])))

    pearson_M_CP = stats.pearsonr(matching["True"], (matching['DeltaCP']))
    MAE_M_CP = metrics.mean_absolute_error(matching["True"], (matching['DeltaCP']))
    RMSE_M_CP = math.sqrt(metrics.mean_squared_error(matching["True"], (matching['DeltaCP'])))

    pearson_M_DD = stats.pearsonr(matching["True"], (matching['DeltaDD']))
    MAE_M_DD = metrics.mean_absolute_error(matching["True"], (matching['DeltaDD']))
    RMSE_M_DD = math.sqrt(metrics.mean_squared_error(matching["True"], (matching['DeltaDD'])))

    scoringM = pd.DataFrame({'Dataset': [dataset], 'Pearson\'s r RF': [round(pearson_M_RF[0], 4)], 'MAE RF': [round(MAE_M_RF, 4)], 'RMSE RF': [round(RMSE_M_RF, 4)],
                            'Pearson\'s r CP': [round(pearson_M_CP[0], 4)], 'MAE CP': [round(MAE_M_CP, 4)], 'RMSE CP': [round(RMSE_M_CP, 4)],
                            'Pearson\'s r DD': [round(pearson_M_DD[0], 4)], 'MAE DD': [round(MAE_M_DD, 4)], 'RMSE DD': [round(RMSE_M_DD, 4)]})

    all_scoresM = pd.concat([all_scoresM, scoringM])

all_scoresNM.to_csv("DeepDelta_CV_Scaffold_NonMatching_Updated_99.csv", index = False) # Save results for unmatched scaffold pairs
all_scoresM.to_csv("DeepDelta_CV_Scaffold_Matching_Updated_99.csv", index = False) # Save results for matched scaffold pairs

#########################




##########################
### External Test Sets ###
##########################
 

all_scoresNM = pd.DataFrame(columns=['Dataset', 'Pearson\'s r RF', 'MAE RF', 'RMSE RF',
                                    'Pearson\'s r CP', 'MAE CP', 'RMSE CP', 'Pearson\'s r DD', 'MAE DD', 'RMSE DD']) # For pairs of Nonmatching Scaffolds
all_scoresM = pd.DataFrame(columns=['Dataset', 'Pearson\'s r RF', 'MAE RF', 'RMSE RF',
                                    'Pearson\'s r CP', 'MAE CP', 'RMSE CP', 'Pearson\'s r DD', 'MAE DD', 'RMSE DD']) # For pairs with Matching Scaffolds

# Evaluate Matching and Non-matching Scaffold Pairs for all Datasets
for dataset in datasets:
    dataframe = pd.read_csv('../Datasets/Updated_99/Test/{}_test.csv'.format(dataset)) # External Test Sets

    # 3 Models to Evaluate
    # 1 - Random Forest 
    predictions_RF = pd.read_csv('../Results/Updated_99/External_Test_Results/RandomForest_Ext_Test/{}_RandomForest_Test.csv'.format(dataset)).T
    predictions_RF.columns =['True', 'Delta']
    # 2 - ChemProp
    predictions_CP = pd.read_csv('../Results/Updated_99/External_Test_Results/ChemProp50_Ext_Test/{}_ChemProp50_Test.csv'.format(dataset)).T
    predictions_CP.columns =['True', 'Delta']
    # 3 - DeepDelta
    predictions_DD = pd.read_csv('../Results/Updated_99/External_Test_Results/DeepDelta5_Ext_Test/{}_DeepDelta5_Test.csv'.format(dataset)).T
    predictions_DD.columns =['True', 'Delta']

    # Prepare Scaffolds
    mols = [Chem.MolFromSmiles(s) for s in dataframe.SMILES]
    scaffolds = [Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m)) for m in mols]
    data = pd.DataFrame(data={'Scaffold':  scaffolds})
    del dataframe

    # Cross-Merge the External Test Sets and Extract the scaffolds
    datapoint_x = []
    datapoint_y = []

    pair_subset_test = pd.merge(data, data, how='cross')
    datapoint_x += [pair_subset_test.Scaffold_x]
    datapoint_y += [pair_subset_test.Scaffold_y]
    del pair_subset_test

    datapoints = pd.DataFrame(data={'X':  np.concatenate(datapoint_x), 'Y':  np.concatenate(datapoint_y)})

    # Add the actual deltas and predicted deltas
    trues = predictions_CP['True']
    trues = [float(i) for i in trues]
    datapoints['True'] = trues

    DeltasRF = predictions_RF['Delta']
    DeltasRF = [float(i) for i in DeltasRF]
    datapoints['DeltaRF'] = DeltasRF

    DeltasCP = predictions_CP['Delta']
    DeltasCP = [float(i) for i in DeltasCP]
    datapoints['DeltaCP'] = DeltasCP

    DeltasDD = predictions_DD['Delta']
    DeltasDD = [float(i) for i in DeltasDD]
    datapoints['DeltaDD'] = DeltasDD

    # Grab the datapoints with matching scaffolds 
    matching = datapoints[datapoints['X'] == datapoints['Y']]
    
    # Grab the nonmatching datapoints
    nonmatching = datapoints[datapoints['X'] != datapoints['Y']]

    # Run Stats - Non-matching Scaffolds
    pearson_NM_RF = stats.pearsonr(nonmatching["True"], (nonmatching['DeltaRF']))
    MAE_NM_RF = metrics.mean_absolute_error(nonmatching["True"], (nonmatching['DeltaRF']))
    RMSE_NM_RF = math.sqrt(metrics.mean_squared_error(nonmatching["True"], (nonmatching['DeltaRF'])))

    pearson_NM_CP = stats.pearsonr(nonmatching["True"], (nonmatching['DeltaCP']))
    MAE_NM_CP = metrics.mean_absolute_error(nonmatching["True"], (nonmatching['DeltaCP']))
    RMSE_NM_CP = math.sqrt(metrics.mean_squared_error(nonmatching["True"], (nonmatching['DeltaCP'])))

    pearson_NM_DD = stats.pearsonr(nonmatching["True"], (nonmatching['DeltaDD']))
    MAE_NM_DD = metrics.mean_absolute_error(nonmatching["True"], (nonmatching['DeltaDD']))
    RMSE_NM_DD = math.sqrt(metrics.mean_squared_error(nonmatching["True"], (nonmatching['DeltaDD'])))

    scoringNM = pd.DataFrame({'Dataset': [dataset], 'Pearson\'s r RF': [round(pearson_NM_RF[0], 4)], 'MAE RF': [round(MAE_NM_RF, 4)], 'RMSE RF': [round(RMSE_NM_RF, 4)],
                            'Pearson\'s r CP': [round(pearson_NM_CP[0], 4)], 'MAE CP': [round(MAE_NM_CP, 4)], 'RMSE CP': [round(RMSE_NM_CP, 4)],
                            'Pearson\'s r DD': [round(pearson_NM_DD[0], 4)], 'MAE DD': [round(MAE_NM_DD, 4)], 'RMSE DD': [round(RMSE_NM_DD, 4)]})

    all_scoresNM = pd.concat([all_scoresNM, scoringNM])

    # Run Stats - Matching Scaffolds
    pearson_M_RF = stats.pearsonr(matching["True"], (matching['DeltaRF']))
    MAE_M_RF = metrics.mean_absolute_error(matching["True"], (matching['DeltaRF']))
    RMSE_M_RF = math.sqrt(metrics.mean_squared_error(matching["True"], (matching['DeltaRF'])))

    pearson_M_CP = stats.pearsonr(matching["True"], (matching['DeltaCP']))
    MAE_M_CP = metrics.mean_absolute_error(matching["True"], (matching['DeltaCP']))
    RMSE_M_CP = math.sqrt(metrics.mean_squared_error(matching["True"], (matching['DeltaCP'])))

    pearson_M_DD = stats.pearsonr(matching["True"], (matching['DeltaDD']))
    MAE_M_DD = metrics.mean_absolute_error(matching["True"], (matching['DeltaDD']))
    RMSE_M_DD = math.sqrt(metrics.mean_squared_error(matching["True"], (matching['DeltaDD'])))

    scoringM = pd.DataFrame({'Dataset': [dataset], 'Pearson\'s r RF': [round(pearson_M_RF[0], 4)], 'MAE RF': [round(MAE_M_RF, 4)], 'RMSE RF': [round(RMSE_M_RF, 4)],
                            'Pearson\'s r CP': [round(pearson_M_CP[0], 4)], 'MAE CP': [round(MAE_M_CP, 4)], 'RMSE CP': [round(RMSE_M_CP, 4)],
                            'Pearson\'s r DD': [round(pearson_M_DD[0], 4)], 'MAE DD': [round(MAE_M_DD, 4)], 'RMSE DD': [round(RMSE_M_DD, 4)]})

    all_scoresM = pd.concat([all_scoresM, scoringM])

all_scoresNM.to_csv("DeepDelta_Ext_Scaffold_NonMatching_Updated_99.csv", index = False) # Save results for unmatched scaffold pairs
all_scoresM.to_csv("DeepDelta_Ext_Scaffold_Matching_Updated_99.csv", index = False) # Save results for matched scaffold pairs

#########################