# Imports
import os
import abc
import math
import shutil
import tempfile
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn import metrics
from scipy import stats as stats
from sklearn.model_selection import KFold
import chemprop
from sklearn.ensemble import RandomForestRegressor as RF
from models import *


############################
### Original 56 Datasets ###
############################

properties = ['CHEMBL1267245',
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
 
models = [DeepDelta(), Trad_ChemProp(), Trad_RF()]

for model in models:
    for prop in properties:
        dataset = '../Datasets/Original_56/Train/{}_train.csv'.format(prop) # Training dataset
        pred_dataset = '../Datasets/Original_56/Test/{}_test.csv'.format(prop) # For prediction
        
        # Fit model on entire training dataset
        train_df = pd.read_csv(dataset)
        x_train = train_df[train_df.columns[0]]
        y_train = train_df[train_df.columns[1]]
        model.fit(x_train,y_train) 
        
        # Predict on external test set
        test_df = pd.read_csv(pred_dataset)
        x_test = test_df[test_df.columns[0]]
        y_test = test_df[test_df.columns[1]]
        
        if str(model) != 'DeepDelta5': # Traditional learning
            preds = model.predict_single(x_test) # Make predictions
            results = [y_test, preds] # Save the true values and predictions
            pd.DataFrame(results).to_csv("{}_{}_Test_Single_Predictions.csv".format(prop, model), index=False) # Save results
            #If you .T the dataframe, then the first column is ground truth, the second is predictions
          
        else: # Delta Predictions 
            y_pairs = pd.merge(y_train, y_test, how='cross') # Cross-merge train and test sets into pairs
            vals = y_pairs.Y_y - y_pairs.Y_x # Calculate delta values
            preds = model.predict2(x_train, x_test) # Make predictions from the cross-merge of the two datasets
            intermediate_results = [vals, preds] # Save the true delta values and predictions
            pd.DataFrame(intermediate_results).to_csv("{}_{}_Test_Cross-Merged_Predictions.csv".format(prop, model), index=False) # Save intermediate results
            
            pair_subset_test = pd.merge(train_df, test_df, how='cross') # Get the SMILES for both molecules during the cross-merge
            pair_subset_test = pair_subset_test.drop(columns=['Y_x', 'Y_y'])
            preds = pd.read_csv("{}_{}_Test_Cross-Merged_Predictions.csv".format(prop, model)).T
            preds.columns =['True', 'Delta']

            trues = preds['True'].tolist()
            trues = [float(i) for i in trues]
            pair_subset_test['True'] = trues # Add the true delta values

            Deltas = preds['Delta']
            Deltas = [float(i) for i in Deltas]
            pair_subset_test['Pred'] = Deltas # Add the predicted delta values

            # Get average values for each molecule
            molecule_values = pd.DataFrame(columns=['SMILES', 'True', 'Pred'])

            for SMILES in pair_subset_test['SMILES_y'].unique(): # For each unique molecule in the test set
              working_df = pair_subset_test.loc[pair_subset_test['SMILES_y'] == SMILES] # Get all predictions involving this molecule
              inter_df = pd.DataFrame({'SMILES': [SMILES], 'True': [working_df['True'].mean()],
                                       'Pred': [working_df['Pred'].mean()]}) # Get the Mean value 
              molecule_values = pd.concat([molecule_values, inter_df]) 
            molecule_values = molecule_values.drop(['SMILES'], axis = 1)
            molecule_values = molecule_values.reset_index(drop=True)
            molecule_values = molecule_values.T
            molecule_values.to_csv("{}_{}_Test_Single_Predictions.csv".format(prop, model), index=False) # Save results
            
 
 
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


models = [DeepDelta(), Trad_ChemProp(), Trad_RF()]

for model in models:
    for prop in properties:
        dataset = '../Datasets/Updated_99/Train/{}_train.csv'.format(prop) # Training dataset
        pred_dataset = '../Datasets/Updated_99/Test/{}_test.csv'.format(prop) # For prediction
        
        # Fit model on entire training dataset
        train_df = pd.read_csv(dataset)
        x_train = train_df[train_df.columns[0]]
        y_train = train_df[train_df.columns[1]]
        model.fit(x_train,y_train) 
        
        # Predict on external test set
        test_df = pd.read_csv(pred_dataset)
        x_test = test_df[test_df.columns[0]]
        y_test = test_df[test_df.columns[1]]
        
        if str(model) != 'DeepDelta5': # Traditional learning
            preds = model.predict_single(x_test) # Make predictions
            results = [y_test, preds] # Save the true values and predictions
            pd.DataFrame(results).to_csv("{}_{}_Test_Single_Predictions.csv".format(prop, model), index=False) # Save results
            #If you .T the dataframe, then the first column is ground truth, the second is predictions
          
        else: # Delta Predictions 
            y_pairs = pd.merge(y_train, y_test, how='cross') # Cross-merge train and test sets into pairs
            vals = y_pairs.Y_y - y_pairs.Y_x # Calculate delta values
            preds = model.predict2(x_train, x_test) # Make predictions from the cross-merge of the two datasets
            intermediate_results = [vals, preds] # Save the true delta values and predictions
            pd.DataFrame(intermediate_results).to_csv("{}_{}_Test_Cross-Merged_Predictions.csv".format(prop, model), index=False) # Save intermediate results
            
            pair_subset_test = pd.merge(train_df, test_df, how='cross') # Get the SMILES for both molecules during the cross-merge
            pair_subset_test = pair_subset_test.drop(columns=['Y_x', 'Y_y'])
            preds = pd.read_csv("{}_{}_Test_Cross-Merged_Predictions.csv".format(prop, model)).T
            preds.columns =['True', 'Delta']

            trues = preds['True'].tolist()
            trues = [float(i) for i in trues]
            pair_subset_test['True'] = trues # Add the true delta values

            Deltas = preds['Delta']
            Deltas = [float(i) for i in Deltas]
            pair_subset_test['Pred'] = Deltas # Add the predicted delta values

            # Get average values for each molecule
            molecule_values = pd.DataFrame(columns=['SMILES', 'True', 'Pred'])

            for SMILES in pair_subset_test['SMILES_y'].unique(): # For each unique molecule in the test set
              working_df = pair_subset_test.loc[pair_subset_test['SMILES_y'] == SMILES] # Get all predictions involving this molecule
              inter_df = pd.DataFrame({'SMILES': [SMILES], 'True': [working_df['True'].mean()],
                                       'Pred': [working_df['Pred'].mean()]}) # Get the Mean value 
              molecule_values = pd.concat([molecule_values, inter_df]) 
            molecule_values = molecule_values.drop(['SMILES'], axis = 1)
            molecule_values = molecule_values.reset_index(drop=True)
            molecule_values = molecule_values.T
            molecule_values.to_csv("{}_{}_Test_Single_Predictions.csv".format(prop, model), index=False) # Save results
