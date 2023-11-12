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

def cross_validation(x, y, prop, model, k=10, seed=1): # provide option to cross validate with x and y instead of file
  kf = KFold(n_splits=k, random_state=seed, shuffle=True)
  cnt = 1 # Used to keep track of current fold
  preds = []
  vals  = []

  for train, test in kf.split(x):
        model.fit(x[train],y[train]) # Fit on training data
        preds = np.append(preds, model.predict(x[test])) # Predict on testing data
        y_pairs = pd.merge(y[test],y[test],how='cross') # Cross-merge data values
        vals = np.append(vals, y_pairs.Y_y - y_pairs.Y_x) # Calculate true delta values
  return [vals,preds] # Return true delta values and predicted delta values


def cross_validation_file(data_path, prop, model, k=10, seed=1): # Cross-validate from a file
  df = pd.read_csv(data_path)
  x = df[df.columns[0]]
  y = df[df.columns[1]]
  return cross_validation(x,y,prop,model,k,seed)



###################
####  5x10 CV  ####
###################

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
        delta = pd.DataFrame(columns=['Pearson\'s r', 'MAE', 'RMSE']) # For storing results
        for i in range(5): # Allow for 5x10-fold cross validation
            dataset = '../Datasets/Train/{}_train.csv'.format(prop) # Training dataset
            results = cross_validation_file(data_path=dataset, prop = prop, model=model, k=10, seed = i) # Run cross-validation

            pd.DataFrame(results).to_csv("{}_{}_{}.csv".format(prop, str(model), i), index=False) # Save results
            # If you .T the dataframe, then the first column is ground truth, the second is predictions

            # Read saved dataframe to calculate statistics
            df = pd.read_csv("{}_{}_{}.csv".format(prop, model, i)).T
            df.columns =['True', 'Delta']
            trues = df['True'].tolist()
            preds = df['Delta'].tolist() 
            
            # Calculate statistics for each round
            pearson = stats.pearsonr(trues, preds)
            MAE = metrics.mean_absolute_error(trues, preds)
            RMSE = math.sqrt(metrics.mean_squared_error(trues, preds))
            scoring = pd.DataFrame({'Pearson\'s r': [round(pearson[0], 4)], 'MAE': [round(MAE, 4)], 'RMSE': [round(RMSE, 4)]})
            delta = pd.concat([delta, scoring])

        # Calculate overall statistics 
        average = pd.DataFrame({'Pearson\'s r': [round(np.mean(delta['Pearson\'s r']), 3)], 'MAE': [round(np.mean(delta['MAE']), 3)], 'RMSE': [round(np.mean(delta['RMSE']), 3)]})
        std = pd.DataFrame({'Pearson\'s r': [round(np.std(delta['Pearson\'s r']), 3)], 'MAE': [round(np.std(delta['MAE']), 3)], 'RMSE': [round(np.std(delta['RMSE']), 3)]})
        delta = pd.concat([delta, average])
        delta = pd.concat([delta, std])
        delta = delta.set_index([pd.Index([1, 2, 3, 4, 5, 'Avg', 'Std. Dev.'])])
        delta.to_csv("{}_{}_delta_scoring.csv".format(prop, model)) # Save data
        


