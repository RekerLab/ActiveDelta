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
        dataset = '../Datasets/Train/{}_train.csv'.format(prop) # Training dataset
        pred_dataset = '../Datasets/Test/{}_test.csv'.format(prop) # For prediction
        
        # Fit model on entire training dataset
        df = pd.read_csv(dataset)
        x = df[df.columns[0]]
        y = df[df.columns[1]]
        model.fit(x,y) 
        
        # Predict on cross-merged external test set
        df = pd.read_csv(pred_dataset)
        x = df[df.columns[0]]
        y = df[df.columns[1]]
        y_pairs = pd.merge(y, y, how='cross') # Cross-merge into pairs
        vals = y_pairs.Y_y - y_pairs.Y_x # Calculate delta values
        preds = model.predict(x) # Make predictions
        
        results = [vals, preds] # Save the true delta values and predictions
        pd.DataFrame(results).to_csv("{}_{}_Test.csv".format(prop, model), index=False) # Save results
        #If you .T the dataframe, then the first column is ground truth, the second is predictions

