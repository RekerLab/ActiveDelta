# Imports
import os
import abc
import math
import shutil
import tempfile
import numpy as np
import pandas as pd
import chemprop
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn import metrics
from scipy import stats as stats
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor as RF


# Define abstract class to define interface of models
class abstractDeltaModel(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def fit(self, x, y):
        pass

    @abc.abstractmethod
    def predict(self, x):
        pass    




class DeepDelta(abstractDeltaModel): # Used for ActiveDelta implementation of Chemprop active learning
    epochs = None
    dirpath = None 

    def __init__(self, epochs=5, dirpath = None):
        self.epochs = epochs
        self.dirpath = dirpath


    def fit(self, x, y, metric='r2'):
        
        self.dirpath = tempfile.NamedTemporaryFile().name # use temporary file to store model
        
        # create pairs of training data
        train = pd.merge(x, x, how='cross') 
        y_values = pd.merge(y, y, how='cross')
        train["Y"] = y_values.Y_y - y_values.Y_x
        del y_values 

        temp_datafile = tempfile.NamedTemporaryFile() 
        train.to_csv(temp_datafile.name, index=False)
        
        # store default arguments for model
        arguments = [ 
            '--data_path', temp_datafile.name,
            '--separate_val_path', temp_datafile.name, 
            '--dataset_type', 'regression', 
            '--save_dir', self.dirpath,
            '--num_folds', '1',
            '--split_sizes', '1.0', '0', '0',
            '--ensemble_size', '1', 
            '--epochs', str(self.epochs),
            '--metric', metric,
            '--number_of_molecules', '2',
            '--aggregation', 'sum'
        ]
        
        args = chemprop.args.TrainArgs().parse_args(arguments)
        chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training) # Train

        temp_datafile.close()


    def predict(self, x): 

        dataset = pd.merge(x, x, how='cross') # Make pairs by cross-merging

        temp_datafile = tempfile.NamedTemporaryFile()
        dataset.to_csv(temp_datafile.name, index=False)
        temp_predfile = tempfile.NamedTemporaryFile()

        arguments = [
            '--test_path', temp_datafile.name,
            '--preds_path', temp_predfile.name, 
            '--checkpoint_dir', self.dirpath,
            '--number_of_molecules', '2'
        ]

        args = chemprop.args.PredictArgs().parse_args(arguments)
        chemprop.train.make_predictions(args=args) # Predict

        predictions = pd.read_csv(temp_predfile.name)['Y']

        temp_datafile.close()
        temp_predfile.close()

        return predictions


    def predict2(self, x1, x2): # Adjusted prediction using two datasets

        dataset = pd.merge(x1, x2, how='cross') # merge two datasets together

        temp_datafile = tempfile.NamedTemporaryFile()
        dataset.to_csv(temp_datafile.name, index=False)
        temp_predfile = tempfile.NamedTemporaryFile()

        arguments = [
            '--test_path', temp_datafile.name,
            '--preds_path', temp_predfile.name,
            '--checkpoint_dir', self.dirpath,
            '--number_of_molecules', '2'
        ]

        args = chemprop.args.PredictArgs().parse_args(arguments)
        chemprop.train.make_predictions(args=args) # Predict

        predictions = pd.read_csv(temp_predfile.name)['Y']

        temp_datafile.close()
        temp_predfile.close()

        return predictions

    
    def __str__(self):
        return "DeepDelta" + str(self.epochs)




class Trad_ChemProp(abstractDeltaModel): # Used for standard implementation of Chemprop active learning
    epochs = None
    dirpath = None  
    dirpath_single = None

    def __init__(self, epochs=50, dirpath = None, dirpath_single = None):
        self.epochs = epochs
        self.dirpath = dirpath
        self.dirpath_single = dirpath_single

    def fit(self, x, y, metric='r2'):
        self.dirpath_single = tempfile.NamedTemporaryFile().name # use temporary file to store model
        
        train = pd.DataFrame(np.transpose(np.vstack([x,y])),columns=["X","Y"])

        temp_datafile = tempfile.NamedTemporaryFile()
        train.to_csv(temp_datafile.name, index=False)

        # store default arguments for model
        arguments = [
            '--data_path', temp_datafile.name,
            '--separate_val_path', temp_datafile.name,
            '--dataset_type', 'regression',
            '--save_dir', self.dirpath_single,
            '--num_folds', '1',
            '--split_sizes', '1.0', '0', '0',
            '--ensemble_size', '1',
            '--epochs', str(self.epochs),
            '--number_of_molecules', '1',
            '--metric', metric, 
            '--aggregation', 'sum'
        ]

        args = chemprop.args.TrainArgs().parse_args(arguments)
        chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training) # Train

        temp_datafile.close()


    def predict(self, x):  # Used for predicting molecular differences

        dataset = pd.DataFrame(x)
        temp_datafile = tempfile.NamedTemporaryFile()
        dataset.to_csv(temp_datafile.name, index=False)
        temp_predfile = tempfile.NamedTemporaryFile()

        arguments = [
            '--test_path', temp_datafile.name,
            '--preds_path', temp_predfile.name, 
            '--checkpoint_dir', self.dirpath_single,
            '--number_of_molecules', '1'
        ]

        args = chemprop.args.PredictArgs().parse_args(arguments)
        chemprop.train.make_predictions(args=args) # Make prediction

        predictions = pd.read_csv(temp_predfile.name)['Y'] 

        preds = pd.merge(predictions,predictions,how='cross') # Cross merge to make pairs

        temp_datafile.close()
        temp_predfile.close()

        return preds.Y_y - preds.Y_x   # Calculate and return the delta values


    def predict_single(self, x):  # Used for predicting for only one molecule

        dataset = pd.DataFrame(x)
        temp_datafile = tempfile.NamedTemporaryFile()
        dataset.to_csv(temp_datafile.name, index=False)
        temp_predfile = tempfile.NamedTemporaryFile()

        arguments = [
            '--test_path', temp_datafile.name,
            '--preds_path', temp_predfile.name, 
            '--checkpoint_dir', self.dirpath_single,
            '--number_of_molecules', '1'
        ]

        args = chemprop.args.PredictArgs().parse_args(arguments)
        chemprop.train.make_predictions(args=args) # Make prediction

        predictions = pd.read_csv(temp_predfile.name)['Y'] 

        temp_datafile.close()
        temp_predfile.close()

        return predictions  
    
    def __str__(self):
        return "ChemProp" + str(self.epochs)




class Trad_RF(abstractDeltaModel): # Used for standard implementation of random forest active learning
    model = None

    def __init__(self):
        self.model = RF()

    def fit(self, x, y, metric='r2'):
        mols = [Chem.MolFromSmiles(s) for s in x]
        fps = [np.array(AllChem.GetMorganFingerprintAsBitVect(m,2)) for m in mols]
        self.model.fit(fps,y) # Fit using traditional methods

    def predict(self, x): # Used for predicting molecular differences
        mols = [Chem.MolFromSmiles(s) for s in x]
        fps = [np.array(AllChem.GetMorganFingerprintAsBitVect(m,2)) for m in mols]
        
        predictions = pd.DataFrame(self.model.predict(fps)) # Predict using traditional methods
        results = pd.merge(predictions,predictions,how='cross') # Cross merge into pairs after predictions
        return results['0_y'] - results['0_x']  # Calculate and return the delta values

    def predict_single(self, x): # Used for predicting for only one molecule
        mols = [Chem.MolFromSmiles(s) for s in x]
        fps = [np.array(AllChem.GetMorganFingerprintAsBitVect(m,2)) for m in mols]
        predictions = pd.Series(self.model.predict(fps))
        return predictions
    
    def __str__(self):
        return "RandomForest"



class Trad_XGB(abstractDeltaModel): # Used for standard implementation of XGBoost active learning
    model = None

    def __init__(self):
        self.model = XGBRegressor(tree_method='hist', device="cuda")

    def fit(self, x, y, metric='r2'):
        mols = [Chem.MolFromSmiles(s) for s in x]
        fps = [np.array(AllChem.GetMorganFingerprintAsBitVect(m,2)) for m in mols]
        self.model.fit(fps,y) # Fit using traditional methods

    def predict(self, x): # Used for predicting molecular differences
        mols = [Chem.MolFromSmiles(s) for s in x]
        fps = [np.array(AllChem.GetMorganFingerprintAsBitVect(m,2)) for m in mols]
        
        predictions = pd.DataFrame(self.model.predict(fps)) # Predict using traditional methods
        results = pd.merge(predictions,predictions,how='cross') # Cross merge into pairs after predictions
        return results['0_y'] - results['0_x']  # Calculate and return the delta values

    def predict_single(self, x): # Used for predicting for only one molecule
        mols = [Chem.MolFromSmiles(s) for s in x]
        fps = [np.array(AllChem.GetMorganFingerprintAsBitVect(m,2)) for m in mols]
        predictions = pd.Series(self.model.predict(fps))
        return predictions
    
    def __str__(self):
        return "XGBoost"



class Delta_XGB(abstractDeltaModel): # Used for ActiveDelta implementation of XGBoost active learning
    model = None

    def __init__(self):
        self.model = XGBRegressor(tree_method='hist', device="cuda")

    def fit(self, x, y, metric='r2'):
        
        # create pairs of training data
        train = pd.merge(x, x, how='cross') 
        y_values = pd.merge(y, y, how='cross')
        train["Y"] = y_values.Y_y - y_values.Y_x
        del y_values 

        # Convert SMILES to fingerprints
        mols_x = [Chem.MolFromSmiles(s_x) for s_x in train[train.columns[0]]]
        fps_x = [np.array(AllChem.GetMorganFingerprintAsBitVect(m_x, 4, nBits=2048)) for m_x in mols_x]
        mols_y = [Chem.MolFromSmiles(s_y) for s_y in train[train.columns[1]]]
        fps_y = [np.array(AllChem.GetMorganFingerprintAsBitVect(m_y, 4, nBits=2048)) for m_y in mols_y]

        # Merge Fingerprints
        pair_data = pd.DataFrame(data={'Fingerprint_x': list(np.array(fps_x)), 'Fingerprint_y': list(np.array(fps_y))})
        train['fps'] =  pair_data.Fingerprint_x.combine(pair_data.Fingerprint_y, np.append)

        self.model.fit(np.vstack(train.fps.to_numpy()), train.Y) # Fit 



    def predict(self, x): # Used for predicting molecular differences

        # create pairs of testing data
        test = pd.merge(x, x, how='cross') 

        # Convert SMILES to fingerprints
        mols_x = [Chem.MolFromSmiles(s_x) for s_x in test[test.columns[0]]]
        fps_x = [np.array(AllChem.GetMorganFingerprintAsBitVect(m_x, 4, nBits=2048)) for m_x in mols_x]
        mols_y = [Chem.MolFromSmiles(s_x) for s_x in test[test.columns[1]]]
        fps_y = [np.array(AllChem.GetMorganFingerprintAsBitVect(m_y, 4, nBits=2048)) for m_y in mols_y]

        # Merge Fingerprints
        pair_data = pd.DataFrame(data={'Fingerprint_x': list(np.array(fps_x)), 'Fingerprint_y': list(np.array(fps_y))})
        pair_data['fps'] =  pair_data.Fingerprint_x.combine(pair_data.Fingerprint_y, np.append)

        predictions = pd.DataFrame(self.model.predict(np.vstack(pair_data.fps.to_numpy()))) # Predict
        return predictions

    def predict2(self, x1, x2): # Used for predicting for only one molecule
        # create pairs of testing data
        test = pd.merge(x1, x2, how='cross') 

        # Convert SMILES to fingerprints
        mols_x = [Chem.MolFromSmiles(s_x) for s_x in test[test.columns[0]]]
        fps_x = [np.array(AllChem.GetMorganFingerprintAsBitVect(m_x, 4, nBits=2048)) for m_x in mols_x]
        mols_y = [Chem.MolFromSmiles(s_x) for s_x in test[test.columns[1]]]
        fps_y = [np.array(AllChem.GetMorganFingerprintAsBitVect(m_y, 4, nBits=2048)) for m_y in mols_y]

        # Merge Fingerprints
        pair_data = pd.DataFrame(data={'Fingerprint_x': list(np.array(fps_x)), 'Fingerprint_y': list(np.array(fps_y))})
        pair_data['fps'] =  pair_data.Fingerprint_x.combine(pair_data.Fingerprint_y, np.append)

        predictions = pd.DataFrame(self.model.predict(np.vstack(pair_data.fps.to_numpy()))) # Predict

        return predictions
    
    def __str__(self):
        return "Delta_XGBoost"

