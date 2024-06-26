# Imports
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import math
import statistics
from scipy import stats as stats

datasets = ['CHEMBL1075104-1',
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
 
 
 
###############################################
##      Nearest Neighbor Similarity of       ##
## External Lead Predicted to be Most Potent ##
##        Compared to Training Data          ##
###############################################

models = ['DeepDelta5', 'ChemProp50', 'RandomForest', 'Delta_XGBoost', 'XGBoost']
model_short_names = ['ADCP', 'CP', 'RF', 'ADXGB', 'XGB']
groups = ['R1', 'R2', 'R3']
fingerprints = ['Morgan', 'MACCS', 'AtomPair']

for fingerprint in fingerprints:

    for group in groups:

      final_results = pd.DataFrame({'Dataset': datasets})
      final_results["DeepDelta5"] = np.nan
      final_results["ChemProp50"] = np.nan
      final_results["RandomForest"] = np.nan
      final_results["Delta_XGBoost"] = np.nan
      final_results["XGBoost"] = np.nan
         
      for i in range(len(models)):

        similarities = pd.DataFrame({'Dataset': datasets})
        similarities["Value"] = np.nan
        
        dataset_number = 0

        for dataset in datasets:
          # Prepare dataframes
          train_df = pd.read_csv('../Datasets/Train/{}_train.csv'.format(dataset))
          test_set = pd.read_csv("../Datasets/Test/{}_test.csv".format(dataset))
          try:
            potent_molecule_df = pd.read_csv('../Results/External_Test_Results/AL100_ExternalTest_{}_{}/{}_{}_AL100_{}_Test_Single_Predictions.csv'.format(model_short_names[i], group, dataset, model[i], group)).T
          except:
            potent_molecule_df = pd.read_csv('../Results/External_Test_Results/AL100_ExternalTest_{}_{}/{}_{}_AL100_{}_Test_Single_PredictionsCorrect.csv'.format(model_short_names[i], group, dataset, model[i], group)).T
          potent_molecule_df.columns = ['True', 'Pred']
          Preds = potent_molecule_df['Pred']
          Preds = [float(i) for i in Preds]
          test_set['Pred'] = Preds

          # Get most potent prediction
          potent_molecule = pd.DataFrame(test_set.nlargest(1, 'Pred')['SMILES'])
          potent_molecule = potent_molecule.reset_index()
          potent_molecule_mol = Chem.MolFromSmiles(potent_molecule['SMILES'][0])
          
          # Get fingerprints of this molecule
          if fingerprint == 'Morgan':
            potent_molecule_fp = AllChem.GetMorganFingerprintAsBitVect(potent_molecule_mol)
          elif fingerprint == 'MACCS': 
            potent_molecule_fp = AllChem.GetMACCSKeysFingerprint(potent_molecule_mol)
          elif fingerprint == 'AtomPair': 
            potent_molecule_fp = AllChem.GetAtomPairFingerprint(potent_molecule_mol)
          
          # Get fingerprints of training data
          mols = [Chem.MolFromSmiles(s) for s in train_df.SMILES]
          if fingerprint == 'Morgan':
            fps_list = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 1024) for m in mols]
          elif fingerprint == 'MACCS':
            fps_list = [AllChem.GetMACCSKeysFingerprint(m) for m in mols]
          elif fingerprint == 'AtomPair': 
            fps_list = [AllChem.GetAtomPairFingerprint(m) for m in mols]
          
          # Get maximum similarity to the external molecule predicted to be most potent from any training datapoint
          NN_similarity_ = max(DataStructs.BulkTanimotoSimilarity(potent_molecule, fps_list))
          similarities['Value'][dataset_number] = NN_similarity

        final_results[model[i]] = similarities['Value']     

      final_results.to_csv('TanimotoSimilaritiesToTrainingSet_{}_{}.csv'.format(fingerprint, group), index = False)

 
 
##########################################
##  Average Tanimoto Similarity of the  ##
##    Compounds Selected During the     ##
##   First Stages of Active Learning    ##
##########################################

iterations = ['15', '30', '45'] # Representing iterations 1-15, 16-30, and 31-45, respectively 

for fingerprint in fingerprints:

    for iteration in iterations:

        for group in groups:

          final_results = pd.DataFrame({'Dataset': datasets})
          final_results["DeepDelta5"] = np.nan
          final_results["ChemProp50"] = np.nan
          final_results["RandomForest"] = np.nan
          final_results["Delta_XGBoost"] = np.nan
          final_results["XGBoost"] = np.nan
          
          for i in range(len(models)):

            similarities = pd.DataFrame({'Dataset': datasets})
            similarities["Value"] = np.nan
            
            dataset_number = 0

            for dataset in datasets:
              # Prepare dataframes
              df = pd.read_csv('../Results/Exploitative_Active_Learning_Results/AL_Exploitative_{}_{}/{}_train_round_{}_200_{}.csv'.format(model_short_names[i], group, dataset, models[i], group))
              if iteration == '15':
                df = df[df['Iteration'].between(1, 15)]
              elif iteration == '30':
                df = df[df['Iteration'].between(16, 30)]
              else:
                df = df[df['Iteration'].between(31, 45)]      
              
              # Get fingerprints of data
              mols = [Chem.MolFromSmiles(s) for s in train_df.SMILES]
              if fingerprint == 'Morgan':
                fps_list = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 1024) for m in mols]
              elif fingerprint == 'MACCS': 
                fps_list = [AllChem.GetMACCSKeysFingerprint(m) for m in mols]
              elif fingerprint == 'AtomPair':
                fps_list = [AllChem.GetAtomPairFingerprint(m) for m in mols]
                
              # Get average similarity within each iteration range
              similarity_list = sum([DataStructs.BulkTanimotoSimilarity(fps_list[i], fps_list[i+1:]) for i in range(len(fps_list) - 1)], [])
              similarities['Value'][dataset_number] = statistics.mean(similarity_list)

            final_results[model[i]] = similarities['Value']     

          final_results.to_csv('AverageSimilarities_{}_{}_{}.csv'.format(fingerprint, group, iteration), index = False)
