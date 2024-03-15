# Imports
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
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
 
          
models = ['DeepDelta5', 'ChemProp50', 'RandomForest', 'Delta_XGBoost', 'XGBoost', 'Random_Selection']
model_short_names = ['ADCP', 'CP', 'RF', 'ADXGB', 'XGB', 'Random']

groups = ['R1', 'R2', 'R3']

for group in groups: 

    for i in range(len(models)):
      unique_scaffold_dataframe = pd.DataFrame(columns = datasets)
      unique_scaffold_hits_dataframe = pd.DataFrame(columns = datasets)

      for dataset in datasets:
        df = pd.read_csv('../Results/Exploitative_Active_Learning_Results/AL_Exploitative_{}_R1/{}_train_round_{}_200_R1.csv'.format(model_short_names[i], dataset, models[i]))

        # Prepare Scaffolds
        mols = [Chem.MolFromSmiles(s) for s in df.SMILES]
        scaffolds = [Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m)) for m in mols]   
        
        #########################
        ### Scaffold Analysis ###
        #########################
        
        # Get Unique Scaffold Count at Each Iteration
        cnt = 0
        unique_scaffolds = []
        unique_scaffold_counter = []

        for j in range(len(scaffolds)):
          if scaffolds[j] in unique_scaffolds:
            unique_scaffold_counter.append(cnt)
          else:
            cnt += 1
            unique_scaffolds.append(scaffolds[j])
            unique_scaffold_counter.append(cnt)

        unique_scaffold_dataframe[dataset] = unique_scaffold_counter
        
        
        # Get Unique Scaffold Count at Each Iteration for Only the Hit Compounds
        df['Hit'] = df.apply(lambda _: '', axis=1)
        full_dataframe = pd.read_csv("../Datasets/Train/{}_train.csv".format(dataset))
        
        top_ten_percent = pd.DataFrame(full_dataframe.nlargest(len(full_dataframe)//10, 'Y')['SMILES'])
        hit_list = []

        for k in range(len(df['SMILES'])):
          for index, row in top_ten_percent.iterrows():
            if row['SMILES'] == df['SMILES'][k]:
              df['Hit'][k] = 'Yes'
            else:
              df['Hit'][k] = 'No'
        
        cnt = 0
        unique_scaffolds_hits = []
        unique_scaffold_hits_counter = []

        for l in range(len(scaffolds)):
          if scaffolds[l] in unique_scaffolds_hits:
            unique_scaffold_hits_counter.append(cnt)
          else:
            if df['Hit'][l] == 'No':
              unique_scaffold_hits_counter.append(cnt)
            else:
              cnt += 1
              unique_scaffolds_hits.append(scaffolds[l])
              unique_scaffold_hits_counter.append(cnt)
        
        unique_scaffold_hits_dataframe[dataset] = unique_scaffold_hits_counter    
        
      # Save the scaffolds for all compounds
      unique_scaffold_dataframe['Mean'] = unique_scaffold_dataframe.mean(axis=1)
      unique_scaffold_dataframe['Median'] = unique_scaffold_dataframe.median(axis=1)
      unique_scaffold_dataframe['Std'] = unique_scaffold_dataframe.std(axis=1)
      unique_scaffold_dataframe['SEM'] = unique_scaffold_dataframe.sem(axis=1)
      unique_scaffold_dataframe_summary = unique_scaffold_dataframe[['Mean', 'Median', 'Std', 'SEM']]
      unique_scaffold_dataframe_summary.to_csv('Unique_Scaffolds_{}_Summary_{}.csv'.format(models[i], group), index = False)
      unique_scaffold_dataframe.to_csv('Unique_Scaffolds_{}_{}.csv'.format(models[i], group), index = False)        
        
      # Save the scaffolds for the hit compounds
      unique_scaffold_dataframe['Mean'] = unique_scaffold_dataframe.mean(axis=1)
      unique_scaffold_dataframe['Median'] = unique_scaffold_dataframe.median(axis=1)
      unique_scaffold_dataframe['Std'] = unique_scaffold_dataframe.std(axis=1)
      unique_scaffold_dataframe['SEM'] = unique_scaffold_dataframe.sem(axis=1)
      unique_scaffold_dataframe_summary = unique_scaffold_dataframe[['Mean', 'Median', 'Std', 'SEM']]
      unique_scaffold_dataframe_summary.to_csv('Unique_Scaffolds_Hits_{}_Summary_{}.csv'.format(models[i], group), index = False)
      unique_scaffold_dataframe.to_csv('Unique_Scaffolds_Hits_{}_{}.csv'.format(models[i], group), index = False)