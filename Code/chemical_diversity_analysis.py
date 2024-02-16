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
 
          
models = ['DeepDelta5', 'ChemProp50', 'RandomForest', 'Random_Selection']
model_short_names = ['AD', 'CP', 'RF', 'Random']

for i in range(len(models)):
  unique_scaffold_dataframe = pd.DataFrame(columns = datasets)
  unique_scaffold_hits_dataframe = pd.DataFrame(columns = datasets)
  nearest_neighbor_dataframe = pd.DataFrame(columns = datasets)
  average_similarity_dataframe = pd.DataFrame(columns = datasets)

  for dataset in datasets:
    df = pd.read_csv('../Results/Exploitative_Active_Learning_Results/AL100_ExternalTest_{}_R1/{}_train_round_{}_200_R1.csv'.format(model_short_names[i], dataset, model[i]))

    # Prepare Scaffolds and Fingerprints
    mols = [Chem.MolFromSmiles(s) for s in df.SMILES]
    scaffolds = [Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m)) for m in mols]
    fps_list = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 1024) for m in mols]
    
    
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

    for i in range(len(df['SMILES'])):
      for index, row in top_ten_percent.iterrows():
        if row['SMILES'] == df['SMILES'][i]:
          df['Hit'][i] = 'Yes'
        else:
          df['Hit'][i] = 'No'
    
    cnt = 0
    unique_scaffolds_hits = []
    unique_scaffold_hits_counter = []

    for i in range(len(scaffolds)):
      if scaffolds[i] in unique_scaffolds_hits:
        unique_scaffold_hits_counter.append(cnt)
      else:
        if df['Hit'][i] == 'No':
          unique_scaffold_hits_counter.append(cnt)
        else:
          cnt += 1
          unique_scaffolds_hits.append(scaffolds[i])
          unique_scaffold_hits_counter.append(cnt)
    
    unique_scaffold_hits_dataframe[dataset] = unique_scaffold_hits_counter    
    
    
    ###################################
    ### Nearest Neighbor Similarity ###
    ###################################
    
    # Get Nearest Neighbor Tanimoto Similarity at Each Iteration
    similarities = []
    for j in range(len(fps_list)):
      if j > 1:
        current_slice = fps_list[0:j]

        similarity_list=[]
        NN_similarity_list=[]

        for k in range(len(current_slice)):
          if k == 0:
            similarity_list=[]
            similarity_list = DataStructs.BulkTanimotoSimilarity(current_slice[k], current_slice[k+1:])
            NN_similarity_list.append(max(similarity_list))
            continue
          similarity_list=[]
          similarity_list = DataStructs.BulkTanimotoSimilarity(current_slice[k], current_slice[k+1:] + current_slice[:k-1])
          if len(similarity_list) > 0:
            NN_similarity_list.append(max(similarity_list))
        similarities.append(statistics.mean(NN_similarity_list))
    nearest_neighbor_dataframe[dataset] = similarities
    
    
    ###################################
    ### Average Tanimoto Similarity ###
    ###################################
    
    # Get Average Tanimoto Similarity at Each Iteration
    similarities = []
    for j in range(len(fps_list)):
      if j > 1:
        current_slice = fps_list[0:j]
        similarity_list = sum([DataStructs.BulkTanimotoSimilarity(current_slice[j], current_slice[j+1:]) for j in range(len(current_slice) - 1)], [])
        similarities.append(statistics.mean(similarity_list))
    average_similarity_dataframe[dataset] = similarities
    
    
  # Get Summary Scaffold Information
  unique_scaffold_dataframe['Mean'] = unique_scaffold_dataframe.mean(axis=1)
  unique_scaffold_dataframe['Median'] = unique_scaffold_dataframe.median(axis=1)
  unique_scaffold_dataframe['Std'] = unique_scaffold_dataframe.std(axis=1)
  unique_scaffold_dataframe['SEM'] = unique_scaffold_dataframe.sem(axis=1)
  unique_scaffold_dataframe_summary = unique_scaffold_dataframe[['Mean', 'Median', 'Std', 'SEM']]
  unique_scaffold_dataframe_summary.to_csv('Unique_Scaffolds_{}_Summary_R1.csv'.format(model[i]), index = False)
  unique_scaffold_dataframe.to_csv('Unique_Scaffolds_{}_R1.csv'.format(model[i]), index = False)
  
  # Get Summary Scaffold Information for Hits
  unique_scaffold_hits_dataframe['Mean'] = unique_scaffold_hits_dataframe.mean(axis=1)
  unique_scaffold_hits_dataframe['Median'] = unique_scaffold_hits_dataframe.median(axis=1)
  unique_scaffold_hits_dataframe['Std'] = unique_scaffold_hits_dataframe.std(axis=1)
  unique_scaffold_hits_dataframe['SEM'] = unique_scaffold_hits_dataframe.sem(axis=1)
  unique_scaffold_hits_dataframe_summary = unique_scaffold_hits_dataframe[['Mean', 'Median', 'Std', 'SEM']]
  unique_scaffold_hits_dataframe_summary.to_csv('Unique_Scaffolds_Hits_{}_Summary_R1.csv'.format(model), index = False)
  unique_scaffold_hits_dataframe.to_csv('Unique_Scaffolds_Hits_{}_R1.csv'.format(model), index = False)
  
  # Get Summary Nearest Neighbor Information  
  nearest_neighbor_dataframe['Mean'] = nearest_neighbor_dataframe.mean(axis=1)
  nearest_neighbor_dataframe['Median'] = nearest_neighbor_dataframe.median(axis=1)
  nearest_neighbor_dataframe['Std'] = nearest_neighbor_dataframe.std(axis=1)
  nearest_neighbor_dataframe['SEM'] = nearest_neighbor_dataframe.sem(axis=1)
  nearest_neighbor_dataframe_summary = nearest_neighbor_dataframe[['Mean', 'Median', 'Std', 'SEM']]
  nearest_neighbor_dataframe_summary.to_csv('Average_Nearest_Neighbor_Similarity_{}_Summary.csv'.format(model[i]), index = False)
  nearest_neighbor_dataframe.to_csv('Average_Nearest_Neighbor_Similarity_{}.csv'.format(model[i]), index = False)

  # Get Summary Average Similarity Information  
  average_similarity_dataframe['Mean'] = average_similarity_dataframe.mean(axis=1)
  average_similarity_dataframe['Median'] = average_similarity_dataframe.median(axis=1)
  average_similarity_dataframe['Std'] = average_similarity_dataframe.std(axis=1)
  average_similarity_dataframe['SEM'] = average_similarity_dataframe.sem(axis=1)
  average_similarity_dataframe_summary = average_similarity_dataframe[['Mean', 'Median', 'Std', 'SEM']]
  average_similarity_dataframe_summary.to_csv('Average_Tanimoto_Similarity_{}_Summary.csv'.format(model[i]), index = False)
  average_similarity_dataframe.to_csv('Average_Tanimoto_Similarity_{}.csv'.format(model[i]), index = False)