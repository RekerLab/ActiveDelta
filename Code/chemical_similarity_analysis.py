models = ['DeepDelta5', 'Delta_XGBoost', 'ChemProp50', 'RandomForest', 'XGBoost']
rounds = ['R1', 'R2', 'R3']

for round in rounds:

  final_results = pd.DataFrame({'Dataset': datasets})
  final_results["DeepDelta5"] = np.nan
  final_results["Delta_XGBoost"] = np.nan
  final_results["ChemProp50"] = np.nan
  final_results["RandomForest"] = np.nan
  final_results["XGBoost"] = np.nan

  for model in models:

    similarities = pd.DataFrame({'Dataset': datasets})
    similarities["Value"] = np.nan
    dataset_number = 0

    for dataset in datasets:
      # Prepare dataframes
      train_df = pd.read_csv('{}_train.csv'.format(dataset))
      test_set = pd.read_csv("{}_test.csv".format(dataset))

      try:
        potent_molecule_df = pd.read_csv('{}_{}_AL100_{}_Test_Single_Predictions.csv'.format(dataset, model, round)).T
      except:
        potent_molecule_df = pd.read_csv('{}_{}_AL100_{}_Test_Single_PredictionsCorrect.csv'.format(dataset, model, round)).T
      potent_molecule_df.columns =['True', 'Pred']
      Preds = potent_molecule_df['Pred']
      Preds = [float(i) for i in Preds]
      test_set['Pred'] = Preds

      # Get most potent prediction
      potent_molecule = pd.DataFrame(test_set.nlargest(1, 'Pred')['SMILES'])
      potent_molecule = potent_molecule.reset_index()
      potent_molecule_mol = Chem.MolFromSmiles(potent_molecule['SMILES'][0])
      potent_molecule_fp = AllChem.GetAtomPairFingerprint(potent_molecule_mol)

      # Prepare Fingerprints of training data
      mols = [Chem.MolFromSmiles(s) for s in train_df.SMILES]
      fps_list = [AllChem.GetAtomPairFingerprint(m) for m in mols]

      # Get Maximum similarity in test set to the most potent molecule
      NN_similarity = max(DataStructs.BulkTanimotoSimilarity(potent_molecule_fp, fps_list))
      similarities['Value'][dataset_number] = NN_similarity
      dataset_number += 1

    final_results[model] = similarities['Value']

  final_results.to_csv('TanimotoSimilaritiesToTrainingSet_AtomPair_{}.csv'.format(round), index = False)