#####################################################################################################################
###          t-SNE inspired by Practical Cheminformatics Post 'Visualizing Chemical Space' by Pat Walters         ###
###        Post Link: https://practicalcheminformatics.blogspot.com/2019/11/visualizing-chemical-space.html       ###
### Code: https://github.com/PatWalters/workshop/blob/master/predictive_models/2_visualizing_chemical_space.ipynb ###
#####################################################################################################################

# Imports
import os
import tempfile
import shutil
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import tempfile
import copy



# Set a few parameters to improve the appearance of our plots
sns.set(rc={'figure.figsize': (10, 10)})
sns.set(font_scale=1.5)
sns.set_style('whitegrid')

# Define a couple of functions to convert a list SMILES to a list of fingerprints.
def fp_list_from_smiles_list(smiles_list, n_bits=2048):
    fp_list = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol == None:
            break
        fp_list.append(fp_as_array(mol, n_bits))
    return fp_list

def fp_as_array(mol,n_bits=2048):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
    arr = np.zeros((1,), int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

# Color conversion from HEX to RGB values: #FFFFFF -> [255,255,255]
def hex_to_RGB(hex_str):
    #Pass 16 to the integer function for change of base
    return [int(hex_str[i:i+2], 16) for i in range(1,6,2)]

# Given two hex colors, return a color gradient with n colors.
def get_color_gradient(color1, color2, n): 
    assert n > 1
    color1_rgb = np.array(hex_to_RGB(color1))/255
    color2_rgb = np.array(hex_to_RGB(color2))/255
    mix_pcts = [x/(n-1) for x in range(n)]
    rgb_colors = [((1-mix)*color1_rgb + (mix*color2_rgb)) for mix in mix_pcts]
    return ["#" + "".join([format(int(round(val*255)), "02x") for val in item]) for item in rgb_colors]
    


# Adjustable parameters 
dataset = 'CHEMBL232-1' # Dataset of interest
start = 1 # Which iteration you want to start plotting with (starts at 1)
end = 16 # Which iteration you want to end plotting with
colors = get_color_gradient('#DDDDDD', '#000000', end-start+2) # Choose color scale for arrows in hex code
    

    
#############
### t-SNE ###
#############
    
# Read training dataset
df = pd.read_csv('../Datasets/Train/{}_train.csv'.format(dataset)) 

# Convert the SMILES from our dataframe to fingerprints
fp_list = fp_list_from_smiles_list(df.SMILES)

# Perform principal component analysis (PCA) on the fingerprints.
pca = PCA(n_components=50)
crds = pca.fit_transform(fp_list)
   
# Run the t-sne on the 50 principal component database we created above. 
crds_embedded = TSNE(n_components=2).fit_transform(crds)
tsne_df = pd.DataFrame(crds_embedded,columns=["X","Y"])
tsne_df['SMILES'] = df['SMILES']
    

################
### Plotting ###    
################

models = ['DeepDelta5', 'ChemProp50', 'Delta_XGBoost', 'XGBoost', 'RandomForest', 'Random_Selection']
model_short_names = ['ADCP', 'CP', 'ADXGB', 'XGB', 'RF', 'Random']
model_color = ['#721f81', '#31688e', 'b', 'g', '#bc3754', 'k']

for i in range(len(models)):
    
    result_df = pd.read_csv('../Results/Exploitative_Active_Learning_Results/AL_Exploitative_{}_R1/{}_train_round_{}_200_R1.csv'.format(model_short_names[i], dataset, models[i])).rename({'Y': 'Values'}, axis=1)
    tsne_df2 = pd.merge(tsne_df, result_df, on=['SMILES'], how='inner', indicator=False)

    # Plot background values
    fig= plt.figure()
    ax = sns.scatterplot(data=tsne_df,x="X",y="Y",color='lightblue', s=60)

    # Plot top ten percent of values
    top_ten_percent = pd.DataFrame(tsne_df2.nlargest(len(tsne_df2)//10, 'Values'))
    ax = sns.scatterplot(data=top_ten_percent,x="X",y="Y", marker="*", s=360, color='blue')

    for iter in range(start,end + 1):
      iteration_df = tsne_df2.loc[tsne_df2['Iteration'] == iter] # Current iteration
      iteration_next_df = tsne_df2.loc[tsne_df2['Iteration'] == iter+1] # Next iteration

      # Plot the predictions with an arrow indicating going from iteration n to iteration n+1
      plt.plot(iteration_df['X'], iteration_df['Y'], marker="o", markersize=10, color=model_color[i])
      ax.quiver(iteration_df['X'], iteration_df['Y'], (iteration_next_df['X'].values[0]-iteration_df['X'].values[0]), (iteration_next_df['Y'].values[0]-iteration_df['Y'].values[0]), color=colors[iter-start+1], angles='xy', scale_units='xy', scale=1, alpha = 0.9)
      plt.plot(iteration_next_df['X'], iteration_next_df['Y'], marker="o", markersize=10, color=model_color[i])

      # Make a gold star if the datapoint matches one of the top ten values
      for j in range(start, iter):
        iteration_df2 = tsne_df2.loc[tsne_df2['Iteration'] == j]
        for index, row in top_ten_percent.iterrows():
          if row['SMILES'] == iteration_df2['SMILES'].values:
            plt.plot(iteration_df2['X'], iteration_df2['Y'], marker="*", markersize=20, markerfacecolor='#fbb61a', markeredgewidth=1, markeredgecolor='k')

    plt.savefig("t-SNE_{}_{}_{}_to_{}.png".format(dataset, models[i], start, end), facecolor='white')


