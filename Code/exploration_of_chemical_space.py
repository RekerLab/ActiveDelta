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
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import imageio
import tempfile
import copy



# Set a few parameters to improve the appearance of our plots
sns.set(rc={'figure.figsize': (10, 10)})
sns.set(font_scale=1.5)
sns.set_style('whitegrid')

# Define a couple of functions to convert a list SMILES to a list of fingerprints.
def fp_list_from_smiles_list(smiles_list, n_bits=2048):
    fp_list = []
    for smiles in tqdm(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol == None:
            break
        fp_list.append(fp_as_array(mol, n_bits))
    return fp_list

def fp_as_array(mol,n_bits=2048):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
    arr = np.zeros((1,), np.int)
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
    
    

############################
### Original 56 Datasets ###
############################   


# Adjustable parameters 
dataset = 'CHEMBL3887887' # Dataset of interest
start = 20 # Which iteration you want to start plotting with (starts at 1)
end = 41 # Which iteration you want to end plotting with
colors = get_color_gradient('#DDDDDD', '#000000', end-start+2) # Choose color scale for arrows in hex code
    

    
#############
### t-SNE ###
#############
    
# Read training dataset
df = pd.read_csv('../Datasets/Original_56/Train/{}_train.csv'.format(dataset)) 

# Convert the SMILES from our dataframe to fingerprints
fp_list = fp_list_from_smiles_list(df.SMILES)

# Perform principal component analysis (PCA) on the fingerprints.
pca = PCA(n_components=50)
crds = pca.fit_transform(fp_list)
   
# Run the t-sne on the 50 principal component database we created above. 
%time crds_embedded = TSNE(n_components=2).fit_transform(crds)
tsne_df = pd.DataFrame(crds_embedded,columns=["X","Y"])
tsne_df['SMILES'] = df['SMILES']
    



################
### Plotting ###    
################

models = ['Random', 'RF', 'CP', 'DD']
model_color = ['k', '#bc3754', '#31688e', '#721f81']
cnt = 0 # Counter for model colors

for model in models:
    
    # Read
    result_df = pd.read_csv('../Results/Original_56/Exploitative_Active_Learning_Results/Plotting/{}/{}_train_round_{}.csv'.format(model, dataset, model)).rename({'Y': 'Values'}, axis=1)

    tsne_df2 = pd.merge(tsne_df, result_df, on=['SMILES'], how='inner', indicator=False)

    # Plotting
    fig= plt.figure()

    # Plot background values
    ax = sns.scatterplot(data=tsne_df,x="X",y="Y",color='lightblue', s=50)

    # Plot top ten percent of values
    top_ten_percent = pd.DataFrame(tsne_df2.nlargest(len(tsne_df2)//10, 'Values'))
    ax = sns.scatterplot(data=top_ten_percent,x="X",y="Y", marker="*", s=300, color='blue')


    for iter in range(start,end + 1):
      iteration_df = tsne_df2.loc[tsne_df2['Iteration'] == iter] # Current iteration
      iteration_next_df = tsne_df2.loc[tsne_df2['Iteration'] == iter+1] # Next iteration

      # Plot the predictions with an arrow indicating going from iteration n to iteration n+1
      plt.plot(iteration_df['X'], iteration_df['Y'], marker="o", markersize=5, color=model_color[cnt])
      ax.quiver(iteration_df['X'], iteration_df['Y'], (iteration_next_df['X'].values[0]-iteration_df['X'].values[0]), (iteration_next_df['Y'].values[0]-iteration_df['Y'].values[0]), color=colors[iter-start+1], angles='xy', scale_units='xy', scale=1, alpha = 0.9)
      plt.plot(iteration_next_df['X'], iteration_next_df['Y'], marker="o", markersize=5, color=model_color[cnt])

      # Make a Gold star if the datapoint matches one of the top ten values
      for i in range(start, iter):
        iteration_df2 = tsne_df2.loc[tsne_df2['Iteration'] == i]
        for index, row in top_ten_percent.iterrows():
          if row['SMILES'] == iteration_df2['SMILES'].values:
            plt.plot(iteration_df2['X'], iteration_df2['Y'], marker="*", markersize=20, markerfacecolor='#fbb61a', markeredgewidth=1, markeredgecolor='k')

    plt.savefig("t-SNE_{}_{}_{}_to_{}.png".format(dataset, model, start, end), facecolor='white')
    
    cnt += 1
    
    

###########################
### Updated 99 Datasets ###
###########################  


# Adjustable parameters 
dataset = 'CHEMBL3887887' # Dataset of interest # ADJUST 
start = 20 # Which iteration you want to start plotting with (starts at 1)
end = 41 # Which iteration you want to end plotting with
colors = get_color_gradient('#DDDDDD', '#000000', end-start+2) # Choose color scale for arrows in hex code

    
#############
### t-SNE ###
#############
    
# Read training dataset
df = pd.read_csv('../Datasets/Updated_99/Train/{}_train.csv'.format(dataset)) 

# Convert the SMILES from our dataframe to fingerprints
fp_list = fp_list_from_smiles_list(df.SMILES)

# Perform principal component analysis (PCA) on the fingerprints.
pca = PCA(n_components=50)
crds = pca.fit_transform(fp_list)
   
# Run the t-sne on the 50 principal component database we created above. 
%time crds_embedded = TSNE(n_components=2).fit_transform(crds)
tsne_df = pd.DataFrame(crds_embedded,columns=["X","Y"])
tsne_df['SMILES'] = df['SMILES']
    



################
### Plotting ###    
################

models = ['Random', 'RF', 'CP', 'DD']
model_color = ['k', '#bc3754', '#31688e', '#721f81']
cnt = 0 # Counter for model colors

for model in models:
    
    # Read
    result_df = pd.read_csv('../Results/Updated_99/Exploitative_Active_Learning_Results/Plotting/{}/{}_train_round_{}.csv'.format(model, dataset, model)).rename({'Y': 'Values'}, axis=1)

    tsne_df2 = pd.merge(tsne_df, result_df, on=['SMILES'], how='inner', indicator=False)

    # Plotting
    fig= plt.figure()

    # Plot background values
    ax = sns.scatterplot(data=tsne_df,x="X",y="Y",color='lightblue', s=50)

    # Plot top ten percent of values
    top_ten_percent = pd.DataFrame(tsne_df2.nlargest(len(tsne_df2)//10, 'Values'))
    ax = sns.scatterplot(data=top_ten_percent,x="X",y="Y", marker="*", s=300, color='blue')


    for iter in range(start,end + 1):
      iteration_df = tsne_df2.loc[tsne_df2['Iteration'] == iter] # Current iteration
      iteration_next_df = tsne_df2.loc[tsne_df2['Iteration'] == iter+1] # Next iteration

      # Plot the predictions with an arrow indicating going from iteration n to iteration n+1
      plt.plot(iteration_df['X'], iteration_df['Y'], marker="o", markersize=5, color=model_color[cnt])
      ax.quiver(iteration_df['X'], iteration_df['Y'], (iteration_next_df['X'].values[0]-iteration_df['X'].values[0]), (iteration_next_df['Y'].values[0]-iteration_df['Y'].values[0]), color=colors[iter-start+1], angles='xy', scale_units='xy', scale=1, alpha = 0.9)
      plt.plot(iteration_next_df['X'], iteration_next_df['Y'], marker="o", markersize=5, color=model_color[cnt])

      # Make a Gold star if the datapoint matches one of the top ten values
      for i in range(start, iter):
        iteration_df2 = tsne_df2.loc[tsne_df2['Iteration'] == i]
        for index, row in top_ten_percent.iterrows():
          if row['SMILES'] == iteration_df2['SMILES'].values:
            plt.plot(iteration_df2['X'], iteration_df2['Y'], marker="*", markersize=20, markerfacecolor='#fbb61a', markeredgewidth=1, markeredgecolor='k')

    plt.savefig("t-SNE_{}_{}_{}_to_{}.png".format(dataset, model, start, end), facecolor='white')
    
    cnt += 1