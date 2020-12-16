import numpy as np
import os
import matplotlib.pyplot as plt
from print_values import *
from plot_data_all_phonemes import *
from plot_data import *
import random
from sklearn.preprocessing import normalize
from get_predictions import *
from plot_gaussians import *

# File that contains the data
data_npy_file = 'data/PB_data.npy'

# Loading data from .npy file
data = np.load(data_npy_file, allow_pickle=True)
data = np.ndarray.tolist(data)

# Make a folder to save the figures
figures_folder = os.path.join(os.getcwd(), 'figures')
if not os.path.exists(figures_folder):
    os.makedirs(figures_folder, exist_ok=True)

# Array that contains the phoneme ID (1-10) of each sample
phoneme_id = data['phoneme_id']
# frequencies f1 and f2
f1 = data['f1']
f2 = data['f2']

# Initialize array containing f1 & f2, of all phonemes.
X_full = np.zeros((len(f1), 2))
#########################################
# Write your code here
# Store f1 in the first column of X_full, and f2 in the second column of X_full
X_full[:,0]=f1;
X_full[:,1]=f2;
########################################/
X_full = X_full.astype(np.float32)

# number of GMM components
k = 3


#########################################
# Write your code here
# Create an array named "X_phonemes_1_2", containing only samples that belong to phoneme 1 and samples that belong to phoneme 2.
# The shape of X_phonemes_1_2 will be two-dimensional. Each row will represent a sample of the dataset, and each column will represent a feature (e.g. f1 or f2)
# Fill X_phonemes_1_2 with the samples of X_full that belong to the chosen phonemes
# To fill X_phonemes_1_2, you can leverage the phoneme_id array, that contains the ID of each sample of X_full
# X_phonemes_1_2 = ...

row=np.sum(phoneme_id==1)+np.sum(phoneme_id==2 )
X_phonemes_1_2 = np.zeros((row,2))
label=np.zeros(row)
i=0;
for j in range(len(phoneme_id)):
    if phoneme_id[j] == 1 or phoneme_id[j] == 2:
        X_phonemes_1_2[i] = X_full[j]
        label[i]=phoneme_id[j]
        i=i+1

########################################/

# Plot array containing the chosen phonemes

# Create a figure and a subplot
fig, ax1 = plt.subplots()

title_string = 'Phoneme 1 & 2'
# plot the samples of the dataset, belonging to the chosen phoneme (f1 & f2, phoneme 1 & 2)
plot_data(X=X_phonemes_1_2, title_string=title_string, ax=ax1)
# save the plotted points of phoneme 1 as a figure
plot_filename = os.path.join(os.getcwd(), 'figures', 'dataset_phonemes_1_2.png')
plt.savefig(plot_filename)


#########################################
# Write your code here
# Get predictions on samples from both phonemes 1 and 2, from a GMM with k components, pretrained on phoneme 1
# Get predictions on samples from both phonemes 1 and 2, from a GMM with k components, pretrained on phoneme 2
# Compare these predictions for each sample of the dataset, and calculate the accuracy, and store it in a scalar variable named "accuracy"
GMM_Model = 'data/GMM_params_phoneme_01_k_03.npy';
Model = np.load(GMM_Model, allow_pickle=True)
M1 = Model.tolist()
mu = M1.get('mu')
s = M1.get('s')
p = M1.get('p')
Z1 = get_predictions(mu, s, p, X_phonemes_1_2)
GMM_Model = 'data/GMM_params_phoneme_02_k_03.npy';
Model = np.load(GMM_Model, allow_pickle=True)
M2= Model.tolist()
mu = M2.get('mu')
s = M2.get('s')
p = M2.get('p')
Z2 = get_predictions(mu, s, p, X_phonemes_1_2)
pred_label = np.zeros(row)
for i in range(len(label)):
    if np.sum(Z1[i]) > np.sum(Z2[i, 1]):
        pred_label[i] = 1
    else:
        pred_label[i] = 2
score = 0;
for i in range(len(label)):
    if pred_label[i] == label[i]:
        score = score + 1

accuracy = score / len(label) * 100


########################################

print('Accuracy using GMMs with {} components: {:.2f}%'.format(k, accuracy))

################################################
# enter non-interactive mode of matplotlib, to keep figures open
plt.ioff()
plt.show()