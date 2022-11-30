# %%
from ramanScript import ramanSpectra, loadSpectralData, splitDataBalanced
from resnet1d import ResNet1D, MyDataset

import numpy as np
import pickle
from collections import Counter
from tqdm import tqdm
from matplotlib import pyplot as plt
import random 
import umap

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# %matplotlib inline
# %%
# %%
def shuffleLists(l, seed=1234):
    random.seed(seed)
    l = list(zip(*l))
    random.shuffle(l)
    return list(zip(*l))
# %% Getting data
experiment = 'esamInit'
ramanData = np.load(f'../data/{experiment}/{experiment}.npy', allow_pickle=True)
scans = [scan for scan in ramanData if scan.cellSpectra.size>0]

# %%
phenotypes, spectra, cellLabels, identifiers = [], [], [], []

# Get testing
for scan in tqdm(scans):
    isCell = np.where(scan.cellSpectra>0)[0]
    for cell in isCell:
        spectra.append(scan.spectra[cell])
        phenotypes.append(scan.phenotype)
        identifier = scan.__str__()
        cellNum = scan.cellSpectra[cell]
        labeledCell = f'{identifier}-{cellNum}'
        identifiers.append(labeledCell)
spectra, phenotypes, identifiers = shuffleLists([spectra, phenotypes, identifiers])
spectra = np.array(spectra)
identifiers = np.array(identifiers)
phenotypes = np.array(phenotypes)

uniquePheno = set(phenotypes)
phenoDict = {phenotype: n for n, phenotype in zip(range(len(uniquePheno)), uniquePheno)}
phenoLabels = np.array([phenoDict[phenotype] for phenotype in phenotypes])

# I want to hold out n # cells from each phenotype
nHoldouts = 3

# Shuffle so that we know scans + cell numbers to hold out on
uniqueCells = list(set(identifiers))
random.seed(1234)
random.shuffle(uniqueCells)
cellsHoldoutCt = {phenotype: 0 for phenotype in set(phenotypes)}
cellsHoldout = []
# Add cells identifiers until we have the correct number of holdouts
for cell in uniqueCells:
    phenotype = cell.split('-')[1]
    if cellsHoldoutCt[phenotype]<nHoldouts:
        cellsHoldout.append(cell)
        cellsHoldoutCt[phenotype]+=1

    # Break early if we've found each value
    if sum(cellsHoldoutCt.values()) == len(set(phenotypes))*nHoldouts:
        break
        
testIdx = np.where(np.isin(identifiers, cellsHoldout))[0]
X_test = spectra[testIdx]
y_test = phenoLabels[testIdx]
trainIdx = np.where(~np.isin(identifiers, cellsHoldout))[0]
# Balance training data
phenoCt = {phenotype: 0 for phenotype in set(phenotypes)}
maxAmt = min(np.unique(phenotypes[trainIdx], return_counts=True)[1])

phenotypesTrain = phenoLabels[trainIdx]
spectraTrain = spectra[trainIdx]
labelIdx = []
for phenoLabel in set(phenoLabels):
    labelIdx += list(np.where(phenoLabels == phenoLabel)[0][0:maxAmt])
X_train = spectra[labelIdx, :]
y_train = phenoLabels[labelIdx]
# %%
reducer = umap.UMAP()
embeddingTrain = reducer.fit_transform(X_train)
embeddingTest = reducer.fit_transform(X_test)

embeddingFull = reducer.fit_transform(np.concatenate([X_train, X_test]))
# %%
phenoDict = {0: 'green', 1: 'red'}
phenoColors = [phenoDict[phenotype] for phenotype in y_train]
fig, ax = plt.subplots(figsize=(5,5))
esamNegIdx = np.array(y_train) == 1
plt.scatter(embeddingTrain[esamNegIdx,0], embeddingTrain[esamNegIdx,1], s=1.5, c='red', alpha=0.75, label='ESAM (-)')
plt.scatter(embeddingTrain[~esamNegIdx,0], embeddingTrain[~esamNegIdx,1], s=1.5, c='green', alpha=0.75, label='ESAM (+)')

for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
plt.xticks([])
plt.yticks([])
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.title('Raman Signal')
lgnd = plt.legend(loc='upper right')
for handle in lgnd.legendHandles:
    handle.set_sizes([50.0])
plt.show()
# %%
fig, ax = plt.subplots(figsize=(5,5))
esamNegIdx = np.array(y_test) == 1
plt.scatter(embeddingTest[esamNegIdx,0], embeddingTest[esamNegIdx,1], s=1.5, c='red', alpha=0.75, label='ESAM (-)')
plt.scatter(embeddingTest[~esamNegIdx,0], embeddingTest[~esamNegIdx,1], s=1.5, c='green', alpha=0.75, label='ESAM (+)')

for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
plt.xticks([])
plt.yticks([])
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.title('Raman Signal')
lgnd = plt.legend(loc='upper right')
for handle in lgnd.legendHandles:
    handle.set_sizes([50.0])
plt.show()
# %%
y_full = [f'train {pheno}' for pheno in y_train] + [f'test {pheno}' for pheno in y_test]
colors = ['red', 'green', 'blue', 'magenta']
fullDict = {pheno: color for pheno, color in zip(list(set(y_full)), colors)}
y_full_colors = np.array([fullDict[pheno] for pheno in y_full])

plt.figure(figsize=(10,10))

is_train = np.array([1 if pt.startswith('train') else 0 for pt in y_full ]) == 1
plt.scatter(embeddingFull[is_train,0], embeddingFull[is_train,1], s=1.5, c=y_full_colors[is_train], alpha=0.1)
plt.scatter(embeddingFull[~is_train,0], embeddingFull[~is_train,1], s=6, c=y_full_colors[~is_train])

# %%