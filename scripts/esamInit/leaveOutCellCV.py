# %% [markdown]
"""
To determine the strength of our network, we will perform a cross-validation 
evaluation holding out a certain number of cells each time. 
"""
# %%
from ramanScript import ramanSpectra, shuffleLists
from resnet1d import ResNet1D, MyDataset

import numpy as np
import pickle

from tqdm import tqdm
from matplotlib import pyplot as plt
import random
from sklearn.metrics import classification_report, confusion_matrix


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# %matplotlib inline
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
# %%
trainIdx = np.where(~np.isin(identifiers, cellsHoldout))[0]
# Balance training data
phenoCt = {phenotype: 0 for phenotype in set(phenotypes)}
maxAmt = min(np.unique(phenotypes[trainIdx], return_counts=True)[1])

phenotypesTrain = phenoLabels[trainIdx]
spectraTrain = spectra[trainIdx]
labelIdx = []
for phenoLabel in set(phenoLabels):
    labelIdx += list(np.where(phenotypesTrain == phenoLabel)[0][0:maxAmt])
X_train = spectraTrain[labelIdx, :]
y_train = phenotypesTrain[labelIdx]
# Shuffle again
X_train, y_train = shuffleLists([X_train, y_train])
X_train = np.array(X_train)
y_train = np.array(y_train)