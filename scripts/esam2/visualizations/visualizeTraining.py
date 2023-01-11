# %%
import sys
sys.path.append('../../ramanScript')
from ramanScript import ramanSpectra, shuffleLists
from resnet1d import ResNet1D, MyDataset

import numpy as np
import pickle
import umap
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
def plotUMAP(embedding, labels, title = '', fileName = ''):
    fig, ax = plt.subplots(figsize=(8,8))
    uniqueLabels = list(set(labels)) 
    uniqueLabels = sorted(uniqueLabels)[::-1]
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for label in uniqueLabels:
        labelIdx = np.where(labels == label)[0]
        if label.startswith('test'):
            alphaVal = 0.5
            sizeVal = 3
        else:
            alphaVal = 0.25
            sizeVal = 2
        plt.scatter(embedding[labelIdx,0], embedding[labelIdx,1], 
                    s = sizeVal, alpha = alphaVal, label=label)
    lgnd = plt.legend()
    for handle in lgnd.legendHandles:
        handle.set_sizes([50.0])
        handle.set_alpha([1])

    if len(fileName) > 0:
        plt.savefig(fileName)
    
    plt.title(title)
    plt.axis('off')
    plt.show()
# %% Getting data
experiment = 'esam2'
reportName = f'{experiment}LOCellDenoisedFinal'

ramanData = np.load(f'../../../data/{experiment}/{experiment}.npy', allow_pickle=True)
# %% Leave out cell
scans = [scan for scan in ramanData if scan.cellSpectra.size>0]

phenotypes, spectra, cellLabels, identifiers = [], [], [], []

# Get testing
for scan in tqdm(scans):
    isCell = np.where(scan.cellSpectra>0)[0]
    for cell in isCell:
        spectra.append(scan.spectraDenoised[cell])
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
nHoldouts = 5

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
    labelIdx += list(np.where(phenotypesTrain == phenoLabel)[0][0:maxAmt])
X_train = spectraTrain[labelIdx, :]
y_train = phenotypesTrain[labelIdx]
# Shuffle again
X_train, y_train = shuffleLists([X_train, y_train])
X_train = np.array(X_train)
y_train = np.array(y_train)

spectra = np.concatenate([X_train, X_test])
labels = np.array(['train-'+str(label) for label in y_train]+['test-'+str(label) for label in y_test])
# Create umap embedding
reducer = umap.UMAP()
embedding = reducer.fit_transform(spectra)
plotUMAP(embedding, labels, title = 'Leave Out Cell UMAP', fileName = '')
# %% Whole
testSize = 0.15
scans = [scan for scan in ramanData if scan.cellSpectra.size>0]
random.seed(1234)
random.shuffle(scans)
spectra, phenotypes = [], []
for scan in scans:
    cellIdx = np.where(scan.cellSpectra>0)[0]
    for cell in cellIdx:
        spectra.append(scan.spectraDenoised[cell,:])
        phenotypes.append(scan.phenotype)

uniquePheno = np.unique(phenotypes)
spectra = np.array(spectra)
unique, cts = np.unique(phenotypes, return_counts=True)
maxCt = int(np.min(cts)*(1-testSize))

spectra, phenotypes = shuffleLists([spectra, phenotypes])
spectra = np.array(spectra)

X_train, y_train, X_test, y_test = [], [], [], []

phenoCount = {pheno: 0 for pheno in uniquePheno}
phenoDict = {phenotype: n for n, phenotype in zip(range(len(uniquePheno)), uniquePheno)}
for signal, phenotype in zip(spectra, phenotypes):
    if phenoCount[phenotype] <= maxCt:
        X_train.append(signal)
        y_train.append(phenoDict[phenotype])

        phenoCount[phenotype] += 1
    else:
        X_test.append(signal)
        y_test.append(phenoDict[phenotype])


X_train = np.array(X_train)
X_test = np.array(X_test)

spectra = np.concatenate([X_train, X_test])
labels = np.array(['train-'+str(label) for label in y_train]+['test-'+str(label) for label in y_test])
# Create umap embedding
reducer = umap.UMAP()
embedding = reducer.fit_transform(spectra)
plotUMAP(embedding, labels, title = 'Train on Whole UMAP', fileName = '')


# %% Leave out scan
nScansTest = 2
scans = [scan for scan in ramanData if scan.cellSpectra.size>0]
phenotypes = [scan.phenotype for scan in scans]
uniquePheno = set(phenotypes)
phenoCount = {phenotype: 0 for phenotype in uniquePheno}
phenoDict = {phenotype: n for n, phenotype in zip(range(len(uniquePheno)), uniquePheno)}

testScans = []
trainScans = []
random.seed(1234)
random.shuffle(scans)
for scan in scans:
    if phenoCount[scan.phenotype] < nScansTest:
        testScans.append(scan)
        phenoCount[scan.phenotype] += 1
    else:
        trainScans.append(scan)

X_train, y_train, X_test, y_test = [], [], [], []
for scan in trainScans:
    cellIdx = np.where(scan.cellSpectra>0)[0]
    X_train.append(scan.spectra[cellIdx,:])
    y_train.append([phenoDict[scan.phenotype]]*len(cellIdx))

X_train = np.concatenate(X_train)
y_train = np.concatenate(y_train)

# Try shuffling all the data
X_train, y_train = shuffleLists([X_train, y_train])
X_train = np.array(X_train)
y_train = np.array(y_train)

for scan in testScans:
    cellIdx = np.where(scan.cellSpectra>0)[0]
    X_test.append(scan.spectra[cellIdx,:])
    y_test.append([phenoDict[scan.phenotype]]*len(cellIdx))

X_test = np.concatenate(X_test)
y_test = np.concatenate(y_test)

reducer = umap.UMAP()
embedding = reducer.fit_transform(spectra)
plotUMAP(embedding, labels, title = 'Leave out Scan UMAP', fileName = '')
# %%