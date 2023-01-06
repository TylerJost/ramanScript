# %% [markdown]
"""
This is the neural network to load and train a neural network on
time series classification data
"""
# %%
import sys
sys.path.append('../../ramanScript')
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
# %% Getting data
experiment = 'esam2'
reportName = f'{experiment}LOCellDenoisedFinal'

ramanData = np.load(f'../../data/{experiment}/{experiment}.npy', allow_pickle=True)
scans = [scan for scan in ramanData if scan.cellSpectra.size>0]

# %%
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
# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

batch_size=32
print(X_train.shape, y_train.shape)
train_dataset = MyDataset(X_train, y_train)
test_dataset = MyDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=False)

# %%
dataiter = iter(train_loader)
spectra, labels = dataiter.next()
print(spectra.shape)
# %%
# Hyper-parameters
num_epochs = 50
# Model parameters
device_str = "cuda"
device = torch.device(device_str if torch.cuda.is_available() else "cpu")
kernel_size = 16
stride = 2
n_block = 48
downsample_gap = 6
increasefilter_gap = 12
model = ResNet1D(
    in_channels=1, 
    base_filters=128, # 64 for ResNet1D, 352 for ResNeXt1D
    kernel_size=kernel_size, 
    stride=stride, 
    groups=32, 
    n_block=n_block, 
    n_classes=2, 
    downsample_gap=downsample_gap, 
    increasefilter_gap=increasefilter_gap, 
    use_do=True)
model.to(device)

# train and test
model.verbose = False
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
criterion = torch.nn.CrossEntropyLoss()
# %%
allLoss = []
for epoch in tqdm(range(num_epochs), desc=f"epoch", leave=False):
    # train
    model.train()
    for i, batch in enumerate(tqdm(train_loader, desc=f"spectra", position=0, leave=True)):

        spectra, labels = tuple(t.to(device) for t in batch)
        # spectra = spectra.to(device)
        # labels = labels.to(device)

        outputs = model(spectra)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        allLoss.append(float(loss.cpu().detach().numpy()))
    scheduler.step(loss)
    if epoch % 5  == 0:
        torch.save(model.state_dict(), f'../../models/{experiment}LOCellFinal.pth')
        np.save(f'../../models/{reportName}.npy', np.array(allLoss))

# %%
# plt.plot(allLoss)

# %%
torch.save(model.state_dict(), f'../../models/{experiment}LOCellFinal.pth')
np.save(f'../../models/{reportName}.npy', np.array(allLoss))
