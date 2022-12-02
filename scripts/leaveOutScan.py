# %% [markdown]
"""

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
import json

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

from skimage.draw import polygon2mask

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# %matplotlib inline
# %% Flags
train = 1
# %% Gather data
experiment = 'esamInit'
ramanData = np.load(f'../data/{experiment}/{experiment}.npy', allow_pickle=True)
scans = [scan for scan in ramanData if scan.cellSpectra.size>0]
# %% Reserve scans for training or testing
random.seed(1234)
random.shuffle(scans)
holdout = [ 'esamInit-esamPos-0.5umPerPixel-Scan3', 
            'esamInit-esamNeg-0.5umPerPixel-Scan3',
            'esamInit-esamNeg-1umPerPixel-Scan2',
            'esamInit-esamPos-1umPerPixel-Scan3']

# holdout = [ 'esamInit-esamPos-0.5umPerPixel-Scan3', 
#             'esamInit-esamNeg-0.5umPerPixel-Scan3']
testScans, trainScans = [], []
for scan in scans:
    if scan.__str__() in holdout:
        testScans.append(scan)
    else:
        trainScans.append(scan)

# %% Separate out balanced train/test splits
phenotypes = []
for scan in trainScans:
    isCell = scan.cellSpectra>0
    phenotypes += [scan.phenotype]*sum(isCell)

maxAmt = min(np.unique(phenotypes, return_counts=True)[1])

# Get all spectra for training
X_train, y_train = [], []
for scan in trainScans:
    isCell = scan.cellSpectra>0
    X_train.append(scan.spectra[isCell])
    y_train += [scan.phenotype]*sum(isCell)
X_train = np.concatenate(X_train)
y_train = np.array(y_train)
# Shuffle so we can get points from each scan
X_train, y_train = shuffleLists([X_train, y_train])
X_train = np.array(X_train)
y_train = np.array(y_train)
trainIdx = []
# Balance dataset
for phenotype in set(y_train):
    isPhenotype = np.where(y_train == phenotype)[0]
    isPhenotype = isPhenotype[0:maxAmt]
    trainIdx += list(isPhenotype)
# Shuffle again
X_train = X_train[trainIdx,:]
y_train = y_train[trainIdx]
X_train, y_train = shuffleLists([X_train, y_train])
X_train = np.array(X_train)
y_train = np.array(y_train)

# Get test spectra
X_test, y_test = [], []
for scan in testScans:
    isCell = scan.cellSpectra>0
    X_test.append(scan.spectra[isCell])
    y_test += [scan.phenotype]*sum(isCell)
X_test = np.concatenate(X_test)
y_test = np.array(y_test)

uniquePheno = set(phenotypes)
phenoDict = {phenotype: n for n, phenotype in zip(range(len(uniquePheno)), uniquePheno)}
y_train = np.array([phenoDict[pt] for pt in y_train])
y_test =  np.array([phenoDict[pt] for pt in y_test])
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
if train:
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
    reportName = f'{experiment}LOScan'
    torch.save(model.state_dict(), f'../models/{reportName}.pth')
# %%
reportName = f'{experiment}LOScan'
model.load_state_dict(torch.load(f'../models/{reportName}.pth', map_location=device))
# %%
probs = []
allLabels = []
scores = []
for i, batch in enumerate(tqdm(train_loader, desc=f"spectra", position=0, leave=True)):

    spectra, labels = tuple(t.to(device) for t in batch)
    # spectra = spectra.to(device)
    # labels = labels.to(device)

    outputs = model(spectra)
    probs.append(outputs.cpu().data.numpy())
    allLabels.append(labels.cpu().data.numpy())
    scores.append(F.softmax(outputs, dim=1).cpu().data.numpy())
    
probs = np.concatenate(probs)
allLabels = np.concatenate(allLabels)
scores = np.concatenate(scores)
# TODO: Fix switched labels(?)
allLabels = ~allLabels+2
assert set(allLabels) == {0,1}
pred = np.argmax(probs, axis=1)
# %%
fpr, tpr, _ = roc_curve(allLabels, scores[:,1])
roc_auc = roc_auc_score(allLabels, scores[:,1])

plt.figure(figsize=(6,6))
plt.rcParams.update({'font.size': 17})
plt.plot(fpr, tpr,linewidth=3)
plt.grid()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'AUC = {roc_auc:0.3f}')