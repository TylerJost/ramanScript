# %% [markdown]
"""
This is a script to train a neural network on background pixels. If the network is
able to predict on background alone it may signal that it is the imaging conditions
that create differences, not spectra identifying differences in cells. 
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

from skimage.draw import polygon2mask

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# %matplotlib inline
# %% Gather data
experiment = 'esamInit'
ramanData = np.load(f'../data/{experiment}/{experiment}.npy', allow_pickle=True)
scans = [scan for scan in ramanData if scan.cellSpectra.size>0]
# %% Set background pixels from COCO file
for scan in tqdm(scans):
    img = scan.makeImage()
    jsonData = f'../data/{scan.experiment}/annotations-background.json'
    with open(jsonData) as f:
        coco = json.load(f)
    imgIds = {}
    for cocoImg in coco['images']:
        imgIds[cocoImg['file_name']] = cocoImg['id']
    imgName = f'{scan.file.split(".")[0]}_{scan.ramanParams}_{scan.phenotype}.png'
    if imgName in imgIds.keys():
        annotations = [annotation for annotation in coco['annotations'] if annotation['image_id'] == imgIds[imgName] ]
    else:
        pass
        # return np.array([])
    fullMask = np.zeros(img.shape)
    c = 1
    for annotation in annotations:
        seg = annotation['segmentation'][0]
        # Converts RLE to polygon coordinates
        seg = np.array(seg).reshape(-1,2)
        # Necessary for mask conversion, otherwise coordinates are wrong
        seg[:,[0,1]] = seg[:,[1,0]]
        mask = polygon2mask(img.shape, seg)
        mask = mask.astype('uint8')
        # Set labels on full mask
        mask[mask == 1] = c
        fullMask += mask
        c += 1
    fullMask = np.zeros(img.shape)
    c = 1
    for annotation in annotations:
        seg = annotation['segmentation'][0]
        # Converts RLE to polygon coordinates
        seg = np.array(seg).reshape(-1,2)
        # Necessary for mask conversion, otherwise coordinates are wrong
        seg[:,[0,1]] = seg[:,[1,0]]
        mask = polygon2mask(img.shape, seg)
        mask = mask.astype('uint8')
        # Set labels on full mask
        mask[mask == 1] = c
        fullMask += mask
        c += 1

    scan.background = fullMask.ravel()
# %%
phenoDict = {'esamNeg': 0, 'esamPos': 1}
spectra, phenotypes = [], []
for scan in scans:
    isBg = scan.background>1
    spectra.append(scan.spectra[isBg])
    phenoNames = [scan.phenotype]*sum(isBg)
    phenoLabels = [phenoDict[name] for name in phenoNames]
    phenotypes.append(phenoLabels)

spectra = np.concatenate(spectra)
phenotypes = np.concatenate(phenotypes)

spectra, phenotypes = shuffleLists([spectra, phenotypes])
spectra = np.array(spectra)
phenotypes = np.array(phenotypes)

# Build train/test set
_, cts = np.unique(phenotypes, return_counts=True)
maxAmt = min(cts)
testSize = 0.1
nTrain = int(maxAmt*(1-testSize))
# Hold out test data
trainIdx, testIdx = [], []
for phenotype in set(phenotypes):
    isPhenotype = np.where(phenotypes == phenotype)[0]
    trainIdx.append(isPhenotype[0:nTrain])
    testIdx.append(isPhenotype[nTrain:])
trainIdx = np.concatenate(trainIdx)
testIdx  = np.concatenate(testIdx)

X_train = spectra[trainIdx,:]
y_train = phenotypes[trainIdx]

X_test = spectra[testIdx, :]
y_test = phenotypes[testIdx]






# %%
from sklearn.model_selection import StratifiedShuffleSplit
stratSplit = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=1234)
for train_index, test_index in stratSplit.split(spectra, phenotypes):
    pass