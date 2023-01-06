# %%
"""
Modeling tools to ease analysis
"""
# %%
import random
import numpy as np

from resnet1d import ResNet1D, MyDataset

import torch
from torch.utils.data import DataLoader
# %%
def shuffleLists(l, seed=1234):
    random.seed(seed)
    l = list(zip(*l))
    random.shuffle(l)
    return list(zip(*l))

def makeSplit(trainScans, testScans, phenoDict):
    """
    Splits lists of scans into suitable formats for model analysis

    Input:
        - trainScans: A list of scans of class ramanSpectra
        - testScans: A list of scans of class ramanSpectra
        - phenoDict: Dictionary to encode labels from phenotypes
    
    Output:
        - X_train, y_train, X_test, y_test: Numpy arrays for training/testing
    """
    X_train, y_train, X_test, y_test = [], [], [], []
    for scan in trainScans:
        cellIdx = np.where(scan.cellSpectra>0)[0]
        X_train.append(scan.spectra[cellIdx,:])
        y_train.append([phenoDict[scan.phenotype]]*len(cellIdx))

    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)

    for scan in testScans:
        cellIdx = np.where(scan.cellSpectra>0)[0]
        X_test.append(scan.spectra[cellIdx,:])
        y_test.append([phenoDict[scan.phenotype]]*len(cellIdx))

    X_test = np.concatenate(X_test)
    y_test = np.concatenate(y_test)

    return X_train, y_train, X_test, y_test


def makeDataset(X_train, y_train, X_test, y_test, batch_size=32):
    train_dataset = MyDataset(X_train, y_train)
    test_dataset = MyDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=False)    

    return train_loader, test_loader