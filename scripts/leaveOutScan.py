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
# %%
random.seed(1234)
random.shuffle(scans)
holdout = [ 'esamInit-esamPos-0.5umPerPixel-Scan3', 
            'esamInit-esamNeg-0.5umPerPixel-Scan3',
            'esamInit-esamNeg-1umPerPixel-Scan2',
            'esamInit-esamPos-1umPerPixel-Scan3']
testScans, trainScans = [], []
for scan in scans:

    if scan.__str__() in holdout:

        testScans.append(scan)
    else:
        trainScans.append(scan)
n = 0
for scan in scans:
    n += sum(scan.cellSpectra>0)

nTest = 0
for scan in testScans:
    nTest += sum(scan.cellSpectra>0)
# %%
X_train, y_train = [], []
for scan in trainScans:
    isCell = scan.cellSpectra>0
    
